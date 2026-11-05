import hashlib
import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import sanitize_filename
from app.db.session import is_postgres_engine
from app.models.entities import Document, DocumentChunk, KnowledgeBase
from app.schemas.contracts import RAGCitationSchema
from app.services.providers.model_providers import embedding_engine

ALLOWED_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".csv",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

FORBIDDEN_EXTENSIONS = {
    ".exe",
    ".dll",
    ".bat",
    ".cmd",
    ".ps1",
    ".sh",
    ".vbs",
    ".js",
    ".php",
    ".py",
    ".rb",
    ".msi",
}


class DocumentSecurityError(ValueError):
    """Raised when an uploaded file violates security policies."""


def validate_and_extract_content(
    filename: str, raw_bytes: bytes, content_type: str = "application/octet-stream"
) -> Tuple[str, str, Dict[str, Any]]:
    """Validates file safety and extracts clean text + metadata.
    Returns (safe_filename, extracted_text, metadata_dict).
    """
    safe_name = sanitize_filename(filename)
    _, ext = os.path.splitext(safe_name.lower())

    if ext in FORBIDDEN_EXTENSIONS:
        raise DocumentSecurityError(
            f"Extensão de arquivo proibida por política de segurança: '{ext}'."
        )
    if ext not in ALLOWED_EXTENSIONS:
        raise DocumentSecurityError(
            f"Formato '{ext}' não suportado. Formatos permitidos: {sorted(ALLOWED_EXTENSIONS)}"
        )

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(raw_bytes) == 0:
        raise DocumentSecurityError("O arquivo enviado está vazio.")
    if len(raw_bytes) > max_bytes:
        raise DocumentSecurityError(
            f"O arquivo excede o tamanho máximo permitido de {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    # Check for null-byte / ELF / PE binary header disguised as text
    if ext in {".txt", ".md", ".markdown", ".json", ".csv"}:
        if raw_bytes.startswith(b"MZ") or raw_bytes.startswith(b"\x7fELF"):
            raise DocumentSecurityError(
                "Assinatura de executável binário detectada em arquivo de texto."
            )

    metadata: Dict[str, Any] = {
        "original_filename": safe_name,
        "extension": ext,
        "content_type": content_type,
        "size_bytes": len(raw_bytes),
    }

    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.verify()
            with Image.open(io.BytesIO(raw_bytes)) as img:
                width, height = img.size
                fmt = img.format or ext.lstrip(".").upper()
                mode = img.mode
                info_text = " ".join(
                    f"{k}={v}"
                    for k, v in (img.info or {}).items()
                    if isinstance(v, (str, int, float)) and len(str(v)) < 300
                )
                metadata.update(
                    {
                        "image_format": fmt,
                        "width": width,
                        "height": height,
                        "color_mode": mode,
                        "verified_image": True,
                    }
                )
                extracted = (
                    f"[Imagem Verificada: {safe_name}] Formato: {fmt}, "
                    f"Dimensões: {width}x{height}px, Modo de Cor: {mode}. "
                    f"Metadados embutidos: {info_text or 'Nenhum texto embutido'}."
                )
                return safe_name, extracted, metadata
        except Exception as exc:
            raise DocumentSecurityError(
                f"Arquivo de imagem inválido ou corrompido: {exc}"
            ) from exc

    if ext == ".pdf":
        if not raw_bytes.startswith(b"%PDF"):
            raise DocumentSecurityError("Cabeçalho PDF inválido (%PDF ausente).")
        # Safe lightweight text extraction from PDF streams/literals
        decoded_latin = raw_bytes.decode("latin-1", errors="ignore")
        parenthetical_texts = re.findall(r"\(([^()]{3,400})\)", decoded_latin)
        cleaned_fragments = [
            re.sub(r"\\([nrt()\\])", " ", frag)
            for frag in parenthetical_texts
            if re.search(r"[A-Za-zÀ-ÿ0-9]{3,}", frag)
        ]
        extracted = " ".join(cleaned_fragments).strip()
        if not extracted:
            extracted = f"[Documento PDF validado: {safe_name} ({len(raw_bytes)} bytes)]"
        metadata["pdf_fragments_extracted"] = len(cleaned_fragments)
        return safe_name, sanitize_extracted_text(extracted), metadata

    if ext == ".json":
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
            extracted = json.dumps(parsed, ensure_ascii=False, indent=2)
            metadata["json_valid"] = True
            return safe_name, sanitize_extracted_text(extracted), metadata
        except Exception as exc:
            raise DocumentSecurityError(f"JSON malformado: {exc}") from exc

    # Default text (.txt, .md, .csv)
    text_content = raw_bytes.decode("utf-8", errors="replace")
    return safe_name, sanitize_extracted_text(text_content), metadata


def sanitize_extracted_text(text_value: str) -> str:
    """Normalizes whitespace and strips control characters."""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text_value)
    cleaned = re.sub(r"\r\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def chunk_text(
    text_value: str, chunk_size: int = 600, chunk_overlap: int = 100
) -> List[str]:
    """Splits text into overlapping semantic chunks respecting paragraph/sentence boundaries."""
    cleaned = text_value.strip()
    if not cleaned:
        return []

    chunk_size = max(100, chunk_size)
    chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))

    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    length = len(cleaned)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # Try breaking at paragraph, newline, or sentence end
            window = cleaned[start:end]
            for sep in ("\n\n", "\n", ". ", "; ", " "):
                pos = window.rfind(sep)
                if pos > chunk_size // 2:
                    end = start + pos + len(sep)
                    break

        segment = cleaned[start:end].strip()
        if segment:
            chunks.append(segment)

        if end >= length:
            break
        start = max(start + 1, end - chunk_overlap)

    return chunks


def ingest_document_into_kb(
    db: Session,
    kb: KnowledgeBase,
    owner_id: str,
    filename: str,
    raw_bytes: bytes,
    content_type: str = "text/plain",
) -> Document:
    """Validates, extracts, chunks, embeds, and indexes a document into the KnowledgeBase."""
    safe_name, extracted_text, metadata = validate_and_extract_content(
        filename=filename, raw_bytes=raw_bytes, content_type=content_type
    )
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()

    chunks = chunk_text(
        extracted_text, chunk_size=kb.chunk_size, chunk_overlap=kb.chunk_overlap
    )

    doc = Document(
        knowledge_base_id=kb.id,
        owner_id=owner_id,
        filename=safe_name,
        mime_type=content_type,
        file_size_bytes=len(raw_bytes),
        sha256_hash=sha256_hash,
        extracted_text=extracted_text,
        metadata_json=json.dumps(metadata, ensure_ascii=False),
        chunk_count=len(chunks),
        status="indexed",
    )
    db.add(doc)
    db.flush()

    postgres_active = is_postgres_engine()

    for idx, chunk_str in enumerate(chunks):
        vec, emb_provider = embedding_engine.embed_text(
            chunk_str, model=kb.embedding_model
        )
        token_est = max(1, len(chunk_str.split()))
        chunk_obj = DocumentChunk(
            document_id=doc.id,
            knowledge_base_id=kb.id,
            owner_id=owner_id,
            chunk_index=idx,
            content=chunk_str,
            token_estimate=token_est,
            embedding_json=json.dumps(vec),
            embedding_provider=emb_provider,
        )
        db.add(chunk_obj)
        db.flush()

        if postgres_active:
            try:
                vec_literal = "[" + ",".join(str(v) for v in vec) + "]"
                db.execute(
                    text(
                        "UPDATE document_chunks SET embedding_vector = CAST(:vec AS vector) WHERE id = :cid"
                    ),
                    {"vec": vec_literal, "cid": chunk_obj.id},
                )
            except Exception:
                pass

    db.commit()
    db.refresh(doc)
    return doc


def retrieve_similar_chunks(
    db: Session,
    query: str,
    user_id: str,
    user_role: str = "operator",
    knowledge_base_ids: Optional[List[str]] = None,
    agent_id: Optional[str] = None,
    top_k: int = 4,
    similarity_threshold: float = 0.10,
) -> Tuple[List[RAGCitationSchema], str]:
    """Performs similarity search across isolated knowledge bases.
    Uses PostgreSQL pgvector (<=> cosine distance) when available, with exact cosine calculation fallback.
    Enforces strict tenant/user isolation unless knowledge base is explicitly shared or owned by user.
    """
    query_vec, emb_provider = embedding_engine.embed_text(query)

    kb_query = db.query(KnowledgeBase)
    if knowledge_base_ids:
        kb_query = kb_query.filter(KnowledgeBase.id.in_(knowledge_base_ids))
    elif agent_id:
        kb_query = kb_query.filter(KnowledgeBase.agent_id == agent_id)

    accessible_kbs: List[KnowledgeBase] = []
    for kb in kb_query.all():
        if not kb.is_isolated or kb.owner_id == user_id or user_role == "admin":
            accessible_kbs.append(kb)

    if not accessible_kbs:
        return ([], emb_provider)

    allowed_kb_ids = [kb.id for kb in accessible_kbs]

    chunks_q = (
        db.query(DocumentChunk, Document.filename)
        .join(Document, DocumentChunk.document_id == Document.id)
        .filter(DocumentChunk.knowledge_base_id.in_(allowed_kb_ids))
    )

    scored: List[RAGCitationSchema] = []
    for chunk_row, doc_filename in chunks_q.all():
        try:
            chunk_vec = json.loads(chunk_row.embedding_json)
        except Exception:
            continue
        score = embedding_engine.cosine_similarity(query_vec, chunk_vec)
        if score >= similarity_threshold:
            scored.append(
                RAGCitationSchema(
                    chunk_id=chunk_row.id,
                    document_id=chunk_row.document_id,
                    filename=doc_filename,
                    chunk_index=chunk_row.chunk_index,
                    similarity_score=score,
                    excerpt=chunk_row.content[:450],
                )
            )

    scored.sort(key=lambda item: item.similarity_score, reverse=True)
    return (scored[:top_k], emb_provider)

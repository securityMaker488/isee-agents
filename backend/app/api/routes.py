import json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_roles
from app.core.config import settings
from app.core.security import create_access_token, hash_password, verify_password
from app.db.session import ensure_pgvector_extension, get_db, is_postgres_engine
from app.models.entities import (
    Agent,
    AuditLog,
    Conversation,
    Document,
    DocumentChunk,
    KnowledgeBase,
    Message,
    ModelConfig,
    User,
)
from app.schemas.contracts import (
    AgentCreate,
    AgentExportBundle,
    AgentRead,
    AgentUpdate,
    AgentVersionRead,
    AuditLogRead,
    ChatMessageRead,
    ChatRequest,
    ChatResponse,
    ConversationRead,
    DocumentRead,
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    ModelConfigCreate,
    ModelConfigRead,
    OllamaStatusSchema,
    RAGSearchRequest,
    RAGSearchResponse,
    SystemDashboardSchema,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserRead,
)
from app.services.agent_registry import (
    agent_model_to_read,
    create_or_update_agent,
    export_agents_bundle,
    list_agent_versions,
    update_existing_agent,
)
from app.services.orchestrator import message_model_to_read, orchestrate_chat_turn
from app.services.providers.model_providers import provider_router
from app.services.rag.pipeline import (
    DocumentSecurityError,
    ingest_document_into_kb,
    retrieve_similar_chunks,
)

api_router = APIRouter(prefix="/api/v1")


# ==============================================================================
# 1. AUTHENTICATION & USERS
# ==============================================================================
@api_router.post(
    "/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED
)
def register_user(payload: UserCreate, db: Session = Depends(get_db)) -> TokenResponse:
    existing = db.query(User).filter(User.email == payload.email.lower()).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="E-mail já cadastrado na plataforma.",
        )

    user = User(
        email=payload.email.lower(),
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
        role=payload.role,
        is_active=True,
    )
    db.add(user)
    db.add(
        AuditLog(
            user_id=user.id,
            event_type="USER_REGISTERED",
            severity="INFO",
            resource=f"user/{user.email}",
            details_json=json.dumps({"role": user.role}),
        )
    )
    db.commit()
    db.refresh(user)

    token = create_access_token(subject=user.id, role=user.role)
    return TokenResponse(access_token=token, user=UserRead.model_validate(user))


@api_router.post("/auth/login", response_model=TokenResponse)
def login_user(payload: UserLogin, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="E-mail ou senha inválidos.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Conta de usuário desativada.",
        )

    token = create_access_token(subject=user.id, role=user.role)
    return TokenResponse(access_token=token, user=UserRead.model_validate(user))


@api_router.get("/auth/me", response_model=UserRead)
def get_me(current_user: User = Depends(get_current_user)) -> UserRead:
    return UserRead.model_validate(current_user)


@api_router.get("/auth/users", response_model=List[UserRead])
def list_users(
    db: Session = Depends(get_db),
    _: User = Depends(require_roles(["admin"])),
) -> List[UserRead]:
    users = db.query(User).order_by(User.created_at.asc()).all()
    return [UserRead.model_validate(u) for u in users]


# ==============================================================================
# 2. SYSTEM DASHBOARD, OLLAMA STATUS & AUDIT LOGS
# ==============================================================================
@api_router.get("/system/ollama", response_model=OllamaStatusSchema)
def get_ollama_status(
    _: User = Depends(get_current_user),
) -> OllamaStatusSchema:
    return provider_router.ollama.check_status()


@api_router.get("/system/dashboard", response_model=SystemDashboardSchema)
def get_system_dashboard(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> SystemDashboardSchema:
    ollama_st = provider_router.ollama.check_status()
    total_agents = db.query(Agent).count()
    demo_agents_count = db.query(Agent).filter(Agent.is_demo.is_(True)).count()
    slots_integrated = (
        db.query(Agent)
        .filter(
            Agent.slot_number.is_not(None),
            Agent.is_original_slot_integrated.is_(True),
        )
        .count()
    )
    total_kbs = db.query(KnowledgeBase).count()
    total_docs = db.query(Document).count()
    total_chunks = db.query(DocumentChunk).count()
    total_convs = db.query(Conversation).count()
    total_msgs = db.query(Message).count()
    avg_lat = (
        db.query(func.avg(Message.latency_ms))
        .filter(Message.role == "assistant", Message.latency_ms > 0)
        .scalar()
        or 0.0
    )

    pg_active = is_postgres_engine()
    pgvec_ok = ensure_pgvector_extension(db) if pg_active else False

    return SystemDashboardSchema(
        platform_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        database_engine="PostgreSQL" if pg_active else "SQLite (Local Runtime)",
        pgvector_enabled=pgvec_ok,
        ollama_status=ollama_st,
        total_agents=total_agents,
        demo_agents_count=demo_agents_count,
        original_slots_total=12,
        original_slots_integrated=slots_integrated,
        original_slots_awaiting=max(0, 12 - slots_integrated),
        total_knowledge_bases=total_kbs,
        total_documents=total_docs,
        total_vector_chunks=total_chunks,
        total_conversations=total_convs,
        total_messages=total_msgs,
        avg_latency_ms=round(float(avg_lat), 2),
    )


@api_router.get("/system/audit-logs", response_model=List[AuditLogRead])
def list_audit_logs(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    _: User = Depends(require_roles(["admin", "operator"])),
) -> List[AuditLogRead]:
    rows = (
        db.query(AuditLog)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        AuditLogRead(
            id=r.id,
            user_id=r.user_id,
            event_type=r.event_type,
            severity=r.severity,
            resource=r.resource,
            details=json.loads(r.details_json or "{}"),
            created_at=r.created_at,
        )
        for r in rows
    ]


# ==============================================================================
# 3. AGENTS CATALOG, VERSIONING & IMPORT/EXPORT (12 SLOTS + CUSTOM)
# ==============================================================================
@api_router.get("/agents", response_model=List[AgentRead])
def list_agents(
    include_slots: bool = Query(default=True),
    only_slots: bool = Query(default=False),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> List[AgentRead]:
    q = db.query(Agent)
    if only_slots:
        q = q.filter(Agent.slot_number.is_not(None))
    elif not include_slots:
        q = q.filter(Agent.slot_number.is_(None))

    rows = q.order_by(Agent.slot_number.asc(), Agent.created_at.asc()).all()
    return [agent_model_to_read(r) for r in rows]


@api_router.get("/agents/export/bundle", response_model=AgentExportBundle)
def export_agents(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> AgentExportBundle:
    return export_agents_bundle(db)


@api_router.post("/agents/import/bundle", response_model=List[AgentRead])
def import_agents(
    bundle: AgentExportBundle,
    overwrite: bool = Query(default=True),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "operator"])),
) -> List[AgentRead]:
    imported: List[AgentRead] = []
    for agent_def in bundle.agents:
        obj = create_or_update_agent(
            db=db,
            payload=agent_def,
            user_id=current_user.id,
            allow_overwrite=overwrite,
        )
        imported.append(agent_model_to_read(obj))

    db.add(
        AuditLog(
            user_id=current_user.id,
            event_type="AGENTS_BUNDLE_IMPORTED",
            severity="INFO",
            resource="agents/import",
            details_json=json.dumps({"count": len(imported)}),
        )
    )
    db.commit()
    return imported


@api_router.get("/agents/{agent_id}", response_model=AgentRead)
def get_agent(
    agent_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> AgentRead:
    ag = db.query(Agent).filter(Agent.id == agent_id).first()
    if not ag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agente não encontrado."
        )
    return agent_model_to_read(ag)


@api_router.post(
    "/agents", response_model=AgentRead, status_code=status.HTTP_201_CREATED
)
def create_agent(
    payload: AgentCreate,
    overwrite: bool = Query(default=False),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "operator"])),
) -> AgentRead:
    try:
        ag = create_or_update_agent(
            db=db,
            payload=payload,
            user_id=current_user.id,
            allow_overwrite=overwrite,
        )
        return agent_model_to_read(ag)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


@api_router.put("/agents/{agent_id}", response_model=AgentRead)
def update_agent(
    agent_id: str,
    patch: AgentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "operator"])),
) -> AgentRead:
    ag = db.query(Agent).filter(Agent.id == agent_id).first()
    if not ag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agente não encontrado."
        )
    updated = update_existing_agent(db, ag, patch, user_id=current_user.id)
    return agent_model_to_read(updated)


@api_router.delete("/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_agent(
    agent_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(require_roles(["admin"])),
) -> None:
    ag = db.query(Agent).filter(Agent.id == agent_id).first()
    if not ag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agente não encontrado."
        )
    if ag.slot_number is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Os 12 slots estruturais reservados não devem ser removidos; atualize ou redefina a definição do slot.",
        )
    db.delete(ag)
    db.commit()


@api_router.get("/agents/{agent_id}/versions", response_model=List[AgentVersionRead])
def get_agent_versions(
    agent_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> List[AgentVersionRead]:
    return list_agent_versions(db, agent_id)


@api_router.post("/agents/{agent_id}/evaluate")
def evaluate_agent_test_cases(
    agent_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "operator"])),
) -> Dict[str, Any]:
    ag = db.query(Agent).filter(Agent.id == agent_id).first()
    if not ag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agente não encontrado."
        )
    spec = agent_model_to_read(ag)
    results = []
    passed_count = 0

    test_cases = spec.test_cases or []
    for tc in test_cases:
        resp = orchestrate_chat_turn(
            db=db,
            user_id=current_user.id,
            user_role=current_user.role,
            request=ChatRequest(
                agent_id=ag.id,
                message=tc.input_prompt,
                use_rag=spec.rag_config.enabled,
            ),
        )
        output_lower = resp.assistant_message.content.lower()
        missing = [
            kw for kw in tc.expected_keywords if kw.lower() not in output_lower
        ]
        forbidden_hit = [
            kw for kw in tc.forbidden_keywords if kw.lower() in output_lower
        ]
        passed = len(missing) == 0 and len(forbidden_hit) == 0
        if passed:
            passed_count += 1
        results.append(
            {
                "test_name": tc.name,
                "passed": passed,
                "latency_ms": resp.assistant_message.latency_ms,
                "provider_used": resp.assistant_message.provider_used,
                "is_demo_response": resp.assistant_message.is_demo_response,
                "citations_count": len(resp.assistant_message.citations),
                "missing_keywords": missing,
                "forbidden_triggered": forbidden_hit,
            }
        )

    total = len(test_cases)
    return {
        "agent_id": ag.id,
        "agent_name": ag.name,
        "version": ag.version,
        "total_test_cases": total,
        "passed_test_cases": passed_count,
        "pass_rate": round(passed_count / total, 4) if total > 0 else 1.0,
        "results": results,
    }


# ==============================================================================
# 4. MODEL CONFIGURATIONS & LORA ADAPTER REGISTRY
# ==============================================================================
@api_router.get("/models", response_model=List[ModelConfigRead])
def list_models(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> List[ModelConfigRead]:
    rows = db.query(ModelConfig).order_by(ModelConfig.created_at.asc()).all()
    return [ModelConfigRead.model_validate(r) for r in rows]


@api_router.post(
    "/models", response_model=ModelConfigRead, status_code=status.HTTP_201_CREATED
)
def create_model_config(
    payload: ModelConfigCreate,
    db: Session = Depends(get_db),
    _: User = Depends(require_roles(["admin", "operator"])),
) -> ModelConfigRead:
    existing = db.query(ModelConfig).filter(ModelConfig.name == payload.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Configuração de modelo com este nome já existe.",
        )
    obj = ModelConfig(**payload.model_dump(), is_active=True)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return ModelConfigRead.model_validate(obj)


@api_router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model_config(
    model_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(require_roles(["admin"])),
) -> None:
    obj = db.query(ModelConfig).filter(ModelConfig.id == model_id).first()
    if not obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Configuração de modelo não encontrada.",
        )
    db.delete(obj)
    db.commit()


# ==============================================================================
# 5. KNOWLEDGE BASES, SECURE DOCUMENT/IMAGE INGESTION & RAG
# ==============================================================================
class TextIngestPayload(BaseModel):
    filename: str = Field(..., min_length=2, max_length=255)
    content: str = Field(..., min_length=5)


@api_router.get("/knowledge/bases", response_model=List[KnowledgeBaseRead])
def list_knowledge_bases(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[KnowledgeBaseRead]:
    q = db.query(KnowledgeBase)
    if current_user.role != "admin":
        q = q.filter(
            (KnowledgeBase.owner_id == current_user.id)
            | (KnowledgeBase.is_isolated.is_(False))
        )
    rows = q.order_by(KnowledgeBase.created_at.desc()).all()

    result: List[KnowledgeBaseRead] = []
    for kb in rows:
        doc_count = (
            db.query(Document).filter(Document.knowledge_base_id == kb.id).count()
        )
        chunk_count = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.knowledge_base_id == kb.id)
            .count()
        )
        result.append(
            KnowledgeBaseRead(
                id=kb.id,
                name=kb.name,
                description=kb.description,
                domain=kb.domain,
                owner_id=kb.owner_id,
                agent_id=kb.agent_id,
                chunk_size=kb.chunk_size,
                chunk_overlap=kb.chunk_overlap,
                embedding_model=kb.embedding_model,
                is_isolated=kb.is_isolated,
                document_count=doc_count,
                chunk_count=chunk_count,
                created_at=kb.created_at,
            )
        )
    return result


@api_router.post(
    "/knowledge/bases",
    response_model=KnowledgeBaseRead,
    status_code=status.HTTP_201_CREATED,
)
def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "operator"])),
) -> KnowledgeBaseRead:
    kb = KnowledgeBase(
        name=payload.name,
        description=payload.description,
        domain=payload.domain,
        owner_id=current_user.id,
        agent_id=payload.agent_id,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        embedding_model=payload.embedding_model,
        is_isolated=payload.is_isolated,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return KnowledgeBaseRead(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        domain=kb.domain,
        owner_id=kb.owner_id,
        agent_id=kb.agent_id,
        chunk_size=kb.chunk_size,
        chunk_overlap=kb.chunk_overlap,
        embedding_model=kb.embedding_model,
        is_isolated=kb.is_isolated,
        document_count=0,
        chunk_count=0,
        created_at=kb.created_at,
    )


@api_router.get(
    "/knowledge/bases/{kb_id}/documents", response_model=List[DocumentRead]
)
def list_kb_documents(
    kb_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[DocumentRead]:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Base de conhecimento não encontrada.",
        )
    if kb.is_isolated and kb.owner_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acesso negado a esta base de conhecimento isolada.",
        )
    docs = (
        db.query(Document)
        .filter(Document.knowledge_base_id == kb_id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return [
        DocumentRead(
            id=d.id,
            knowledge_base_id=d.knowledge_base_id,
            owner_id=d.owner_id,
            filename=d.filename,
            mime_type=d.mime_type,
            file_size_bytes=d.file_size_bytes,
            sha256_hash=d.sha256_hash,
            extracted_preview=d.extracted_text[:300],
            metadata=json.loads(d.metadata_json or "{}"),
            chunk_count=d.chunk_count,
            status=d.status,
            created_at=d.created_at,
        )
        for d in docs
    ]


@api_router.post(
    "/knowledge/bases/{kb_id}/upload",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document_to_kb(
    kb_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "operator"])),
) -> DocumentRead:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Base de conhecimento não encontrada.",
        )
    if kb.is_isolated and kb.owner_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permissão negada para enviar documentos a esta base isolada.",
        )

    raw_bytes = await file.read()
    try:
        doc = ingest_document_into_kb(
            db=db,
            kb=kb,
            owner_id=current_user.id,
            filename=file.filename or "document.txt",
            raw_bytes=raw_bytes,
            content_type=file.content_type or "application/octet-stream",
        )
    except DocumentSecurityError as exc:
        db.add(
            AuditLog(
                user_id=current_user.id,
                event_type="UPLOAD_SECURITY_REJECTED",
                severity="WARNING",
                resource=f"kb/{kb.id}",
                details_json=json.dumps(
                    {"filename": file.filename, "reason": str(exc)},
                    ensure_ascii=False,
                ),
            )
        )
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return DocumentRead(
        id=doc.id,
        knowledge_base_id=doc.knowledge_base_id,
        owner_id=doc.owner_id,
        filename=doc.filename,
        mime_type=doc.mime_type,
        file_size_bytes=doc.file_size_bytes,
        sha256_hash=doc.sha256_hash,
        extracted_preview=doc.extracted_text[:300],
        metadata=json.loads(doc.metadata_json or "{}"),
        chunk_count=doc.chunk_count,
        status=doc.status,
        created_at=doc.created_at,
    )


@api_router.post(
    "/knowledge/bases/{kb_id}/ingest-text",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
)
def ingest_text_to_kb(
    kb_id: str,
    payload: TextIngestPayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "operator"])),
) -> DocumentRead:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Base de conhecimento não encontrada.",
        )
    if kb.is_isolated and kb.owner_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permissão negada para indexar textos nesta base isolada.",
        )

    try:
        doc = ingest_document_into_kb(
            db=db,
            kb=kb,
            owner_id=current_user.id,
            filename=payload.filename,
            raw_bytes=payload.content.encode("utf-8"),
            content_type="text/markdown",
        )
    except DocumentSecurityError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return DocumentRead(
        id=doc.id,
        knowledge_base_id=doc.knowledge_base_id,
        owner_id=doc.owner_id,
        filename=doc.filename,
        mime_type=doc.mime_type,
        file_size_bytes=doc.file_size_bytes,
        sha256_hash=doc.sha256_hash,
        extracted_preview=doc.extracted_text[:300],
        metadata=json.loads(doc.metadata_json or "{}"),
        chunk_count=doc.chunk_count,
        status=doc.status,
        created_at=doc.created_at,
    )


@api_router.post("/knowledge/search", response_model=RAGSearchResponse)
def search_knowledge(
    payload: RAGSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RAGSearchResponse:
    citations, emb_provider = retrieve_similar_chunks(
        db=db,
        query=payload.query,
        user_id=current_user.id,
        user_role=current_user.role,
        knowledge_base_ids=payload.knowledge_base_ids or None,
        top_k=payload.top_k,
        similarity_threshold=payload.similarity_threshold,
    )
    return RAGSearchResponse(
        query=payload.query,
        embedding_provider=emb_provider,
        results=citations,
    )


# ==============================================================================
# 6. CHAT & CONVERSATION HISTORY
# ==============================================================================
@api_router.post("/chat/completions", response_model=ChatResponse)
def create_chat_completion(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    try:
        return orchestrate_chat_turn(
            db=db,
            user_id=current_user.id,
            user_role=current_user.role,
            request=payload,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


@api_router.get("/chat/conversations", response_model=List[ConversationRead])
def list_conversations(
    agent_id: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[ConversationRead]:
    q = db.query(Conversation).filter(Conversation.user_id == current_user.id)
    if agent_id:
        q = q.filter(Conversation.agent_id == agent_id)
    convs = q.order_by(Conversation.updated_at.desc()).all()

    result: List[ConversationRead] = []
    for c in convs:
        ag = db.query(Agent).filter(Agent.id == c.agent_id).first()
        m_count = db.query(Message).filter(Message.conversation_id == c.id).count()
        result.append(
            ConversationRead(
                id=c.id,
                title=c.title,
                user_id=c.user_id,
                agent_id=c.agent_id,
                agent_name=ag.name if ag else None,
                message_count=m_count,
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
        )
    return result


@api_router.get(
    "/chat/conversations/{conversation_id}/messages",
    response_model=List[ChatMessageRead],
)
def list_conversation_messages(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[ChatMessageRead]:
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversa não encontrada ou acesso negado.",
        )
    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return [message_model_to_read(m) for m in msgs]


@api_router.delete(
    "/chat/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT
)
def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversa não encontrada.",
        )
    db.delete(conv)
    db.commit()

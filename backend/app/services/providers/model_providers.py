from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx

from app.core.config import settings
from app.schemas.contracts import OllamaStatusSchema, RAGCitationSchema


@dataclass
class GenerationResult:
    content: str
    provider_used: str
    model_used: str
    is_demo_response: bool
    latency_ms: float
    tokens_generated: int
    tokens_per_second: float


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_chat(
        self,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        rag_citations: Optional[List[RAGCitationSchema]] = None,
        agent_name: str = "Agent",
        agent_domain: str = "General",
        adapter_info: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        raise NotImplementedError


class OllamaProvider(BaseLLMProvider):
    """Real HTTP integration with local or remote Ollama runtime."""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.OLLAMA_TIMEOUT_SECONDS

    def check_status(self) -> OllamaStatusSchema:
        start = time.perf_counter()
        try:
            with httpx.Client(timeout=2.5) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
                data = resp.json()
                models = [
                    m.get("name", "")
                    for m in data.get("models", [])
                    if m.get("name")
                ]
                return OllamaStatusSchema(
                    reachable=True,
                    base_url=self.base_url,
                    latency_ms=latency_ms,
                    installed_models=models,
                    active_mode="ollama_live",
                    error_detail=None,
                )
        except Exception as exc:
            return OllamaStatusSchema(
                reachable=False,
                base_url=self.base_url,
                latency_ms=None,
                installed_models=[],
                active_mode="demo_fallback" if settings.ALLOW_DEMO_FALLBACK else "offline",
                error_detail=str(exc),
            )

    def generate_chat(
        self,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        rag_citations: Optional[List[RAGCitationSchema]] = None,
        agent_name: str = "Agent",
        agent_domain: str = "General",
        adapter_info: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        start = time.perf_counter()
        ollama_messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        ollama_messages.extend(messages)

        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        content = data.get("message", {}).get("content", "").strip()
        eval_count = int(data.get("eval_count") or max(1, len(content.split())))
        eval_duration_ns = int(data.get("eval_duration") or 0)
        if eval_duration_ns > 0:
            tps = round(eval_count / (eval_duration_ns / 1e9), 2)
        else:
            tps = round(eval_count / max(elapsed_ms / 1000.0, 0.001), 2)

        return GenerationResult(
            content=content,
            provider_used="ollama",
            model_used=model,
            is_demo_response=False,
            latency_ms=elapsed_ms,
            tokens_generated=eval_count,
            tokens_per_second=tps,
        )


class DemoProvider(BaseLLMProvider):
    """Explicitly labeled DEMO provider used when Ollama is offline or when
    testing demonstrative agents before real model runtime is attached.
    """

    def generate_chat(
        self,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        rag_citations: Optional[List[RAGCitationSchema]] = None,
        agent_name: str = "Agent",
        agent_domain: str = "General",
        adapter_info: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        start = time.perf_counter()
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        adapter_badge = ""
        if adapter_info and adapter_info.get("enabled"):
            method = str(adapter_info.get("method", "lora")).upper()
            rank = adapter_info.get("rank", 16)
            alpha = adapter_info.get("lora_alpha", 32)
            adapter_badge = f" | Adaptador Configurado: `{method} (r={rank}, alpha={alpha})`"

        lines = [
            f"**[MODO DEMO — RESPOSTA SIMULADA]**",
            f"**Agente:** {agent_name} (`{agent_domain}`) | **Modelo Alvo:** `{model}`{adapter_badge}",
            "",
            f"Recebi sua solicitação: *\"{last_user_msg[:280]}\"*.",
        ]

        if rag_citations:
            lines.append("")
            lines.append("### Contexto Recuperado via Pipeline RAG")
            for idx, cit in enumerate(rag_citations, start=1):
                lines.append(
                    f"{idx}. **Fonte:** `{cit.filename}` (Chunk #{cit.chunk_index}, Similaridade: `{cit.similarity_score:.3f}`)\n"
                    f"   > {cit.excerpt[:260]}"
                )
            lines.append("")
            lines.append(
                "**Síntese Contextualizada (DEMO):** Com base estrita nos trechos indexados na base de conhecimento deste agente, "
                "as informações acima respondem diretamente aos termos consultados. Para geração neural completa em produção, "
                "inicie o runtime local do **Ollama** (`ollama serve`) com o modelo configurado."
            )
        else:
            lines.append("")
            lines.append(
                "**Análise Técnica (DEMO):** Esta resposta foi processada pelo provedor demonstrativo (`DemoProvider`) porque o "
                "agente está em modo `DEMO` ou o servidor local **Ollama** não está acessível no momento em "
                f"`{settings.OLLAMA_BASE_URL}`. Os parâmetros ativos de inferência (`temperature={temperature}`, `top_p={top_p}`, "
                f"`max_tokens={max_tokens}`) e as restrições do System Prompt foram validados pelo orquestrador."
            )

        content = "\n".join(lines)
        elapsed_ms = round(max((time.perf_counter() - start) * 1000.0, 1.25), 2)
        tokens_est = max(12, len(content.split()))
        tps = round(tokens_est / max(elapsed_ms / 1000.0, 0.001), 2)

        return GenerationResult(
            content=content,
            provider_used="demo",
            model_used=f"{model} [DEMO]",
            is_demo_response=True,
            latency_ms=elapsed_ms,
            tokens_generated=tokens_est,
            tokens_per_second=tps,
        )


class ProviderRouter:
    """Routes inference requests to Ollama (live) or DemoProvider (explicit fallback)."""

    def __init__(self) -> None:
        self.ollama = OllamaProvider()
        self.demo = DemoProvider()

    def generate(
        self,
        requested_provider: str,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        rag_citations: Optional[List[RAGCitationSchema]] = None,
        agent_name: str = "Agent",
        agent_domain: str = "General",
        adapter_info: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        if requested_provider == "demo":
            return self.demo.generate_chat(
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                rag_citations=rag_citations,
                agent_name=agent_name,
                agent_domain=agent_domain,
                adapter_info=adapter_info,
            )

        if requested_provider in ("ollama", "huggingface"):
            try:
                return self.ollama.generate_chat(
                    model=model,
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    rag_citations=rag_citations,
                    agent_name=agent_name,
                    agent_domain=agent_domain,
                    adapter_info=adapter_info,
                )
            except Exception:
                if settings.ALLOW_DEMO_FALLBACK:
                    return self.demo.generate_chat(
                        model=model,
                        system_prompt=system_prompt,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        rag_citations=rag_citations,
                        agent_name=agent_name,
                        agent_domain=agent_domain,
                        adapter_info=adapter_info,
                    )
                raise

        return self.demo.generate_chat(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            rag_citations=rag_citations,
            agent_name=agent_name,
            agent_domain=agent_domain,
            adapter_info=adapter_info,
        )


class EmbeddingEngine:
    """Generates dense vector embeddings via Ollama (/api/embeddings) when reachable,
    or deterministic L2-normalized semantic token/n-gram vectors (384-d) in offline/DEMO mode.
    """

    def __init__(self, dimension: int = settings.EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self._offline_until: float = 0.0

    def embed_text(
        self, text: str, model: Optional[str] = None
    ) -> Tuple[List[float], str]:
        target_model = model or settings.OLLAMA_EMBEDDING_MODEL
        now = time.monotonic()
        if now >= self._offline_until:
            try:
                timeout_cfg = httpx.Timeout(2.0, connect=0.35)
                with httpx.Client(timeout=timeout_cfg) as client:
                    resp = client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": target_model, "prompt": text},
                    )
                    if resp.status_code == 200:
                        vec = resp.json().get("embedding")
                        if isinstance(vec, list) and len(vec) > 0:
                            return (self._normalize_and_fit(vec), f"ollama:{target_model}")
            except Exception:
                self._offline_until = now + 15.0

        return (self._deterministic_semantic_embedding(text), "deterministic-local-384d [DEMO]")

    def _normalize_and_fit(self, vec: List[float]) -> List[float]:
        if len(vec) >= self.dimension:
            sliced = [float(x) for x in vec[: self.dimension]]
        else:
            sliced = [float(x) for x in vec] + [0.0] * (self.dimension - len(vec))
        norm = math.sqrt(sum(v * v for v in sliced)) or 1.0
        return [round(v / norm, 6) for v in sliced]

    def _deterministic_semantic_embedding(self, text: str) -> List[float]:
        """Produces a 384-d L2-normalized vector based on word unigrams, bigrams,
        and character trigrams so semantically/lexically related passages have high cosine similarity.
        """
        vec = [0.0] * self.dimension
        cleaned = re.sub(r"[^a-zA-Z0-9áéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ_]+", " ", text.lower())
        tokens = [t for t in cleaned.split() if len(t) > 1]
        if not tokens:
            tokens = ["empty"]

        features: List[Tuple[str, float]] = []
        for idx, tok in enumerate(tokens):
            features.append((f"w:{tok}", 2.0))
            if idx + 1 < len(tokens):
                features.append((f"bg:{tok}_{tokens[idx + 1]}", 1.5))
            for c_i in range(max(1, len(tok) - 2)):
                features.append((f"tri:{tok[c_i:c_i + 3]}", 0.6))

        for feat, weight in features:
            digest = hashlib.sha256(feat.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[bucket] += sign * weight

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 6) for v in vec]

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        n = min(len(vec_a), len(vec_b))
        dot = sum(vec_a[i] * vec_b[i] for i in range(n))
        norm_a = math.sqrt(sum(vec_a[i] * vec_a[i] for i in range(n))) or 1.0
        norm_b = math.sqrt(sum(vec_b[i] * vec_b[i] for i in range(n))) or 1.0
        return round(float(dot / (norm_a * norm_b)), 4)


provider_router = ProviderRouter()
embedding_engine = EmbeddingEngine()

from datetime import datetime, timezone
import json
from typing import Dict, List, Optional
from sqlalchemy.orm import Session

from app.core.security import inspect_prompt_security
from app.models.entities import Agent, AuditLog, Conversation, Message
from app.schemas.contracts import (
    ChatMessageRead,
    ChatRequest,
    ChatResponse,
    ConversationRead,
    RAGCitationSchema,
)
from app.services.agent_registry import agent_model_to_read
from app.services.providers.model_providers import provider_router
from app.services.rag.pipeline import retrieve_similar_chunks


def message_model_to_read(msg: Message) -> ChatMessageRead:
    citations_raw = json.loads(msg.citations_json or "[]")
    flags_raw = json.loads(msg.security_flags_json or "[]")
    return ChatMessageRead(
        id=msg.id,
        conversation_id=msg.conversation_id,
        role=msg.role,
        content=msg.content,
        provider_used=msg.provider_used,
        model_used=msg.model_used,
        is_demo_response=msg.is_demo_response,
        latency_ms=msg.latency_ms,
        tokens_generated=msg.tokens_generated,
        tokens_per_second=msg.tokens_per_second,
        citations=[RAGCitationSchema(**c) for c in citations_raw],
        security_flags=flags_raw,
        created_at=msg.created_at,
    )


def orchestrate_chat_turn(
    db: Session,
    user_id: str,
    user_role: str,
    request: ChatRequest,
) -> ChatResponse:
    """Executes a full multi-agent orchestration turn:
    1. Resolves target Agent and validates security guardrails on user input.
    2. Retrieves or creates persistent Conversation.
    3. Executes isolated RAG retrieval if enabled for the agent/request.
    4. Builds augmented system prompt with citations and dispatches to ProviderRouter (Ollama or explicit DEMO).
    5. Persists user message, assistant message, and audit logs when security flags trigger.
    """
    agent = db.query(Agent).filter(Agent.id == request.agent_id).first()
    if not agent:
        raise ValueError(f"Agente '{request.agent_id}' não encontrado.")

    agent_spec = agent_model_to_read(agent)

    # Conversation resolution
    if request.conversation_id:
        conv = (
            db.query(Conversation)
            .filter(
                Conversation.id == request.conversation_id,
                Conversation.user_id == user_id,
            )
            .first()
        )
        if not conv:
            raise ValueError("Conversa não encontrada ou acesso negado para este usuário.")
    else:
        title_preview = request.message.strip()[:60]
        conv = Conversation(
            title=title_preview or f"Conversa com {agent.name}",
            user_id=user_id,
            agent_id=agent.id,
        )
        db.add(conv)
        db.flush()

    # Security guardrail inspection
    is_safe, security_flags = inspect_prompt_security(
        request.message, agent_spec.security_constraints
    )

    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=request.message,
        provider_used="client",
        model_used=agent_spec.base_model,
        is_demo_response=False,
        latency_ms=0.0,
        tokens_generated=max(1, len(request.message.split())),
        tokens_per_second=0.0,
        citations_json="[]",
        security_flags_json=json.dumps(security_flags),
    )
    db.add(user_msg)
    db.flush()

    if not is_safe:
        db.add(
            AuditLog(
                user_id=user_id,
                event_type="PROMPT_SECURITY_BLOCKED",
                severity="WARNING",
                resource=f"agent/{agent.slug}",
                details_json=json.dumps(
                    {
                        "conversation_id": conv.id,
                        "flags": security_flags,
                        "prompt_excerpt": request.message[:160],
                    },
                    ensure_ascii=False,
                ),
            )
        )
        blocked_text = (
            "### [GUARDRAIL DE SEGURANÇA ATIVADO]\n"
            "A solicitação foi bloqueada pelas políticas de proteção do agente porque acionou "
            f"regras contra **Prompt Injection / Violação de Restrições**: `{', '.join(security_flags)}`.\n\n"
            "O System Prompt, credenciais internas e configurações de isolamento permanecem protegidos."
        )
        assistant_msg = Message(
            conversation_id=conv.id,
            role="assistant",
            content=blocked_text,
            provider_used="security_guardrail",
            model_used=agent_spec.base_model,
            is_demo_response=agent_spec.is_demo,
            latency_ms=0.8,
            tokens_generated=len(blocked_text.split()),
            tokens_per_second=100.0,
            citations_json="[]",
            security_flags_json=json.dumps(security_flags),
        )
        db.add(assistant_msg)
        conv.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(user_msg)
        db.refresh(assistant_msg)
        msg_count = (
            db.query(Message).filter(Message.conversation_id == conv.id).count()
        )
        return ChatResponse(
            conversation=ConversationRead(
                id=conv.id,
                title=conv.title,
                user_id=conv.user_id,
                agent_id=conv.agent_id,
                agent_name=agent.name,
                message_count=msg_count,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
            ),
            user_message=message_model_to_read(user_msg),
            assistant_message=message_model_to_read(assistant_msg),
        )

    # RAG Retrieval
    should_use_rag = (
        request.use_rag if request.use_rag is not None else agent_spec.rag_config.enabled
    )
    citations: List[RAGCitationSchema] = []
    if should_use_rag:
        citations, _ = retrieve_similar_chunks(
            db=db,
            query=request.message,
            user_id=user_id,
            user_role=user_role,
            knowledge_base_ids=agent_spec.rag_config.knowledge_base_ids or None,
            agent_id=agent.id if not agent_spec.rag_config.knowledge_base_ids else None,
            top_k=agent_spec.rag_config.top_k,
            similarity_threshold=agent_spec.rag_config.similarity_threshold,
        )

    # Build conversation history context
    prior_messages = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id)
        .order_by(Message.created_at.asc())
        .limit(16)
        .all()
    )
    history_payload: List[Dict[str, str]] = [
        {"role": m.role, "content": m.content} for m in prior_messages
    ]

    augmented_system_prompt = agent_spec.system_prompt
    if citations:
        context_blocks = "\n\n".join(
            f"[Fonte {i}: {c.filename} | Chunk #{c.chunk_index} | Score={c.similarity_score}]\n{c.excerpt}"
            for i, c in enumerate(citations, start=1)
        )
        augmented_system_prompt = (
            f"{agent_spec.system_prompt}\n\n"
            "=== CONTEXTO VERIFICADO DA BASE DE CONHECIMENTO (RAG) ===\n"
            f"{context_blocks}\n"
            "=========================================================\n"
            "Utilize as fontes acima para fundamentar sua resposta e referencie os documentos."
        )

    temperature = (
        request.override_temperature
        if request.override_temperature is not None
        else agent_spec.inference_config.temperature
    )
    top_p = (
        request.override_top_p
        if request.override_top_p is not None
        else agent_spec.inference_config.top_p
    )

    gen_result = provider_router.generate(
        requested_provider=agent_spec.provider,
        model=agent_spec.base_model,
        system_prompt=augmented_system_prompt,
        messages=history_payload,
        temperature=temperature,
        top_p=top_p,
        max_tokens=agent_spec.inference_config.max_tokens,
        rag_citations=citations,
        agent_name=agent_spec.name,
        agent_domain=agent_spec.domain,
        adapter_info=agent_spec.lora_adapter.model_dump(),
    )

    assistant_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=gen_result.content,
        provider_used=gen_result.provider_used,
        model_used=gen_result.model_used,
        is_demo_response=gen_result.is_demo_response or agent_spec.is_demo,
        latency_ms=gen_result.latency_ms,
        tokens_generated=gen_result.tokens_generated,
        tokens_per_second=gen_result.tokens_per_second,
        citations_json=json.dumps(
            [c.model_dump() for c in citations], ensure_ascii=False
        ),
        security_flags_json="[]",
    )
    db.add(assistant_msg)
    conv.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user_msg)
    db.refresh(assistant_msg)

    msg_count = db.query(Message).filter(Message.conversation_id == conv.id).count()
    return ChatResponse(
        conversation=ConversationRead(
            id=conv.id,
            title=conv.title,
            user_id=conv.user_id,
            agent_id=conv.agent_id,
            agent_name=agent.name,
            message_count=msg_count,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
        ),
        user_message=message_model_to_read(user_msg),
        assistant_message=message_model_to_read(assistant_msg),
    )

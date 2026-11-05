from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ==============================================================================
# AUTHENTICATION & USERS
# ==============================================================================
class UserCreate(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=255)
    full_name: str = Field(..., min_length=2, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    role: str = Field(default="operator", pattern="^(admin|operator|viewer)$")


class UserLogin(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=255)
    password: str


class UserRead(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead


# ==============================================================================
# MODEL CONFIGURATIONS & LORA ADAPTERS
# ==============================================================================
class ModelConfigCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    provider: str = Field(..., pattern="^(ollama|huggingface|demo)$")
    base_model: str = Field(..., min_length=2, max_length=150)
    version: str = Field(default="1.0.0", max_length=50)
    quantization: Optional[str] = Field(default=None, max_length=50)
    context_window: int = Field(default=8192, ge=512, le=131072)
    default_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    default_top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    default_max_tokens: int = Field(default=1024, ge=32, le=32768)
    adapter_type: Optional[str] = Field(default=None, pattern="^(lora|qlora|none)?$")
    adapter_path: Optional[str] = None
    adapter_rank: Optional[int] = Field(default=None, ge=1, le=256)
    adapter_alpha: Optional[int] = Field(default=None, ge=1, le=512)
    is_demo: bool = False


class ModelConfigRead(ModelConfigCreate):
    id: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==============================================================================
# AGENT SPECIALIZATION (PREPARED FOR 12 ORIGINAL AGENTS + CUSTOM)
# ==============================================================================
class RAGConfigSchema(BaseModel):
    enabled: bool = True
    knowledge_base_ids: List[str] = Field(default_factory=list)
    top_k: int = Field(default=4, ge=1, le=20)
    similarity_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    chunk_size: int = Field(default=600, ge=100, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)


class LoRAAdapterSchema(BaseModel):
    enabled: bool = False
    method: str = Field(default="none", pattern="^(none|lora|qlora)$")
    adapter_name: Optional[str] = None
    adapter_path: Optional[str] = None
    rank: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    quantization_bits: Optional[int] = Field(default=None)


class InferenceConfigSchema(BaseModel):
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=32, le=16384)
    context_window: int = Field(default=8192, ge=512, le=131072)


class AgentTestCaseSchema(BaseModel):
    name: str
    input_prompt: str
    expected_keywords: List[str] = Field(default_factory=list)
    forbidden_keywords: List[str] = Field(default_factory=list)


class AgentCreate(BaseModel):
    slug: str = Field(..., min_length=2, max_length=120)
    slot_number: Optional[int] = Field(default=None, ge=1, le=12)
    name: str = Field(..., min_length=2, max_length=150)
    purpose: str = Field(..., min_length=5)
    domain: str = Field(..., min_length=2, max_length=150)
    base_model: str = Field(default="llama3.2:3b", max_length=150)
    provider: str = Field(default="ollama", pattern="^(ollama|huggingface|demo)$")
    system_prompt: str = Field(..., min_length=10)
    tools: List[str] = Field(default_factory=list)
    knowledge_sources: List[str] = Field(default_factory=list)
    rag_config: RAGConfigSchema = Field(default_factory=RAGConfigSchema)
    lora_adapter: LoRAAdapterSchema = Field(default_factory=LoRAAdapterSchema)
    version: str = Field(default="1.0.0", max_length=40)
    inference_config: InferenceConfigSchema = Field(default_factory=InferenceConfigSchema)
    test_cases: List[AgentTestCaseSchema] = Field(default_factory=list)
    quality_criteria: List[str] = Field(default_factory=list)
    security_constraints: List[str] = Field(default_factory=list)
    status: str = Field(default="ready", pattern="^(ready|awaiting_integration|draft|archived)$")
    is_demo: bool = True
    is_original_slot_integrated: bool = False
    changelog: Optional[str] = "Initial agent registration"


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    purpose: Optional[str] = None
    domain: Optional[str] = None
    base_model: Optional[str] = None
    provider: Optional[str] = Field(default=None, pattern="^(ollama|huggingface|demo)$")
    system_prompt: Optional[str] = None
    tools: Optional[List[str]] = None
    knowledge_sources: Optional[List[str]] = None
    rag_config: Optional[RAGConfigSchema] = None
    lora_adapter: Optional[LoRAAdapterSchema] = None
    version: Optional[str] = None
    inference_config: Optional[InferenceConfigSchema] = None
    test_cases: Optional[List[AgentTestCaseSchema]] = None
    quality_criteria: Optional[List[str]] = None
    security_constraints: Optional[List[str]] = None
    status: Optional[str] = None
    is_demo: Optional[bool] = None
    is_original_slot_integrated: Optional[bool] = None
    changelog: str = "Updated agent definition"


class AgentRead(BaseModel):
    id: str
    slug: str
    slot_number: Optional[int]
    name: str
    purpose: str
    domain: str
    base_model: str
    provider: str
    system_prompt: str
    tools: List[str]
    knowledge_sources: List[str]
    rag_config: RAGConfigSchema
    lora_adapter: LoRAAdapterSchema
    version: str
    inference_config: InferenceConfigSchema
    test_cases: List[AgentTestCaseSchema]
    quality_criteria: List[str]
    security_constraints: List[str]
    status: str
    is_demo: bool
    is_original_slot_integrated: bool
    owner_id: Optional[str]
    created_at: datetime
    updated_at: datetime


class AgentVersionRead(BaseModel):
    id: str
    agent_id: str
    version: str
    changelog: str
    snapshot: Dict[str, Any]
    created_by: Optional[str]
    created_at: datetime


class AgentExportBundle(BaseModel):
    schema_version: str = "isee-agent-definition/v2.0"
    exported_at: datetime
    platform: str = "I.See Agents — Multi-Agent AI Platform"
    agents: List[AgentCreate]


# ==============================================================================
# KNOWLEDGE BASES & RAG
# ==============================================================================
class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=150)
    description: str = Field(..., min_length=3)
    domain: str = Field(..., min_length=2, max_length=120)
    agent_id: Optional[str] = None
    chunk_size: int = Field(default=600, ge=100, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)
    embedding_model: str = Field(default="nomic-embed-text")
    is_isolated: bool = True


class KnowledgeBaseRead(BaseModel):
    id: str
    name: str
    description: str
    domain: str
    owner_id: str
    agent_id: Optional[str]
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    is_isolated: bool
    document_count: int = 0
    chunk_count: int = 0
    created_at: datetime


class DocumentRead(BaseModel):
    id: str
    knowledge_base_id: str
    owner_id: str
    filename: str
    mime_type: str
    file_size_bytes: int
    sha256_hash: str
    extracted_preview: str
    metadata: Dict[str, Any]
    chunk_count: int
    status: str
    created_at: datetime


class RAGCitationSchema(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    similarity_score: float
    excerpt: str


class RAGSearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    knowledge_base_ids: List[str] = Field(default_factory=list)
    top_k: int = Field(default=4, ge=1, le=20)
    similarity_threshold: float = Field(default=0.10, ge=0.0, le=1.0)


class RAGSearchResponse(BaseModel):
    query: str
    embedding_provider: str
    results: List[RAGCitationSchema]


# ==============================================================================
# CHAT & CONVERSATIONS
# ==============================================================================
class ChatRequest(BaseModel):
    agent_id: str
    message: str = Field(..., min_length=1, max_length=16000)
    conversation_id: Optional[str] = None
    override_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    override_top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    use_rag: Optional[bool] = None


class ChatMessageRead(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    provider_used: str
    model_used: str
    is_demo_response: bool
    latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    citations: List[RAGCitationSchema]
    security_flags: List[str]
    created_at: datetime


class ConversationRead(BaseModel):
    id: str
    title: str
    user_id: str
    agent_id: str
    agent_name: Optional[str] = None
    message_count: int = 0
    created_at: datetime
    updated_at: datetime


class ChatResponse(BaseModel):
    conversation: ConversationRead
    user_message: ChatMessageRead
    assistant_message: ChatMessageRead


# ==============================================================================
# SYSTEM STATUS, TELEMETRY & AUDIT
# ==============================================================================
class OllamaStatusSchema(BaseModel):
    reachable: bool
    base_url: str
    latency_ms: Optional[float] = None
    installed_models: List[str] = Field(default_factory=list)
    active_mode: str  # "ollama_live" or "demo_fallback"
    error_detail: Optional[str] = None


class SystemDashboardSchema(BaseModel):
    platform_name: str
    version: str
    database_engine: str
    pgvector_enabled: bool
    ollama_status: OllamaStatusSchema
    total_agents: int
    demo_agents_count: int
    original_slots_total: int = 12
    original_slots_integrated: int
    original_slots_awaiting: int
    total_knowledge_bases: int
    total_documents: int
    total_vector_chunks: int
    total_conversations: int
    total_messages: int
    avg_latency_ms: float


class AuditLogRead(BaseModel):
    id: str
    user_id: Optional[str]
    event_type: str
    severity: str
    resource: str
    details: Dict[str, Any]
    created_at: datetime

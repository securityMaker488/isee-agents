// ==============================================================================
// I.See Agents — API TypeScript Contracts (Mirrors backend/app/schemas/contracts.py)
// ==============================================================================

export type UserRole = 'admin' | 'operator' | 'viewer';

export interface UserCreate {
  email: string;
  full_name: string;
  password: string;
  role?: UserRole;
}

export interface UserLogin {
  email: string;
  password: string;
}

export interface UserRead {
  id: string;
  email: string;
  full_name: string;
  role: UserRole;
  is_active: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: UserRead;
}

// ==============================================================================
// MODEL CONFIGURATIONS & LORA ADAPTERS
// ==============================================================================
export type ModelProvider = 'ollama' | 'huggingface' | 'demo';
export type AdapterMethod = 'none' | 'lora' | 'qlora';

export interface ModelConfigCreate {
  name: string;
  provider: ModelProvider;
  base_model: string;
  version?: string;
  quantization?: string | null;
  context_window?: number;
  default_temperature?: number;
  default_top_p?: number;
  default_max_tokens?: number;
  adapter_type?: AdapterMethod | null;
  adapter_path?: string | null;
  adapter_rank?: number | null;
  adapter_alpha?: number | null;
  is_demo?: boolean;
}

export interface ModelConfigRead extends ModelConfigCreate {
  id: string;
  is_active: boolean;
  created_at: string;
}

// ==============================================================================
// AGENT SPECIALIZATION (12 ORIGINAL SLOTS + DEMO / CUSTOM AGENTS)
// ==============================================================================
export interface RAGConfigSchema {
  enabled: boolean;
  knowledge_base_ids: string[];
  top_k: number;
  similarity_threshold: number;
  chunk_size: number;
  chunk_overlap: number;
}

export interface LoRAAdapterSchema {
  enabled: boolean;
  method: AdapterMethod;
  adapter_name?: string | null;
  adapter_path?: string | null;
  rank: number;
  lora_alpha: number;
  lora_dropout: number;
  target_modules: string[];
  quantization_bits?: number | null;
}

export interface InferenceConfigSchema {
  temperature: number;
  top_p: number;
  max_tokens: number;
  context_window: number;
}

export interface AgentTestCaseSchema {
  name: string;
  input_prompt: string;
  expected_keywords: string[];
  forbidden_keywords: string[];
}

export type AgentStatus = 'ready' | 'awaiting_integration' | 'draft' | 'archived';

export interface AgentCreate {
  slug: string;
  slot_number?: number | null;
  name: string;
  purpose: string;
  domain: string;
  base_model: string;
  provider: ModelProvider;
  system_prompt: string;
  tools: string[];
  knowledge_sources: string[];
  rag_config: RAGConfigSchema;
  lora_adapter: LoRAAdapterSchema;
  version: string;
  inference_config: InferenceConfigSchema;
  test_cases: AgentTestCaseSchema[];
  quality_criteria: string[];
  security_constraints: string[];
  status: AgentStatus;
  is_demo: boolean;
  is_original_slot_integrated: boolean;
  changelog?: string;
}

export interface AgentUpdate {
  name?: string;
  purpose?: string;
  domain?: string;
  base_model?: string;
  provider?: ModelProvider;
  system_prompt?: string;
  tools?: string[];
  knowledge_sources?: string[];
  rag_config?: RAGConfigSchema;
  lora_adapter?: LoRAAdapterSchema;
  version?: string;
  inference_config?: InferenceConfigSchema;
  test_cases?: AgentTestCaseSchema[];
  quality_criteria?: string[];
  security_constraints?: string[];
  status?: AgentStatus;
  is_demo?: boolean;
  is_original_slot_integrated?: boolean;
  changelog: string;
}

export interface AgentRead {
  id: string;
  slug: string;
  slot_number: number | null;
  name: string;
  purpose: string;
  domain: string;
  base_model: string;
  provider: ModelProvider;
  system_prompt: string;
  tools: string[];
  knowledge_sources: string[];
  rag_config: RAGConfigSchema;
  lora_adapter: LoRAAdapterSchema;
  version: string;
  inference_config: InferenceConfigSchema;
  test_cases: AgentTestCaseSchema[];
  quality_criteria: string[];
  security_constraints: string[];
  status: AgentStatus;
  is_demo: boolean;
  is_original_slot_integrated: boolean;
  owner_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface AgentVersionRead {
  id: string;
  agent_id: string;
  version: string;
  changelog: string;
  snapshot: Record<string, unknown>;
  created_by: string | null;
  created_at: string;
}

export interface AgentExportBundle {
  schema_version: string;
  exported_at: string;
  platform: string;
  agents: AgentCreate[];
}

export interface AgentEvaluationTestResult {
  test_name: string;
  passed: boolean;
  latency_ms: number;
  provider_used: string;
  is_demo_response: boolean;
  citations_count: number;
  missing_keywords: string[];
  forbidden_triggered: string[];
}

export interface AgentEvaluationResponse {
  agent_id: string;
  agent_name: string;
  version: string;
  total_test_cases: number;
  passed_test_cases: number;
  pass_rate: number;
  results: AgentEvaluationTestResult[];
}

// ==============================================================================
// KNOWLEDGE BASES & RAG
// ==============================================================================
export interface KnowledgeBaseCreate {
  name: string;
  description: string;
  domain: string;
  agent_id?: string | null;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
  is_isolated: boolean;
}

export interface KnowledgeBaseRead {
  id: string;
  name: string;
  description: string;
  domain: string;
  owner_id: string;
  agent_id: string | null;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
  is_isolated: boolean;
  document_count: number;
  chunk_count: number;
  created_at: string;
}

export interface DocumentRead {
  id: string;
  knowledge_base_id: string;
  owner_id: string;
  filename: string;
  mime_type: string;
  file_size_bytes: number;
  sha256_hash: string;
  extracted_preview: string;
  metadata: Record<string, unknown>;
  chunk_count: number;
  status: string;
  created_at: string;
}

export interface RAGCitationSchema {
  chunk_id: string;
  document_id: string;
  filename: string;
  chunk_index: number;
  similarity_score: number;
  excerpt: string;
}

export interface RAGSearchRequest {
  query: string;
  knowledge_base_ids: string[];
  top_k: number;
  similarity_threshold: number;
}

export interface RAGSearchResponse {
  query: string;
  embedding_provider: string;
  results: RAGCitationSchema[];
}

export interface TextIngestPayload {
  filename: string;
  content: string;
}

// ==============================================================================
// CHAT & CONVERSATIONS
// ==============================================================================
export interface ChatRequest {
  agent_id: string;
  message: string;
  conversation_id?: string | null;
  override_temperature?: number | null;
  override_top_p?: number | null;
  use_rag?: boolean | null;
}

export interface ChatMessageRead {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | string;
  content: string;
  provider_used: string;
  model_used: string;
  is_demo_response: boolean;
  latency_ms: number;
  tokens_generated: number;
  tokens_per_second: number;
  citations: RAGCitationSchema[];
  security_flags: string[];
  created_at: string;
}

export interface ConversationRead {
  id: string;
  title: string;
  user_id: string;
  agent_id: string;
  agent_name: string | null;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ChatResponse {
  conversation: ConversationRead;
  user_message: ChatMessageRead;
  assistant_message: ChatMessageRead;
}

// ==============================================================================
// SYSTEM STATUS, TELEMETRY & AUDIT
// ==============================================================================
export interface OllamaStatusSchema {
  reachable: boolean;
  base_url: string;
  latency_ms: number | null;
  installed_models: string[];
  active_mode: 'ollama_live' | 'demo_fallback' | string;
  error_detail: string | null;
}

export interface SystemDashboardSchema {
  platform_name: string;
  version: string;
  database_engine: string;
  pgvector_enabled: boolean;
  ollama_status: OllamaStatusSchema;
  total_agents: number;
  demo_agents_count: number;
  original_slots_total: number;
  original_slots_integrated: number;
  original_slots_awaiting: number;
  total_knowledge_bases: number;
  total_documents: number;
  total_vector_chunks: number;
  total_conversations: number;
  total_messages: number;
  avg_latency_ms: number;
}

export interface AuditLogRead {
  id: string;
  user_id: string | null;
  event_type: string;
  severity: 'INFO' | 'WARNING' | 'CRITICAL' | string;
  resource: string;
  details: Record<string, unknown>;
  created_at: string;
}

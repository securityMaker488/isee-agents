// ==============================================================================
// I.See Agents — Real REST API Client Service
// ==============================================================================
import type {
  AgentCreate,
  AgentEvaluationResponse,
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
  TextIngestPayload,
  TokenResponse,
  UserCreate,
  UserLogin,
  UserRead,
} from '../types/api';

const TOKEN_STORAGE_KEY = 'isee_agents_jwt_token';

export function getApiBaseUrl(): string {
  const envUrl = import.meta.env?.VITE_API_BASE_URL;
  if (envUrl && typeof envUrl === 'string' && envUrl.trim().length > 0) {
    return envUrl.replace(/\/+$/, '');
  }
  return 'http://localhost:8000/api/v1';
}

export function getStoredToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function setStoredToken(token: string | null): void {
  try {
    if (token) {
      localStorage.setItem(TOKEN_STORAGE_KEY, token);
    } else {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
    }
  } catch {
    // ignore storage errors in restricted environments
  }
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${getApiBaseUrl()}${endpoint}`;
  const token = getStoredToken();

  const headers = new Headers(options.headers || {});
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  if (!(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  let response: Response;
  try {
    response = await fetch(url, {
      ...options,
      headers,
    });
  } catch (networkError) {
    throw new Error(
      `Falha de conexão com a API (${getApiBaseUrl()}). Verifique se o backend FastAPI está em execução na porta 8000. (${
        networkError instanceof Error ? networkError.message : 'Network error'
      })`
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }

  const contentType = response.headers.get('content-type') || '';
  const isJson = contentType.includes('application/json');
  const payload = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    let errorDetail = `Erro HTTP ${response.status}`;
    if (payload && typeof payload === 'object') {
      if ('detail' in payload) {
        if (typeof payload.detail === 'string') {
          errorDetail = payload.detail;
        } else if (Array.isArray(payload.detail)) {
          errorDetail = payload.detail
            .map((e: { msg?: string; loc?: string[] }) =>
              `${e.loc?.join('.') || 'campo'}: ${e.msg || 'inválido'}`
            )
            .join('; ');
        }
      }
    } else if (typeof payload === 'string' && payload.length > 0) {
      errorDetail = payload;
    }
    throw new Error(errorDetail);
  }

  return payload as T;
}

export const api = {
  // 1. Auth
  login: async (credentials: UserLogin): Promise<TokenResponse> => {
    const res = await request<TokenResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials),
    });
    setStoredToken(res.access_token);
    return res;
  },

  register: async (data: UserCreate): Promise<TokenResponse> => {
    const res = await request<TokenResponse>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(data),
    });
    setStoredToken(res.access_token);
    return res;
  },

  getMe: (): Promise<UserRead> => request<UserRead>('/auth/me'),

  listUsers: (): Promise<UserRead[]> => request<UserRead[]>('/auth/users'),

  logout: (): void => {
    setStoredToken(null);
  },

  // 2. System & Telemetry
  getDashboard: (): Promise<SystemDashboardSchema> =>
    request<SystemDashboardSchema>('/system/dashboard'),

  getOllamaStatus: (): Promise<OllamaStatusSchema> =>
    request<OllamaStatusSchema>('/system/ollama'),

  listAuditLogs: (limit = 80): Promise<AuditLogRead[]> =>
    request<AuditLogRead[]>(`/system/audit-logs?limit=${limit}`),

  // 3. Agents Catalog & 12 Specialized Slots
  listAgents: (params?: {
    include_slots?: boolean;
    only_slots?: boolean;
  }): Promise<AgentRead[]> => {
    const search = new URLSearchParams();
    if (params?.include_slots !== undefined) {
      search.set('include_slots', String(params.include_slots));
    }
    if (params?.only_slots !== undefined) {
      search.set('only_slots', String(params.only_slots));
    }
    const qs = search.toString();
    return request<AgentRead[]>(`/agents${qs ? `?${qs}` : ''}`);
  },

  getAgent: (agentId: string): Promise<AgentRead> =>
    request<AgentRead>(`/agents/${agentId}`),

  createAgent: (payload: AgentCreate, overwrite = false): Promise<AgentRead> =>
    request<AgentRead>(`/agents?overwrite=${overwrite}`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  updateAgent: (agentId: string, patch: AgentUpdate): Promise<AgentRead> =>
    request<AgentRead>(`/agents/${agentId}`, {
      method: 'PUT',
      body: JSON.stringify(patch),
    }),

  deleteAgent: (agentId: string): Promise<void> =>
    request<void>(`/agents/${agentId}`, {
      method: 'DELETE',
    }),

  getAgentVersions: (agentId: string): Promise<AgentVersionRead[]> =>
    request<AgentVersionRead[]>(`/agents/${agentId}/versions`),

  evaluateAgent: (agentId: string): Promise<AgentEvaluationResponse> =>
    request<AgentEvaluationResponse>(`/agents/${agentId}/evaluate`, {
      method: 'POST',
    }),

  exportAgentsBundle: (): Promise<AgentExportBundle> =>
    request<AgentExportBundle>('/agents/export/bundle'),

  importAgentsBundle: (
    bundle: AgentExportBundle,
    overwrite = true
  ): Promise<AgentRead[]> =>
    request<AgentRead[]>(`/agents/import/bundle?overwrite=${overwrite}`, {
      method: 'POST',
      body: JSON.stringify(bundle),
    }),

  // 4. Models & LoRA/QLoRA Configurations
  listModels: (): Promise<ModelConfigRead[]> =>
    request<ModelConfigRead[]>('/models'),

  createModel: (payload: ModelConfigCreate): Promise<ModelConfigRead> =>
    request<ModelConfigRead>('/models', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  deleteModel: (modelId: string): Promise<void> =>
    request<void>(`/models/${modelId}`, {
      method: 'DELETE',
    }),

  // 5. Knowledge Bases & RAG
  listKnowledgeBases: (): Promise<KnowledgeBaseRead[]> =>
    request<KnowledgeBaseRead[]>('/knowledge/bases'),

  createKnowledgeBase: (
    payload: KnowledgeBaseCreate
  ): Promise<KnowledgeBaseRead> =>
    request<KnowledgeBaseRead>('/knowledge/bases', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  listDocuments: (kbId: string): Promise<DocumentRead[]> =>
    request<DocumentRead[]>(`/knowledge/bases/${kbId}/documents`),

  uploadDocument: (kbId: string, file: File): Promise<DocumentRead> => {
    const formData = new FormData();
    formData.append('file', file);
    return request<DocumentRead>(`/knowledge/bases/${kbId}/upload`, {
      method: 'POST',
      body: formData,
    });
  },

  ingestText: (
    kbId: string,
    payload: TextIngestPayload
  ): Promise<DocumentRead> =>
    request<DocumentRead>(`/knowledge/bases/${kbId}/ingest-text`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  searchKnowledge: (payload: RAGSearchRequest): Promise<RAGSearchResponse> =>
    request<RAGSearchResponse>('/knowledge/search', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // 6. Multi-Agent Chat & Conversations
  sendChatMessage: (payload: ChatRequest): Promise<ChatResponse> =>
    request<ChatResponse>('/chat/completions', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  listConversations: (agentId?: string): Promise<ConversationRead[]> => {
    const qs = agentId ? `?agent_id=${encodeURIComponent(agentId)}` : '';
    return request<ConversationRead[]>(`/chat/conversations${qs}`);
  },

  listConversationMessages: (
    conversationId: string
  ): Promise<ChatMessageRead[]> =>
    request<ChatMessageRead[]>(
      `/chat/conversations/${conversationId}/messages`
    ),

  deleteConversation: (conversationId: string): Promise<void> =>
    request<void>(`/chat/conversations/${conversationId}`, {
      method: 'DELETE',
    }),
};

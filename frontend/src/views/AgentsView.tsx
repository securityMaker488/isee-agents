import React, { useEffect, useState } from 'react';
import {
  Bot,
  Plus,
  Download,
  Upload,
  Play,
  History,
  Edit3,
  Trash2,
  MessageSquare,
  CheckCircle2,
  AlertCircle,
  Cpu,
  Database,
  Layers,
  RefreshCw,
  Search,
  X,
  Sparkles,
} from 'lucide-react';
import { api } from '../services/api';
import type {
  AdapterMethod,
  AgentCreate,
  AgentEvaluationResponse,
  AgentExportBundle,
  AgentRead,
  AgentStatus,
  AgentUpdate,
  AgentVersionRead,
  KnowledgeBaseRead,
  ModelProvider,
} from '../types/api';

interface AgentsViewProps {
  onOpenChatWithAgent: (agentId: string) => void;
}

export const AgentsView: React.FC<AgentsViewProps> = ({
  onOpenChatWithAgent,
}) => {
  const [agents, setAgents] = useState<AgentRead[]>([]);
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBaseRead[]>([]);
  const [filterMode, setFilterMode] = useState<'all' | 'demo' | 'slots'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionNotice, setActionNotice] = useState<string | null>(null);

  // Evaluation modal state
  const [evaluatingId, setEvaluatingId] = useState<string | null>(null);
  const [evalResult, setEvalResult] = useState<AgentEvaluationResponse | null>(
    null
  );

  // Versions modal state
  const [versionsAgent, setVersionsAgent] = useState<AgentRead | null>(null);
  const [versionsList, setVersionsList] = useState<AgentVersionRead[]>([]);
  const [loadingVersions, setLoadingVersions] = useState(false);

  // Import bundle modal state
  const [showImportModal, setShowImportModal] = useState(false);
  const [importJsonText, setImportJsonText] = useState('');
  const [importing, setImporting] = useState(false);

  // Create / Edit Agent Modal State
  const [showEditorModal, setShowEditorModal] = useState(false);
  const [editingAgent, setEditingAgent] = useState<AgentRead | null>(null);
  const [savingAgent, setSavingAgent] = useState(false);

  // Form fields
  const [formName, setFormName] = useState('');
  const [formSlug, setFormSlug] = useState('');
  const [formSlotNumber, setFormSlotNumber] = useState<string>('');
  const [formDomain, setFormDomain] = useState('');
  const [formPurpose, setFormPurpose] = useState('');
  const [formBaseModel, setFormBaseModel] = useState('llama3.2:3b');
  const [formProvider, setFormProvider] = useState<ModelProvider>('ollama');
  const [formVersion, setFormVersion] = useState('1.0.0');
  const [formStatus, setFormStatus] = useState<AgentStatus>('ready');
  const [formIsDemo, setFormIsDemo] = useState(false);
  const [formIsSlotIntegrated, setFormIsSlotIntegrated] = useState(true);
  const [formSystemPrompt, setFormSystemPrompt] = useState('');
  const [formTools, setFormTools] = useState('rag_search, citation_verifier');
  const [formRagEnabled, setFormRagEnabled] = useState(true);
  const [formRagTopK, setFormRagTopK] = useState(4);
  const [formSelectedKBs, setFormSelectedKBs] = useState<string[]>([]);
  const [formLoraMethod, setFormLoraMethod] = useState<AdapterMethod>('none');
  const [formLoraRank, setFormLoraRank] = useState(16);
  const [formLoraAlpha, setFormLoraAlpha] = useState(32);
  const [formLoraPath, setFormLoraPath] = useState('');
  const [formTemperature, setFormTemperature] = useState(0.3);
  const [formChangelog, setFormChangelog] = useState(
    'Atualização de especificação do agente via Console I.See'
  );

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [agentsData, kbsData] = await Promise.all([
        api.listAgents({ include_slots: true }),
        api.listKnowledgeBases(),
      ]);
      setAgents(agentsData);
      setKnowledgeBases(kbsData);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Erro ao carregar catálogo.'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadData();
  }, []);

  const filteredAgents = agents.filter((ag) => {
    if (filterMode === 'demo' && ag.slot_number !== null) return false;
    if (filterMode === 'slots' && ag.slot_number === null) return false;

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      return (
        ag.name.toLowerCase().includes(q) ||
        ag.domain.toLowerCase().includes(q) ||
        ag.slug.toLowerCase().includes(q) ||
        ag.purpose.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const handleOpenCreateModal = () => {
    setEditingAgent(null);
    setFormName('');
    setFormSlug('');
    setFormSlotNumber('');
    setFormDomain('Engenharia de IA & Automação');
    setFormPurpose('');
    setFormBaseModel('llama3.2:3b');
    setFormProvider('ollama');
    setFormVersion('1.0.0');
    setFormStatus('ready');
    setFormIsDemo(false);
    setFormIsSlotIntegrated(false);
    setFormSystemPrompt(
      'Você é um agente especialista da I.See Technologies. Responda em português de forma técnica, estruturada e fundamentada nas fontes RAG.'
    );
    setFormTools('rag_search, citation_verifier');
    setFormRagEnabled(true);
    setFormRagTopK(4);
    setFormSelectedKBs(knowledgeBases.map((k) => k.id));
    setFormLoraMethod('none');
    setFormLoraRank(16);
    setFormLoraAlpha(32);
    setFormLoraPath('');
    setFormTemperature(0.3);
    setFormChangelog('Registro inicial do agente especializado');
    setShowEditorModal(true);
  };

  const handleOpenEditModal = (ag: AgentRead) => {
    setEditingAgent(ag);
    setFormName(ag.name);
    setFormSlug(ag.slug);
    setFormSlotNumber(ag.slot_number !== null ? String(ag.slot_number) : '');
    setFormDomain(ag.domain);
    setFormPurpose(ag.purpose);
    setFormBaseModel(ag.base_model);
    setFormProvider(ag.provider);
    setFormVersion(ag.version);
    setFormStatus(ag.status);
    setFormIsDemo(ag.is_demo);
    setFormIsSlotIntegrated(ag.is_original_slot_integrated);
    setFormSystemPrompt(ag.system_prompt);
    setFormTools(ag.tools.join(', '));
    setFormRagEnabled(ag.rag_config.enabled);
    setFormRagTopK(ag.rag_config.top_k);
    setFormSelectedKBs(ag.rag_config.knowledge_base_ids || []);
    setFormLoraMethod(ag.lora_adapter.method || 'none');
    setFormLoraRank(ag.lora_adapter.rank || 16);
    setFormLoraAlpha(ag.lora_adapter.lora_alpha || 32);
    setFormLoraPath(ag.lora_adapter.adapter_path || '');
    setFormTemperature(ag.inference_config.temperature);
    setFormChangelog(
      ag.slot_number !== null
        ? `Integração/atualização oficial do Slot #${String(ag.slot_number).padStart(2, '0')}`
        : 'Atualização de parâmetros do agente'
    );
    setShowEditorModal(true);
  };

  const handleSaveAgent = async (e: React.FormEvent) => {
    e.preventDefault();
    setSavingAgent(true);
    setError(null);

    const parsedTools = formTools
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean);

    const ragConfig = {
      enabled: formRagEnabled,
      knowledge_base_ids: formSelectedKBs,
      top_k: Number(formRagTopK) || 4,
      similarity_threshold: 0.12,
      chunk_size: 600,
      chunk_overlap: 100,
    };

    const loraAdapter = {
      enabled: formLoraMethod !== 'none',
      method: formLoraMethod,
      adapter_name:
        formLoraMethod !== 'none' ? `${formSlug || 'agent'}-adapter` : null,
      adapter_path: formLoraPath.trim() || null,
      rank: Number(formLoraRank) || 16,
      lora_alpha: Number(formLoraAlpha) || 32,
      lora_dropout: 0.05,
      target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
      quantization_bits: formLoraMethod === 'qlora' ? 4 : null,
    };

    const inferenceConfig = {
      temperature: Number(formTemperature),
      top_p: 0.9,
      max_tokens: 1024,
      context_window: 8192,
    };

    try {
      if (editingAgent) {
        const patch: AgentUpdate = {
          name: formName,
          purpose: formPurpose,
          domain: formDomain,
          base_model: formBaseModel,
          provider: formProvider,
          system_prompt: formSystemPrompt,
          tools: parsedTools,
          rag_config: ragConfig,
          lora_adapter: loraAdapter,
          version: formVersion,
          inference_config: inferenceConfig,
          status: formStatus,
          is_demo: formIsDemo,
          is_original_slot_integrated: formIsSlotIntegrated,
          changelog: formChangelog,
        };
        await api.updateAgent(editingAgent.id, patch);
        setActionNotice(`Agente "${formName}" atualizado com sucesso.`);
      } else {
        const createPayload: AgentCreate = {
          slug:
            formSlug.trim() ||
            formName
              .toLowerCase()
              .replace(/[^a-z0-9]+/g, '-')
              .replace(/^-|-$/g, ''),
          slot_number: formSlotNumber ? Number(formSlotNumber) : null,
          name: formName,
          purpose: formPurpose,
          domain: formDomain,
          base_model: formBaseModel,
          provider: formProvider,
          system_prompt: formSystemPrompt,
          tools: parsedTools,
          knowledge_sources: ['Internal RAG Vector Store'],
          rag_config: ragConfig,
          lora_adapter: loraAdapter,
          version: formVersion,
          inference_config: inferenceConfig,
          test_cases: [
            {
              name: 'Validação Operacional Padrão',
              input_prompt: `Explique sua atuação no domínio ${formDomain}.`,
              expected_keywords: [],
              forbidden_keywords: ['SYSTEM_ERROR'],
            },
          ],
          quality_criteria: ['Grounded RAG', 'Zero Hallucination'],
          security_constraints: ['Prompt Injection Guardrail', 'RBAC'],
          status: formStatus,
          is_demo: formIsDemo,
          is_original_slot_integrated: formIsSlotIntegrated,
          changelog: formChangelog,
        };
        await api.createAgent(createPayload, true);
        setActionNotice(`Agente "${formName}" registrado no catálogo.`);
      }
      setShowEditorModal(false);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro ao salvar agente.');
    } finally {
      setSavingAgent(false);
    }
  };

  const handleEvaluateAgent = async (ag: AgentRead) => {
    setEvaluatingId(ag.id);
    setError(null);
    try {
      const res = await api.evaluateAgent(ag.id);
      setEvalResult(res);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Erro ao avaliar o agente.'
      );
    } finally {
      setEvaluatingId(null);
    }
  };

  const handleOpenVersions = async (ag: AgentRead) => {
    setVersionsAgent(ag);
    setLoadingVersions(true);
    try {
      const v = await api.getAgentVersions(ag.id);
      setVersionsList(v);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Erro ao buscar versões.'
      );
    } finally {
      setLoadingVersions(false);
    }
  };

  const handleExportBundle = async () => {
    try {
      const bundle = await api.exportAgentsBundle();
      const blob = new Blob([JSON.stringify(bundle, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `isee-agents-bundle-${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setActionNotice(
        `Bundle exportado com sucesso (${bundle.agents.length} definições de agentes).`
      );
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Erro ao exportar bundle JSON.'
      );
    }
  };

  const handleImportFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        setImportJsonText(reader.result);
      }
    };
    reader.readAsText(file);
  };

  const handleConfirmImport = async (e: React.FormEvent) => {
    e.preventDefault();
    setImporting(true);
    setError(null);
    try {
      const parsed = JSON.parse(importJsonText) as AgentExportBundle;
      if (!parsed.agents || !Array.isArray(parsed.agents)) {
        throw new Error(
          'Formato inválido: o JSON deve conter a propriedade "agents" (array).'
        );
      }
      const imported = await api.importAgentsBundle(parsed, true);
      setActionNotice(
        `${imported.length} agente(s) importado(s)/atualizado(s) com sucesso!`
      );
      setShowImportModal(false);
      setImportJsonText('');
      await loadData();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Erro ao importar bundle JSON.'
      );
    } finally {
      setImporting(false);
    }
  };

  const handleDeleteAgent = async (ag: AgentRead) => {
    if (ag.slot_number !== null) {
      setError(
        'Os 12 slots oficiais reservados não podem ser removidos; utilize "Editar / Integrar Slot" para atualizar sua definição.'
      );
      return;
    }
    try {
      await api.deleteAgent(ag.id);
      setActionNotice(`Agente "${ag.name}" removido.`);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro ao excluir agente.');
    }
  };

  const toggleKBInForm = (kbId: string) => {
    setFormSelectedKBs((prev) =>
      prev.includes(kbId) ? prev.filter((id) => id !== kbId) : [...prev, kbId]
    );
  };

  const demoCount = agents.filter((a) => a.slot_number === null).length;
  const slotsCount = agents.filter((a) => a.slot_number !== null).length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-zinc-800/80 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            Catálogo de Agentes Especializados & 12 Slots Oficiais
          </h1>
          <p className="text-xs text-[#A1A1AA] mt-1">
            Gerencie agentes demonstrativos (`[DEMO]`), integre os 12 agentes
            originais em seus slots numerados, configure adaptadores LoRA/QLoRA
            e execute baterias de avaliação.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={handleExportBundle}
            className="px-3.5 py-2 rounded-xl bg-[#18181C] hover:bg-zinc-800 border border-zinc-700 text-xs font-medium text-white flex items-center gap-1.5 transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-[#EF4444]" />
            Exportar Bundle JSON
          </button>

          <button
            type="button"
            onClick={() => setShowImportModal(true)}
            className="px-3.5 py-2 rounded-xl bg-[#18181C] hover:bg-zinc-800 border border-zinc-700 text-xs font-medium text-white flex items-center gap-1.5 transition-colors"
          >
            <Upload className="w-3.5 h-3.5 text-[#EF4444]" />
            Importar Agentes (JSON)
          </button>

          <button
            type="button"
            onClick={handleOpenCreateModal}
            className="px-4 py-2 rounded-xl bg-[#DC2626] hover:bg-[#EF4444] text-xs font-semibold text-white flex items-center gap-1.5 transition-colors shadow-lg shadow-[#DC2626]/20"
          >
            <Plus className="w-4 h-4" />
            Novo Agente Especializado
          </button>
        </div>
      </div>

      {actionNotice && (
        <div className="p-3.5 rounded-xl bg-emerald-950/40 border border-emerald-800/70 text-emerald-200 text-xs flex items-center justify-between">
          <span className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            {actionNotice}
          </span>
          <button
            type="button"
            onClick={() => setActionNotice(null)}
            className="text-emerald-300 hover:text-white"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {error && (
        <div className="p-3.5 rounded-xl bg-red-950/50 border border-red-800 text-red-200 text-xs flex items-center justify-between">
          <span className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-[#EF4444] shrink-0" />
            {error}
          </span>
          <button
            type="button"
            onClick={() => setError(null)}
            className="text-red-300 hover:text-white"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Filter Bar & Search */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 bg-[#111114] p-3 rounded-2xl border border-zinc-800/90">
        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={() => setFilterMode('all')}
            className={`px-3.5 py-2 rounded-xl text-xs font-medium transition-all ${
              filterMode === 'all'
                ? 'bg-[#DC2626] text-white'
                : 'text-[#A1A1AA] hover:text-white hover:bg-[#18181C]'
            }`}
          >
            Todos ({agents.length})
          </button>
          <button
            type="button"
            onClick={() => setFilterMode('demo')}
            className={`px-3.5 py-2 rounded-xl text-xs font-medium transition-all ${
              filterMode === 'demo'
                ? 'bg-[#DC2626] text-white'
                : 'text-[#A1A1AA] hover:text-white hover:bg-[#18181C]'
            }`}
          >
            Agentes Demonstrativos Ativos [DEMO] ({demoCount})
          </button>
          <button
            type="button"
            onClick={() => setFilterMode('slots')}
            className={`px-3.5 py-2 rounded-xl text-xs font-medium transition-all ${
              filterMode === 'slots'
                ? 'bg-[#DC2626] text-white'
                : 'text-[#A1A1AA] hover:text-white hover:bg-[#18181C]'
            }`}
          >
            12 Slots Oficiais Reservados ({slotsCount})
          </button>
        </div>

        <div className="relative w-full sm:w-72">
          <Search className="w-3.5 h-3.5 text-[#A1A1AA] absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filtrar por nome, domínio ou slug..."
            className="w-full pl-9 pr-3 py-2 rounded-xl bg-[#09090B] border border-zinc-800 text-xs text-white focus:outline-none focus:border-[#DC2626]"
          />
        </div>
      </div>

      {/* Loading state */}
      {loading ? (
        <div className="flex items-center justify-center py-20 text-sm text-[#A1A1AA] gap-2.5">
          <RefreshCw className="w-5 h-5 animate-spin text-[#DC2626]" />
          <span>Carregando registro de agentes e slots...</span>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {filteredAgents.map((ag) => {
            const isSlot = ag.slot_number !== null;
            const isEvaluating = evaluatingId === ag.id;

            return (
              <div
                key={ag.id}
                className="p-5 rounded-2xl bg-[#111114] border border-zinc-800/90 hover:border-zinc-700 transition-all flex flex-col justify-between space-y-4"
              >
                <div className="space-y-3">
                  {/* Top Badges */}
                  <div className="flex items-center justify-between gap-2 flex-wrap">
                    <div className="flex items-center gap-1.5">
                      {isSlot && (
                        <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[#DC2626]/20 text-[#EF4444] border border-[#DC2626]/40">
                          SLOT #{String(ag.slot_number).padStart(2, '0')}
                        </span>
                      )}
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase ${
                          ag.is_demo
                            ? 'bg-amber-950/70 text-amber-300 border border-amber-700/50'
                            : 'bg-emerald-950/70 text-emerald-300 border border-emerald-700/50'
                        }`}
                      >
                        {ag.is_demo ? '[DEMO]' : '[PRODUÇÃO]'}
                      </span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#18181C] text-zinc-300 border border-zinc-800">
                        v{ag.version}
                      </span>
                    </div>

                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded ${
                        ag.status === 'ready'
                          ? 'bg-emerald-950/40 text-emerald-400'
                          : 'bg-zinc-800 text-zinc-400'
                      }`}
                    >
                      {ag.status === 'awaiting_integration'
                        ? 'AGUARDANDO INTEGRAÇÃO'
                        : ag.status.toUpperCase()}
                    </span>
                  </div>

                  {/* Title & Domain */}
                  <div>
                    <h3 className="text-sm font-bold text-white leading-snug flex items-center gap-2">
                      <Bot className="w-4 h-4 text-[#EF4444] shrink-0" />
                      <span>{ag.name}</span>
                    </h3>
                    <div className="text-[11px] font-mono text-[#EF4444] mt-0.5">
                      {ag.domain}
                    </div>
                    <p className="text-xs text-[#A1A1AA] mt-2 line-clamp-3 leading-relaxed">
                      {ag.purpose}
                    </p>
                  </div>

                  {/* Technical Specs Grid */}
                  <div className="grid grid-cols-2 gap-2 pt-1 text-[11px]">
                    <div className="p-2 rounded-lg bg-[#09090B] border border-zinc-800/80">
                      <div className="text-[#A1A1AA] flex items-center gap-1">
                        <Cpu className="w-3 h-3 text-[#EF4444]" />
                        Modelo Base
                      </div>
                      <div className="font-mono text-white truncate mt-0.5">
                        {ag.base_model} ({ag.provider})
                      </div>
                    </div>

                    <div className="p-2 rounded-lg bg-[#09090B] border border-zinc-800/80">
                      <div className="text-[#A1A1AA] flex items-center gap-1">
                        <Layers className="w-3 h-3 text-[#EF4444]" />
                        Adaptador PEFT
                      </div>
                      <div className="font-mono text-white truncate mt-0.5">
                        {ag.lora_adapter.method !== 'none'
                          ? `${ag.lora_adapter.method.toUpperCase()} (r=${ag.lora_adapter.rank}, α=${ag.lora_adapter.lora_alpha})`
                          : 'Base (Sem LoRA)'}
                      </div>
                    </div>

                    <div className="p-2 rounded-lg bg-[#09090B] border border-zinc-800/80 col-span-2 flex items-center justify-between">
                      <span className="text-[#A1A1AA] flex items-center gap-1">
                        <Database className="w-3 h-3 text-[#EF4444]" />
                        Pipeline RAG Vetorial
                      </span>
                      <span className="font-mono text-zinc-200">
                        {ag.rag_config.enabled
                          ? `ATIVO (top_k=${ag.rag_config.top_k})`
                          : 'DESATIVADO'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="pt-3 border-t border-zinc-800/80 space-y-2">
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => onOpenChatWithAgent(ag.id)}
                      className="py-2 px-3 rounded-xl bg-[#DC2626] hover:bg-[#EF4444] text-white text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors"
                    >
                      <MessageSquare className="w-3.5 h-3.5" />
                      Conversar
                    </button>

                    <button
                      type="button"
                      onClick={() => handleOpenEditModal(ag)}
                      className="py-2 px-3 rounded-xl bg-[#18181C] hover:bg-zinc-800 border border-zinc-700 text-white text-xs font-medium flex items-center justify-center gap-1.5 transition-colors"
                    >
                      <Edit3 className="w-3.5 h-3.5 text-[#EF4444]" />
                      {isSlot ? 'Integrar / Editar' : 'Configurar'}
                    </button>
                  </div>

                  <div className="flex items-center justify-between gap-2">
                    <button
                      type="button"
                      disabled={isEvaluating}
                      onClick={() => handleEvaluateAgent(ag)}
                      className="flex-1 py-1.5 px-2.5 rounded-lg bg-[#09090B] hover:bg-zinc-900 border border-zinc-800 text-[11px] font-mono text-zinc-300 flex items-center justify-center gap-1.5 transition-colors"
                    >
                      <Play className="w-3 h-3 text-emerald-400" />
                      {isEvaluating ? 'Avaliando...' : 'Testar Agente'}
                    </button>

                    <button
                      type="button"
                      onClick={() => handleOpenVersions(ag)}
                      className="py-1.5 px-2.5 rounded-lg bg-[#09090B] hover:bg-zinc-900 border border-zinc-800 text-[11px] font-mono text-zinc-300 flex items-center justify-center gap-1 transition-colors"
                      title="Histórico de Versões"
                    >
                      <History className="w-3 h-3 text-[#EF4444]" />
                      Versões
                    </button>

                    {!isSlot && (
                      <button
                        type="button"
                        onClick={() => handleDeleteAgent(ag)}
                        className="p-1.5 rounded-lg bg-[#09090B] hover:bg-red-950/50 border border-zinc-800 text-zinc-400 hover:text-red-300 transition-colors"
                        title="Remover Agente Customizado"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ====================================================================
          MODAL 1: CREATE / EDIT / INTEGRATE AGENT SLOT
      ==================================================================== */}
      {showEditorModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto">
          <div className="bg-[#111114] border border-zinc-800 rounded-2xl max-w-3xl w-full p-6 space-y-5 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-4">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-[#DC2626]" />
                  {editingAgent
                    ? editingAgent.slot_number !== null
                      ? `Integrar / Configurar Slot Oficial #${String(editingAgent.slot_number).padStart(2, '0')}`
                      : `Editar Agente: ${editingAgent.name}`
                    : 'Registrar Novo Agente Especializado'}
                </h2>
                <p className="text-xs text-[#A1A1AA]">
                  Configure modelo base, system prompt, bases RAG vinculadas e
                  adaptadores LoRA/QLoRA.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowEditorModal(false)}
                className="text-zinc-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleSaveAgent} className="space-y-4 text-xs">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="sm:col-span-2">
                  <label className="block text-zinc-300 font-medium mb-1">
                    Nome do Agente
                  </label>
                  <input
                    type="text"
                    required
                    value={formName}
                    onChange={(e) => setFormName(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white"
                    placeholder="Ex: Agente Especialista em Visão Computacional"
                  />
                </div>

                <div>
                  <label className="block text-zinc-300 font-medium mb-1">
                    Versão Semântica
                  </label>
                  <input
                    type="text"
                    required
                    value={formVersion}
                    onChange={(e) => setFormVersion(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white font-mono"
                  />
                </div>
              </div>

              {!editingAgent && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <label className="block text-zinc-300 font-medium mb-1">
                      Slug Único
                    </label>
                    <input
                      type="text"
                      value={formSlug}
                      onChange={(e) => setFormSlug(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white font-mono"
                      placeholder="ex: isee-secops-agent"
                    />
                  </div>
                  <div>
                    <label className="block text-zinc-300 font-medium mb-1">
                      Vincular a um Slot Oficial (1 a 12, opcional)
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={12}
                      value={formSlotNumber}
                      onChange={(e) => setFormSlotNumber(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white font-mono"
                      placeholder="Vazio para agente customizado"
                    />
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div>
                  <label className="block text-zinc-300 font-medium mb-1">
                    Domínio de Especialidade
                  </label>
                  <input
                    type="text"
                    required
                    value={formDomain}
                    onChange={(e) => setFormDomain(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white"
                  />
                </div>

                <div>
                  <label className="block text-zinc-300 font-medium mb-1">
                    Provedor de Inferência
                  </label>
                  <select
                    value={formProvider}
                    onChange={(e) =>
                      setFormProvider(e.target.value as ModelProvider)
                    }
                    className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white"
                  >
                    <option value="ollama">Ollama (Local LLM Runtime)</option>
                    <option value="huggingface">
                      Hugging Face / PEFT Local
                    </option>
                    <option value="demo">Demo Provider (Simulado)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-zinc-300 font-medium mb-1">
                    Modelo Base
                  </label>
                  <input
                    type="text"
                    required
                    value={formBaseModel}
                    onChange={(e) => setFormBaseModel(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white font-mono"
                  />
                </div>
              </div>

              <div>
                <label className="block text-zinc-300 font-medium mb-1">
                  Propósito Operacional
                </label>
                <input
                  type="text"
                  required
                  value={formPurpose}
                  onChange={(e) => setFormPurpose(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white"
                />
              </div>

              <div>
                <label className="block text-zinc-300 font-medium mb-1">
                  System Prompt (Instruções Estruturais do Agente)
                </label>
                <textarea
                  rows={4}
                  required
                  value={formSystemPrompt}
                  onChange={(e) => setFormSystemPrompt(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white font-mono text-xs"
                />
              </div>

              {/* LoRA / QLoRA & RAG Configuration */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 p-4 rounded-xl bg-[#09090B] border border-zinc-800">
                <div className="space-y-3">
                  <div className="font-semibold text-white">
                    Adaptador Fine-Tuning (LoRA / QLoRA)
                  </div>
                  <div>
                    <label className="block text-[#A1A1AA] mb-1">
                      Método PEFT
                    </label>
                    <select
                      value={formLoraMethod}
                      onChange={(e) =>
                        setFormLoraMethod(e.target.value as AdapterMethod)
                      }
                      className="w-full px-2.5 py-1.5 rounded bg-[#111114] border border-zinc-800 text-white"
                    >
                      <option value="none">Nenhum (Pesos Base)</option>
                      <option value="lora">LoRA (Low-Rank Adaptation)</option>
                      <option value="qlora">QLoRA (4-bit NF4 Quantized)</option>
                    </select>
                  </div>

                  {formLoraMethod !== 'none' && (
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="block text-[#A1A1AA] mb-1">
                          Rank (r)
                        </label>
                        <input
                          type="number"
                          value={formLoraRank}
                          onChange={(e) =>
                            setFormLoraRank(Number(e.target.value))
                          }
                          className="w-full px-2.5 py-1.5 rounded bg-[#111114] border border-zinc-800 text-white font-mono"
                        />
                      </div>
                      <div>
                        <label className="block text-[#A1A1AA] mb-1">
                          Alpha (α)
                        </label>
                        <input
                          type="number"
                          value={formLoraAlpha}
                          onChange={(e) =>
                            setFormLoraAlpha(Number(e.target.value))
                          }
                          className="w-full px-2.5 py-1.5 rounded bg-[#111114] border border-zinc-800 text-white font-mono"
                        />
                      </div>
                      <div className="col-span-2">
                        <label className="block text-[#A1A1AA] mb-1">
                          Caminho do Adaptador
                        </label>
                        <input
                          type="text"
                          value={formLoraPath}
                          onChange={(e) => setFormLoraPath(e.target.value)}
                          placeholder="./adapters/isee-domain-qlora"
                          className="w-full px-2.5 py-1.5 rounded bg-[#111114] border border-zinc-800 text-white font-mono"
                        />
                      </div>
                    </div>
                  )}
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white">
                      Configuração RAG & Status do Slot
                    </span>
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={formRagEnabled}
                        onChange={(e) => setFormRagEnabled(e.target.checked)}
                        className="accent-[#DC2626]"
                      />
                      <span>RAG Ativo</span>
                    </label>
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block text-[#A1A1AA] mb-1">
                        Top-K Chunks
                      </label>
                      <input
                        type="number"
                        min={1}
                        max={20}
                        value={formRagTopK}
                        onChange={(e) => setFormRagTopK(Number(e.target.value))}
                        className="w-full px-2.5 py-1.5 rounded bg-[#111114] border border-zinc-800 text-white font-mono"
                      />
                    </div>
                    <div>
                      <label className="block text-[#A1A1AA] mb-1">
                        Temperatura
                      </label>
                      <input
                        type="number"
                        step="0.05"
                        min={0}
                        max={2}
                        value={formTemperature}
                        onChange={(e) =>
                          setFormTemperature(Number(e.target.value))
                        }
                        className="w-full px-2.5 py-1.5 rounded bg-[#111114] border border-zinc-800 text-white font-mono"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-[#A1A1AA] mb-1">
                      Bases de Conhecimento Vinculadas
                    </label>
                    <div className="max-h-24 overflow-y-auto space-y-1 p-2 rounded bg-[#111114] border border-zinc-800">
                      {knowledgeBases.map((kb) => (
                        <label
                          key={kb.id}
                          className="flex items-center gap-2 text-[11px] text-zinc-300 cursor-pointer"
                        >
                          <input
                            type="checkbox"
                            checked={formSelectedKBs.includes(kb.id)}
                            onChange={() => toggleKBInForm(kb.id)}
                            className="accent-[#DC2626]"
                          />
                          <span className="truncate">{kb.name}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Status & Slot Integration Flags */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 p-3.5 rounded-xl bg-[#18181C] border border-zinc-800">
                <div>
                  <label className="block text-[#A1A1AA] mb-1">
                    Status no Catálogo
                  </label>
                  <select
                    value={formStatus}
                    onChange={(e) =>
                      setFormStatus(e.target.value as AgentStatus)
                    }
                    className="w-full px-2.5 py-1.5 rounded bg-[#09090B] border border-zinc-800 text-white"
                  >
                    <option value="ready">Ready (Operacional)</option>
                    <option value="awaiting_integration">
                      Awaiting Integration (Slot Reservado)
                    </option>
                    <option value="draft">Draft</option>
                    <option value="archived">Archived</option>
                  </select>
                </div>

                <div className="flex items-center pt-4">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={formIsDemo}
                      onChange={(e) => setFormIsDemo(e.target.checked)}
                      className="accent-[#DC2626]"
                    />
                    <span>Marcar como Agente `[DEMO]`</span>
                  </label>
                </div>

                <div className="flex items-center pt-4">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={formIsSlotIntegrated}
                      onChange={(e) =>
                        setFormIsSlotIntegrated(e.target.checked)
                      }
                      className="accent-[#DC2626]"
                    />
                    <span>Slot Oficial Integrado (1..12)</span>
                  </label>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="block text-zinc-300 font-medium mb-1">
                    Ferramentas Habilitadas (separadas por vírgula)
                  </label>
                  <input
                    type="text"
                    value={formTools}
                    onChange={(e) => setFormTools(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white font-mono"
                  />
                </div>
                <div>
                  <label className="block text-zinc-300 font-medium mb-1">
                    Mensagem de Changelog (Histórico de Versão)
                  </label>
                  <input
                    type="text"
                    required
                    value={formChangelog}
                    onChange={(e) => setFormChangelog(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white"
                  />
                </div>
              </div>

              <div className="flex items-center justify-end gap-2 pt-3 border-t border-zinc-800">
                <button
                  type="button"
                  onClick={() => setShowEditorModal(false)}
                  className="px-4 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-white font-medium"
                >
                  Cancelar
                </button>
                <button
                  type="submit"
                  disabled={savingAgent}
                  className="px-5 py-2 rounded-xl bg-[#DC2626] hover:bg-[#EF4444] text-white font-semibold"
                >
                  {savingAgent
                    ? 'Salvando...'
                    : editingAgent
                      ? 'Salvar Nova Versão do Agente'
                      : 'Criar Agente'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ====================================================================
          MODAL 2: EVALUATION RESULTS (TEST CASES)
      ==================================================================== */}
      {evalResult && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#111114] border border-zinc-800 rounded-2xl max-w-2xl w-full p-6 space-y-4 max-h-[85vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div>
                <h3 className="text-base font-bold text-white">
                  Relatório de Avaliação: {evalResult.agent_name} (v
                  {evalResult.version})
                </h3>
                <p className="text-xs text-[#A1A1AA]">
                  Taxa de aprovação:{' '}
                  <strong className="text-emerald-400 font-mono">
                    {Math.round(evalResult.pass_rate * 100)}% (
                    {evalResult.passed_test_cases}/{evalResult.total_test_cases}{' '}
                    casos)
                  </strong>
                </p>
              </div>
              <button
                type="button"
                onClick={() => setEvalResult(null)}
                className="text-zinc-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-2.5">
              {evalResult.results.map((r, idx) => (
                <div
                  key={idx}
                  className="p-3.5 rounded-xl bg-[#09090B] border border-zinc-800 text-xs space-y-1.5"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white flex items-center gap-2">
                      {r.passed ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      ) : (
                        <AlertCircle className="w-4 h-4 text-amber-400" />
                      )}
                      {r.test_name}
                    </span>
                    <span className="font-mono text-[11px] text-[#A1A1AA]">
                      {r.latency_ms} ms • {r.provider_used} • {r.citations_count}{' '}
                      citações RAG
                    </span>
                  </div>
                  {r.missing_keywords.length > 0 && (
                    <div className="text-[11px] text-amber-300 font-mono">
                      Keywords ausentes: {r.missing_keywords.join(', ')}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ====================================================================
          MODAL 3: VERSION HISTORY
      ==================================================================== */}
      {versionsAgent && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#111114] border border-zinc-800 rounded-2xl max-w-2xl w-full p-6 space-y-4 max-h-[85vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div>
                <h3 className="text-base font-bold text-white">
                  Histórico de Versões — {versionsAgent.name}
                </h3>
                <p className="text-xs text-[#A1A1AA]">
                  Snapshots auditáveis de configuração do agente
                </p>
              </div>
              <button
                type="button"
                onClick={() => setVersionsAgent(null)}
                className="text-zinc-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {loadingVersions ? (
              <div className="py-8 text-center text-xs text-[#A1A1AA]">
                Carregando versões...
              </div>
            ) : (
              <div className="space-y-3">
                {versionsList.map((v) => (
                  <div
                    key={v.id}
                    className="p-3.5 rounded-xl bg-[#09090B] border border-zinc-800 text-xs space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono font-bold text-[#EF4444]">
                        v{v.version}
                      </span>
                      <span className="font-mono text-[11px] text-[#A1A1AA]">
                        {new Date(v.created_at).toLocaleString('pt-BR')}
                      </span>
                    </div>
                    <p className="text-zinc-200">{v.changelog}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ====================================================================
          MODAL 4: IMPORT JSON BUNDLE
      ==================================================================== */}
      {showImportModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#111114] border border-zinc-800 rounded-2xl max-w-2xl w-full p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div>
                <h3 className="text-base font-bold text-white">
                  Importar Bundle Versionado de Agentes (JSON)
                </h3>
                <p className="text-xs text-[#A1A1AA]">
                  Importe definições completas para preencher os 12 slots
                  oficiais ou adicionar novos agentes.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowImportModal(false)}
                className="text-zinc-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleConfirmImport} className="space-y-4 text-xs">
              <div>
                <label className="block text-zinc-300 font-medium mb-1">
                  Selecionar Arquivo `.json`
                </label>
                <input
                  type="file"
                  accept=".json,application/json"
                  onChange={handleImportFileSelect}
                  className="w-full text-xs text-zinc-300 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-[#18181C] file:text-white hover:file:bg-zinc-700"
                />
              </div>

              <div>
                <label className="block text-zinc-300 font-medium mb-1">
                  Conteúdo do Bundle JSON (`AgentExportBundle`)
                </label>
                <textarea
                  rows={8}
                  required
                  value={importJsonText}
                  onChange={(e) => setImportJsonText(e.target.value)}
                  placeholder='{"schema_version": "isee-agent-definition/v2.0", "exported_at": "2026-11-05T00:00:00Z", "platform": "I.See Agents", "agents": [...]}'
                  className="w-full p-3 rounded-xl bg-[#09090B] border border-zinc-800 text-white font-mono text-xs"
                />
              </div>

              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowImportModal(false)}
                  className="px-4 py-2 rounded-xl bg-zinc-800 text-white"
                >
                  Cancelar
                </button>
                <button
                  type="submit"
                  disabled={importing}
                  className="px-5 py-2 rounded-xl bg-[#DC2626] hover:bg-[#EF4444] text-white font-semibold"
                >
                  {importing ? 'Importando...' : 'Confirmar Importação'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

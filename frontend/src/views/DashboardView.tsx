import React, { useEffect, useState } from 'react';
import {
  Activity,
  Bot,
  Cpu,
  Database,
  FileText,
  Layers,
  MessageSquare,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Clock,
  HardDrive,
  Server,
} from 'lucide-react';
import { api } from '../services/api';
import type { ActiveTab } from '../components/Layout';
import type { OllamaStatusSchema, SystemDashboardSchema } from '../types/api';

interface DashboardViewProps {
  onNavigate: (tab: ActiveTab) => void;
  onOllamaStatusUpdated: (status: OllamaStatusSchema) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  onNavigate,
  onOllamaStatusUpdated,
}) => {
  const [dashboard, setDashboard] = useState<SystemDashboardSchema | null>(
    null
  );
  const [loading, setLoading] = useState(true);
  const [checkingOllama, setCheckingOllama] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadDashboard = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getDashboard();
      setDashboard(data);
      onOllamaStatusUpdated(data.ollama_status);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Erro ao carregar telemetria.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleRecheckOllama = async () => {
    setCheckingOllama(true);
    try {
      const status = await api.getOllamaStatus();
      onOllamaStatusUpdated(status);
      setDashboard((prev) =>
        prev ? { ...prev, ollama_status: status } : prev
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Erro ao verificar conectividade Ollama.'
      );
    } finally {
      setCheckingOllama(false);
    }
  };

  useEffect(() => {
    void loadDashboard();
  }, []);

  if (loading && !dashboard) {
    return (
      <div className="flex items-center justify-center py-24 text-sm text-[#A1A1AA] gap-3">
        <RefreshCw className="w-5 h-5 animate-spin text-[#DC2626]" />
        <span>Consultando telemetria em tempo real da plataforma...</span>
      </div>
    );
  }

  if (error && !dashboard) {
    return (
      <div className="p-6 rounded-2xl bg-[#111114] border border-red-900/60 text-red-200 space-y-4">
        <div className="flex items-center gap-2 font-semibold">
          <AlertTriangle className="w-5 h-5 text-[#EF4444]" />
          <span>Falha ao carregar o Dashboard Administrativo</span>
        </div>
        <p className="text-xs text-[#A1A1AA]">{error}</p>
        <button
          type="button"
          onClick={loadDashboard}
          className="px-4 py-2 rounded-lg bg-[#DC2626] hover:bg-[#EF4444] text-white text-xs font-semibold transition-colors"
        >
          Tentar Novamente
        </button>
      </div>
    );
  }

  if (!dashboard) return null;

  const { ollama_status: ollama } = dashboard;
  const slotsProgressPct = Math.round(
    (dashboard.original_slots_integrated / dashboard.original_slots_total) * 100
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800/80 pb-5">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold tracking-tight text-white">
              Dashboard Operacional & Telemetria
            </h1>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-[#18181C] text-[#A1A1AA] border border-zinc-800">
              v{dashboard.version}
            </span>
          </div>
          <p className="text-xs text-[#A1A1AA] mt-1">
            Métricas reais extraídas do banco de dados ({dashboard.database_engine})
            e status do motor de inferência local.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={loadDashboard}
            disabled={loading}
            className="px-3.5 py-2 rounded-xl bg-[#18181C] hover:bg-zinc-800 border border-zinc-800 text-xs font-medium text-white flex items-center gap-2 transition-colors"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 text-[#EF4444] ${
                loading ? 'animate-spin' : ''
              }`}
            />
            Atualizar Indicadores
          </button>
          <button
            type="button"
            onClick={() => onNavigate('chat')}
            className="px-4 py-2 rounded-xl bg-[#DC2626] hover:bg-[#EF4444] text-xs font-semibold text-white flex items-center gap-2 transition-colors shadow-lg shadow-[#DC2626]/20"
          >
            Abrir Chat Multi-Agente
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* 12 Original Slots Architecture Banner */}
      <div className="p-5 rounded-2xl bg-[#111114] border border-zinc-800/90 relative overflow-hidden">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-5">
          <div className="space-y-2 max-w-3xl">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono uppercase tracking-wider bg-[#DC2626]/15 text-[#EF4444] border border-[#DC2626]/30 font-semibold">
                Arquitetura Preparada para 12 Agentes Especializados
              </span>
              <span className="text-xs font-mono text-[#A1A1AA]">
                Slots Integrados: {dashboard.original_slots_integrated} /{' '}
                {dashboard.original_slots_total}
              </span>
            </div>
            <h2 className="text-base font-bold text-white">
              Status de Integração dos 12 Slots Oficiais da I.See Technologies
            </h2>
            <p className="text-xs text-[#A1A1AA] leading-relaxed">
              A plataforma possui{' '}
              <strong className="text-white">
                {dashboard.demo_agents_count} agentes demonstrativos (`[DEMO]`)
              </strong>{' '}
              operacionais para validação imediata de RAG, ferramentas e
              guardrails, além de{' '}
              <strong className="text-white">
                12 slots estruturais reservados (`slot_number` #01 a #12)
              </strong>
              . Atualmente,{' '}
              <strong className="text-amber-400">
                {dashboard.original_slots_awaiting} slots aguardam a importação
                futura
              </strong>{' '}
              das especificações definitivas dos 12 agentes originais sem
              necessidade de refatorar o banco de dados, API ou frontend.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row lg:flex-col items-stretch sm:items-center lg:items-end gap-3 shrink-0">
            <div className="w-full sm:w-56 bg-[#09090B] p-3 rounded-xl border border-zinc-800">
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="text-[#A1A1AA]">Slots Originais</span>
                <span className="font-mono font-bold text-white">
                  {dashboard.original_slots_integrated}/12 ({slotsProgressPct}%)
                </span>
              </div>
              <div className="w-full h-2 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-[#DC2626] transition-all duration-500"
                  style={{ width: `${Math.max(6, slotsProgressPct)}%` }}
                />
              </div>
            </div>

            <button
              type="button"
              onClick={() => onNavigate('agents')}
              className="px-4 py-2 rounded-xl bg-[#18181C] hover:bg-zinc-800 border border-zinc-700 text-xs font-medium text-white flex items-center justify-center gap-2 transition-colors"
            >
              Gerenciar Catálogo & 12 Slots
              <ArrowRight className="w-3.5 h-3.5 text-[#EF4444]" />
            </button>
          </div>
        </div>
      </div>

      {/* Primary KPI Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl bg-[#111114] border border-zinc-800/90 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#A1A1AA]">
              Total de Agentes no Registro
            </span>
            <Bot className="w-4 h-4 text-[#EF4444]" />
          </div>
          <div className="mt-4 flex items-baseline gap-2">
            <span className="text-3xl font-bold font-mono text-white">
              {dashboard.total_agents}
            </span>
            <span className="text-xs text-[#A1A1AA]">
              ({dashboard.demo_agents_count} em modo `[DEMO]`)
            </span>
          </div>
          <div className="mt-3 pt-3 border-t border-zinc-800/70 flex items-center justify-between text-[11px] text-[#A1A1AA]">
            <span>12 Slots Oficiais + Custom</span>
            <button
              type="button"
              onClick={() => onNavigate('agents')}
              className="text-[#EF4444] hover:underline font-medium"
            >
              Ver catálogo →
            </button>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-[#111114] border border-zinc-800/90 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#A1A1AA]">
              Bases de Conhecimento & RAG
            </span>
            <Database className="w-4 h-4 text-[#EF4444]" />
          </div>
          <div className="mt-4 flex items-baseline gap-2">
            <span className="text-3xl font-bold font-mono text-white">
              {dashboard.total_knowledge_bases}
            </span>
            <span className="text-xs text-[#A1A1AA]">bases ativas</span>
          </div>
          <div className="mt-3 pt-3 border-t border-zinc-800/70 flex items-center justify-between text-[11px] text-[#A1A1AA]">
            <span>
              {dashboard.total_documents} docs • {dashboard.total_vector_chunks}{' '}
              chunks
            </span>
            <button
              type="button"
              onClick={() => onNavigate('knowledge')}
              className="text-[#EF4444] hover:underline font-medium"
            >
              Pipeline RAG →
            </button>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-[#111114] border border-zinc-800/90 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#A1A1AA]">
              Conversas & Interações
            </span>
            <MessageSquare className="w-4 h-4 text-[#EF4444]" />
          </div>
          <div className="mt-4 flex items-baseline gap-2">
            <span className="text-3xl font-bold font-mono text-white">
              {dashboard.total_conversations}
            </span>
            <span className="text-xs text-[#A1A1AA]">
              sessões ({dashboard.total_messages} msgs)
            </span>
          </div>
          <div className="mt-3 pt-3 border-t border-zinc-800/70 flex items-center justify-between text-[11px] text-[#A1A1AA]">
            <span>Histórico persistente RBAC</span>
            <button
              type="button"
              onClick={() => onNavigate('chat')}
              className="text-[#EF4444] hover:underline font-medium"
            >
              Iniciar chat →
            </button>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-[#111114] border border-zinc-800/90 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[#A1A1AA]">
              Latência Média de Inferência
            </span>
            <Clock className="w-4 h-4 text-[#EF4444]" />
          </div>
          <div className="mt-4 flex items-baseline gap-2">
            <span className="text-3xl font-bold font-mono text-white">
              {dashboard.avg_latency_ms}
            </span>
            <span className="text-xs font-mono text-[#A1A1AA]">ms / turn</span>
          </div>
          <div className="mt-3 pt-3 border-t border-zinc-800/70 flex items-center justify-between text-[11px] text-[#A1A1AA]">
            <span>Medição real de respostas</span>
            <button
              type="button"
              onClick={() => onNavigate('models')}
              className="text-[#EF4444] hover:underline font-medium"
            >
              Ver modelos →
            </button>
          </div>
        </div>
      </div>

      {/* Ollama Real-Time Status & Database Infrastructure */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Ollama Runtime Card */}
        <div className="lg:col-span-7 p-6 rounded-2xl bg-[#111114] border border-zinc-800/90 space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-800/80 pb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-[#18181C] border border-zinc-800 flex items-center justify-center">
                <Cpu className="w-5 h-5 text-[#EF4444]" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  Motor de Inferência Local — Ollama Runtime
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase ${
                      ollama.reachable
                        ? 'bg-emerald-950/70 text-emerald-300 border border-emerald-700/60'
                        : 'bg-amber-950/70 text-amber-300 border border-amber-700/60'
                    }`}
                  >
                    {ollama.active_mode === 'ollama_live'
                      ? 'OLLAMA_LIVE'
                      : 'DEMO_FALLBACK'}
                  </span>
                </h3>
                <p className="text-xs text-[#A1A1AA]">
                  Endpoint configurado:{' '}
                  <code className="font-mono text-zinc-300">
                    {ollama.base_url}
                  </code>
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={handleRecheckOllama}
              disabled={checkingOllama}
              className="px-3.5 py-2 rounded-xl bg-[#18181C] hover:bg-zinc-800 border border-zinc-700 text-xs font-medium text-white flex items-center gap-2 transition-colors self-start sm:self-auto"
            >
              <RefreshCw
                className={`w-3.5 h-3.5 text-[#EF4444] ${
                  checkingOllama ? 'animate-spin' : ''
                }`}
              />
              Re-verificar Conectividade
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="p-3.5 rounded-xl bg-[#09090B] border border-zinc-800/80">
              <div className="text-[11px] text-[#A1A1AA]">Conectividade</div>
              <div className="mt-1 flex items-center gap-1.5 text-xs font-semibold">
                {ollama.reachable ? (
                  <>
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    <span className="text-emerald-300">Online & Acessível</span>
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                    <span className="text-amber-300">
                      Offline (Modo Demo Ativo)
                    </span>
                  </>
                )}
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-[#09090B] border border-zinc-800/80">
              <div className="text-[11px] text-[#A1A1AA]">Ping do Servidor</div>
              <div className="mt-1 text-xs font-mono font-semibold text-white">
                {ollama.latency_ms !== null
                  ? `${ollama.latency_ms} ms`
                  : 'N/A (Fallback Local)'}
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-[#09090B] border border-zinc-800/80">
              <div className="text-[11px] text-[#A1A1AA]">
                Modelos Detectados
              </div>
              <div className="mt-1 text-xs font-mono font-semibold text-white">
                {ollama.installed_models.length} modelo(s)
              </div>
            </div>
          </div>

          {ollama.installed_models.length > 0 ? (
            <div>
              <div className="text-xs font-medium text-zinc-300 mb-2">
                Modelos Instalados no Ollama Local:
              </div>
              <div className="flex flex-wrap gap-2">
                {ollama.installed_models.map((m) => (
                  <span
                    key={m}
                    className="px-2.5 py-1 rounded-lg bg-[#09090B] border border-zinc-800 text-xs font-mono text-emerald-300"
                  >
                    {m}
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <div className="p-3.5 rounded-xl bg-[#18181C] border border-zinc-800 text-xs text-[#A1A1AA] space-y-1">
              <div className="font-semibold text-zinc-200">
                Operando com Provider Demonstrativo Determinístico (`[DEMO]`)
              </div>
              <p>
                {ollama.error_detail
                  ? `Detalhe técnico: ${ollama.error_detail}. `
                  : ''}
                Para ativar inferência LLM real via Ollama, inicie{' '}
                <code className="font-mono text-white">ollama serve</code> e
                execute{' '}
                <code className="font-mono text-white">
                  ollama pull llama3.2:3b
                </code>
                .
              </p>
            </div>
          )}
        </div>

        {/* Persistence & Vector Engine Card */}
        <div className="lg:col-span-5 p-6 rounded-2xl bg-[#111114] border border-zinc-800/90 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="flex items-center gap-3 border-b border-zinc-800/80 pb-4">
              <div className="w-10 h-10 rounded-xl bg-[#18181C] border border-zinc-800 flex items-center justify-center">
                <Server className="w-5 h-5 text-[#EF4444]" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white">
                  Infraestrutura de Dados & Indexação Vetorial
                </h3>
                <p className="text-xs text-[#A1A1AA]">
                  Persistência relacional, embeddings e segurança
                </p>
              </div>
            </div>

            <div className="space-y-2.5 text-xs">
              <div className="flex items-center justify-between p-2.5 rounded-lg bg-[#09090B] border border-zinc-800/80">
                <span className="text-[#A1A1AA] flex items-center gap-2">
                  <HardDrive className="w-3.5 h-3.5 text-[#EF4444]" />
                  Engine de Banco de Dados
                </span>
                <span className="font-mono font-semibold text-white">
                  {dashboard.database_engine}
                </span>
              </div>

              <div className="flex items-center justify-between p-2.5 rounded-lg bg-[#09090B] border border-zinc-800/80">
                <span className="text-[#A1A1AA] flex items-center gap-2">
                  <Layers className="w-3.5 h-3.5 text-[#EF4444]" />
                  Extensão Vetorial pgvector
                </span>
                <span className="font-mono text-zinc-300">
                  {dashboard.pgvector_enabled
                    ? 'Ativo (PostgreSQL pgvector)'
                    : 'Cosine Vector Index (Hybrid Runtime)'}
                </span>
              </div>

              <div className="flex items-center justify-between p-2.5 rounded-lg bg-[#09090B] border border-zinc-800/80">
                <span className="text-[#A1A1AA] flex items-center gap-2">
                  <FileText className="w-3.5 h-3.5 text-[#EF4444]" />
                  Documentos / Chunks Vetoriais
                </span>
                <span className="font-mono font-semibold text-white">
                  {dashboard.total_documents} docs /{' '}
                  {dashboard.total_vector_chunks} chunks
                </span>
              </div>

              <div className="flex items-center justify-between p-2.5 rounded-lg bg-[#09090B] border border-zinc-800/80">
                <span className="text-[#A1A1AA] flex items-center gap-2">
                  <Activity className="w-3.5 h-3.5 text-[#EF4444]" />
                  Guardrails Anti-Prompt Injection
                </span>
                <span className="font-mono text-emerald-400 font-semibold">
                  ATIVO (RBAC + SHA-256)
                </span>
              </div>
            </div>
          </div>

          <div className="pt-2 flex items-center justify-between gap-2">
            <button
              type="button"
              onClick={() => onNavigate('knowledge')}
              className="flex-1 py-2 px-3 rounded-xl bg-[#18181C] hover:bg-zinc-800 border border-zinc-700 text-xs font-medium text-white transition-colors"
            >
              Ingerir Documentos RAG
            </button>
            <button
              type="button"
              onClick={() => onNavigate('models')}
              className="flex-1 py-2 px-3 rounded-xl bg-[#18181C] hover:bg-zinc-800 border border-zinc-700 text-xs font-medium text-white transition-colors"
            >
              Auditoria & Adaptadores LoRA
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

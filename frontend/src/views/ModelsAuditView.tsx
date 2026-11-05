import React, { useEffect, useState } from 'react';
import {
  Cpu,
  Plus,
  Trash2,
  ShieldCheck,
  RefreshCw,
  Layers,
  AlertCircle,
  CheckCircle2,
  X,
  Terminal,
} from 'lucide-react';
import { api } from '../services/api';
import type {
  AdapterMethod,
  AuditLogRead,
  ModelConfigRead,
  ModelProvider,
} from '../types/api';

export const ModelsAuditView: React.FC = () => {
  const [models, setModels] = useState<ModelConfigRead[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditLogRead[]>([]);
  const [severityFilter, setSeverityFilter] = useState<string>('ALL');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  // Create Model Modal
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [name, setName] = useState('');
  const [provider, setProvider] = useState<ModelProvider>('ollama');
  const [baseModel, setBaseModel] = useState('llama3.2:3b');
  const [version, setVersion] = useState('1.0.0');
  const [quantization, setQuantization] = useState('Q4_K_M');
  const [contextWindow, setContextWindow] = useState(8192);
  const [temperature, setTemperature] = useState(0.3);
  const [adapterType, setAdapterType] = useState<AdapterMethod>('none');
  const [adapterPath, setAdapterPath] = useState('');
  const [adapterRank, setAdapterRank] = useState(16);
  const [adapterAlpha, setAdapterAlpha] = useState(32);
  const [isDemo, setIsDemo] = useState(false);
  const [creating, setCreating] = useState(false);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [modelsList, logsList] = await Promise.all([
        api.listModels(),
        api.listAuditLogs(80),
      ]);
      setModels(modelsList);
      setAuditLogs(logsList);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Erro ao carregar configurações de modelos e auditoria.'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadData();
  }, []);

  const handleCreateModel = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    setError(null);
    try {
      const created = await api.createModel({
        name,
        provider,
        base_model: baseModel,
        version,
        quantization: quantization || null,
        context_window: Number(contextWindow),
        default_temperature: Number(temperature),
        default_top_p: 0.9,
        default_max_tokens: 1024,
        adapter_type: adapterType,
        adapter_path:
          adapterType !== 'none' ? adapterPath.trim() || null : null,
        adapter_rank: adapterType !== 'none' ? Number(adapterRank) : null,
        adapter_alpha: adapterType !== 'none' ? Number(adapterAlpha) : null,
        is_demo: isDemo,
      });
      setNotice(`Perfil de modelo "${created.name}" registrado.`);
      setShowCreateModal(false);
      setName('');
      await loadData();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Erro ao registrar modelo.'
      );
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteModel = async (m: ModelConfigRead) => {
    try {
      await api.deleteModel(m.id);
      setNotice(`Modelo "${m.name}" removido do registro.`);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro ao remover modelo.');
    }
  };

  const filteredLogs = auditLogs.filter((l) =>
    severityFilter === 'ALL' ? true : l.severity === severityFilter
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800/80 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            Modelos LLM, Adaptadores LoRA / QLoRA & Auditoria
          </h1>
          <p className="text-xs text-[#A1A1AA] mt-1">
            Gerencie perfis de modelos (`Ollama`, `Hugging Face PEFT 4-bit NF4`,
            `Demo`) e inspecione os registros de segurança e auditoria da
            plataforma.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={loadData}
            className="px-3.5 py-2 rounded-xl bg-[#18181C] hover:bg-zinc-800 border border-zinc-700 text-xs font-medium text-white flex items-center gap-1.5"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 text-[#EF4444] ${
                loading ? 'animate-spin' : ''
              }`}
            />
            Atualizar
          </button>

          <button
            type="button"
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 rounded-xl bg-[#DC2626] hover:bg-[#EF4444] text-xs font-semibold text-white flex items-center gap-1.5 shadow-lg shadow-[#DC2626]/20"
          >
            <Plus className="w-4 h-4" />
            Novo Perfil de Modelo / LoRA
          </button>
        </div>
      </div>

      {notice && (
        <div className="p-3.5 rounded-xl bg-emerald-950/40 border border-emerald-800/70 text-emerald-200 text-xs flex items-center justify-between">
          <span className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            {notice}
          </span>
          <button
            type="button"
            onClick={() => setNotice(null)}
            className="text-emerald-300 hover:text-white"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {error && (
        <div className="p-3.5 rounded-xl bg-red-950/50 border border-red-800 text-red-200 text-xs flex items-center justify-between">
          <span className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-[#EF4444]" />
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

      {/* Model Configurations Registry */}
      <div className="space-y-3">
        <h2 className="text-base font-bold text-white flex items-center gap-2">
          <Cpu className="w-4 h-4 text-[#EF4444]" />
          Perfis de Modelos & Adaptadores PEFT Registrados ({models.length})
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((m) => (
            <div
              key={m.id}
              className="p-5 rounded-2xl bg-[#111114] border border-zinc-800/90 flex flex-col justify-between space-y-4"
            >
              <div className="space-y-2.5">
                <div className="flex items-center justify-between gap-2">
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase ${
                      m.is_demo
                        ? 'bg-amber-950/70 text-amber-300 border border-amber-700/50'
                        : 'bg-emerald-950/70 text-emerald-300 border border-emerald-700/50'
                    }`}
                  >
                    {m.is_demo ? '[DEMO PROFILE]' : m.provider.toUpperCase()}
                  </span>
                  <span className="text-[11px] font-mono text-[#A1A1AA]">
                    v{m.version}
                  </span>
                </div>

                <div>
                  <h3 className="text-sm font-bold text-white">{m.name}</h3>
                  <div className="text-xs font-mono text-[#EF4444] mt-0.5">
                    {m.base_model}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 pt-1 text-[11px] font-mono">
                  <div className="p-2 rounded-lg bg-[#09090B] border border-zinc-800">
                    <span className="text-[#A1A1AA] block text-[10px]">
                      Quantização
                    </span>
                    <span className="text-white">
                      {m.quantization || 'FP16 / Default'}
                    </span>
                  </div>
                  <div className="p-2 rounded-lg bg-[#09090B] border border-zinc-800">
                    <span className="text-[#A1A1AA] block text-[10px]">
                      Janela de Contexto
                    </span>
                    <span className="text-white">{m.context_window} tok</span>
                  </div>
                </div>

                <div className="p-2.5 rounded-lg bg-[#09090B] border border-zinc-800 text-[11px] flex items-center justify-between">
                  <span className="text-[#A1A1AA] flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-[#EF4444]" />
                    Adaptador:
                  </span>
                  <span className="font-mono text-white">
                    {m.adapter_type && m.adapter_type !== 'none'
                      ? `${m.adapter_type.toUpperCase()} (r=${m.adapter_rank}, α=${m.adapter_alpha})`
                      : 'Nenhum (Base)'}
                  </span>
                </div>
              </div>

              <div className="pt-3 border-t border-zinc-800 flex items-center justify-between text-[11px] text-[#A1A1AA]">
                <span>Temp padrão: {m.default_temperature}</span>
                <button
                  type="button"
                  onClick={() => handleDeleteModel(m)}
                  className="text-zinc-400 hover:text-red-400 flex items-center gap-1"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Remover
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Audit & Security Logs Table */}
      <div className="p-5 rounded-2xl bg-[#111114] border border-zinc-800/90 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-800 pb-4">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-[#EF4444]" />
              Trilha de Auditoria, Segurança & Eventos RBAC ({filteredLogs.length}
              )
            </h2>
            <p className="text-xs text-[#A1A1AA]">
              Monitoramento real de autenticações, bloqueios de Prompt Injection
              e alterações de slots.
            </p>
          </div>

          <div className="flex items-center gap-1.5">
            {['ALL', 'INFO', 'WARNING', 'CRITICAL'].map((sev) => (
              <button
                key={sev}
                type="button"
                onClick={() => setSeverityFilter(sev)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-colors ${
                  severityFilter === sev
                    ? 'bg-[#DC2626] text-white'
                    : 'bg-[#09090B] text-[#A1A1AA] hover:text-white border border-zinc-800'
                }`}
              >
                {sev}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-zinc-800 text-[#A1A1AA] font-mono text-[11px]">
                <th className="py-2.5 px-3">Timestamp</th>
                <th className="py-2.5 px-3">Severidade</th>
                <th className="py-2.5 px-3">Evento</th>
                <th className="py-2.5 px-3">Recurso</th>
                <th className="py-2.5 px-3">Detalhes Técnicos</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/70">
              {filteredLogs.map((log) => (
                <tr key={log.id} className="hover:bg-[#18181C]/60">
                  <td className="py-2.5 px-3 font-mono text-[11px] text-zinc-400 whitespace-nowrap">
                    {new Date(log.created_at).toLocaleString('pt-BR')}
                  </td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${
                        log.severity === 'WARNING' ||
                        log.severity === 'CRITICAL'
                          ? 'bg-red-950/80 text-red-300 border border-red-800'
                          : 'bg-zinc-800 text-zinc-200'
                      }`}
                    >
                      {log.severity}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 font-mono font-semibold text-white">
                    {log.event_type}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-[#EF4444]">
                    {log.resource}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-[11px] text-[#A1A1AA]">
                    <code className="flex items-center gap-1">
                      <Terminal className="w-3 h-3 text-zinc-500 shrink-0" />
                      {JSON.stringify(log.details)}
                    </code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modal: Create Model Profile */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#111114] border border-zinc-800 rounded-2xl max-w-lg w-full p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <h3 className="text-base font-bold text-white">
                Registrar Perfil de Modelo / Adaptador LoRA
              </h3>
              <button
                type="button"
                onClick={() => setShowCreateModal(false)}
                className="text-zinc-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleCreateModel} className="space-y-3.5 text-xs">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="sm:col-span-2">
                  <label className="block text-zinc-300 font-medium mb-1">
                    Nome do Perfil
                  </label>
                  <input
                    type="text"
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Ex: Qwen 2.5 Coder 7B + QLoRA NF4"
                    className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white"
                  />
                </div>
                <div>
                  <label className="block text-zinc-300 font-medium mb-1">
                    Versão
                  </label>
                  <input
                    type="text"
                    required
                    value={version}
                    onChange={(e) => setVersion(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white font-mono"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-zinc-300 font-medium mb-1">
                    Provedor
                  </label>
                  <select
                    value={provider}
                    onChange={(e) =>
                      setProvider(e.target.value as ModelProvider)
                    }
                    className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white"
                  >
                    <option value="ollama">Ollama Local</option>
                    <option value="huggingface">Hugging Face / PEFT</option>
                    <option value="demo">Demo Fallback</option>
                  </select>
                </div>

                <div>
                  <label className="block text-zinc-300 font-medium mb-1">
                    Modelo Base
                  </label>
                  <input
                    type="text"
                    required
                    value={baseModel}
                    onChange={(e) => setBaseModel(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white font-mono"
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div>
                  <label className="block text-[#A1A1AA] mb-1">
                    Quantização
                  </label>
                  <input
                    type="text"
                    value={quantization}
                    onChange={(e) => setQuantization(e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded bg-[#09090B] border border-zinc-800 text-white font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[#A1A1AA] mb-1">Contexto</label>
                  <input
                    type="number"
                    value={contextWindow}
                    onChange={(e) => setContextWindow(Number(e.target.value))}
                    className="w-full px-2.5 py-1.5 rounded bg-[#09090B] border border-zinc-800 text-white font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[#A1A1AA] mb-1">
                    Temperatura
                  </label>
                  <input
                    type="number"
                    step="0.05"
                    value={temperature}
                    onChange={(e) => setTemperature(Number(e.target.value))}
                    className="w-full px-2.5 py-1.5 rounded bg-[#09090B] border border-zinc-800 text-white font-mono"
                  />
                </div>
              </div>

              <div>
                <label className="block text-zinc-300 font-medium mb-1">
                  Tipo de Adaptador Fine-Tuning
                </label>
                <select
                  value={adapterType}
                  onChange={(e) =>
                    setAdapterType(e.target.value as AdapterMethod)
                  }
                  className="w-full px-3 py-2 rounded-lg bg-[#09090B] border border-zinc-800 text-white"
                >
                  <option value="none">Sem Adaptador (Modelo Base)</option>
                  <option value="lora">LoRA (Low-Rank Adaptation)</option>
                  <option value="qlora">QLoRA (4-bit NormalFloat NF4)</option>
                </select>
              </div>

              {adapterType !== 'none' && (
                <div className="grid grid-cols-2 gap-3 p-3 rounded-xl bg-[#09090B] border border-zinc-800">
                  <div>
                    <label className="block text-[#A1A1AA] mb-1">
                      Rank (r)
                    </label>
                    <input
                      type="number"
                      value={adapterRank}
                      onChange={(e) => setAdapterRank(Number(e.target.value))}
                      className="w-full px-2.5 py-1.5 rounded bg-[#111114] border border-zinc-800 text-white font-mono"
                    />
                  </div>
                  <div>
                    <label className="block text-[#A1A1AA] mb-1">
                      Alpha (α)
                    </label>
                    <input
                      type="number"
                      value={adapterAlpha}
                      onChange={(e) => setAdapterAlpha(Number(e.target.value))}
                      className="w-full px-2.5 py-1.5 rounded bg-[#111114] border border-zinc-800 text-white font-mono"
                    />
                  </div>
                  <div className="col-span-2">
                    <label className="block text-[#A1A1AA] mb-1">
                      Diretório / Path dos Pesos PEFT
                    </label>
                    <input
                      type="text"
                      value={adapterPath}
                      onChange={(e) => setAdapterPath(e.target.value)}
                      placeholder="./adapters/isee-qlora-v1"
                      className="w-full px-2.5 py-1.5 rounded bg-[#111114] border border-zinc-800 text-white font-mono"
                    />
                  </div>
                </div>
              )}

              <label className="flex items-center gap-2 cursor-pointer pt-1">
                <input
                  type="checkbox"
                  checked={isDemo}
                  onChange={(e) => setIsDemo(e.target.checked)}
                  className="accent-[#DC2626]"
                />
                <span>Marcar como Perfil Demonstrativo (`[DEMO]`)</span>
              </label>

              <div className="flex justify-end gap-2 pt-3 border-t border-zinc-800">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="px-4 py-2 rounded-xl bg-zinc-800 text-white"
                >
                  Cancelar
                </button>
                <button
                  type="submit"
                  disabled={creating}
                  className="px-5 py-2 rounded-xl bg-[#DC2626] hover:bg-[#EF4444] text-white font-semibold"
                >
                  {creating ? 'Salvando...' : 'Registrar Modelo'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

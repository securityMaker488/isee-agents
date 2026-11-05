import React, { useEffect, useRef, useState } from 'react';
import {
  Bot,
  Send,
  Plus,
  Trash2,
  Database,
  Sliders,
  ShieldAlert,
  Clock,
  Zap,
  ChevronDown,
  ChevronUp,
  Sparkles,
  User as UserIcon,
  MessageSquare,
  AlertCircle,
  FileText,
} from 'lucide-react';
import { api } from '../services/api';
import type {
  AgentRead,
  ChatMessageRead,
  ConversationRead,
} from '../types/api';

interface ChatViewProps {
  initialAgentId: string | null;
}

export const ChatView: React.FC<ChatViewProps> = ({ initialAgentId }) => {
  const [agents, setAgents] = useState<AgentRead[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string>('');
  const [conversations, setConversations] = useState<ConversationRead[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<
    string | null
  >(null);
  const [messages, setMessages] = useState<ChatMessageRead[]>([]);
  const [inputPrompt, setInputPrompt] = useState('');
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Inference controls
  const [useRag, setUseRag] = useState(true);
  const [temperature, setTemperature] = useState(0.3);
  const [topP, setTopP] = useState(0.9);
  const [expandedCitations, setExpandedCitations] = useState<
    Record<string, boolean>
  >({});

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const selectedAgent = agents.find((a) => a.id === selectedAgentId) || null;

  const loadInitial = async () => {
    try {
      const [agentsList, convsList] = await Promise.all([
        api.listAgents({ include_slots: true }),
        api.listConversations(),
      ]);
      setAgents(agentsList);
      setConversations(convsList);

      if (initialAgentId && agentsList.some((a) => a.id === initialAgentId)) {
        setSelectedAgentId(initialAgentId);
      } else if (agentsList.length > 0) {
        const firstReady =
          agentsList.find((a) => a.slot_number === null) || agentsList[0];
        setSelectedAgentId(firstReady.id);
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Erro ao inicializar console chat.'
      );
    }
  };

  useEffect(() => {
    void loadInitial();
  }, [initialAgentId]);

  useEffect(() => {
    if (selectedAgent) {
      setUseRag(selectedAgent.rag_config.enabled);
      setTemperature(selectedAgent.inference_config.temperature);
      setTopP(selectedAgent.inference_config.top_p);
    }
  }, [selectedAgentId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSelectConversation = async (conv: ConversationRead) => {
    setActiveConversationId(conv.id);
    setSelectedAgentId(conv.agent_id);
    setError(null);
    try {
      const msgs = await api.listConversationMessages(conv.id);
      setMessages(msgs);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Erro ao carregar mensagens.'
      );
    }
  };

  const handleNewConversation = () => {
    setActiveConversationId(null);
    setMessages([]);
    setError(null);
  };

  const handleDeleteConversation = async (
    e: React.MouseEvent,
    convId: string
  ) => {
    e.stopPropagation();
    try {
      await api.deleteConversation(convId);
      setConversations((prev) => prev.filter((c) => c.id !== convId));
      if (activeConversationId === convId) {
        setActiveConversationId(null);
        setMessages([]);
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Erro ao remover conversa.'
      );
    }
  };

  const handleSendMessage = async (e?: React.FormEvent, customText?: string) => {
    if (e) e.preventDefault();
    const promptText = (customText ?? inputPrompt).trim();
    if (!promptText || !selectedAgentId || sending) return;

    setSending(true);
    setError(null);
    if (!customText) {
      setInputPrompt('');
    }

    try {
      const response = await api.sendChatMessage({
        agent_id: selectedAgentId,
        message: promptText,
        conversation_id: activeConversationId,
        override_temperature: temperature,
        override_top_p: topP,
        use_rag: useRag,
      });

      setActiveConversationId(response.conversation.id);
      setMessages((prev) => [
        ...prev,
        response.user_message,
        response.assistant_message,
      ]);

      // Refresh conversation list
      const updatedConvs = await api.listConversations();
      setConversations(updatedConvs);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Falha ao processar mensagem no orquestrador.'
      );
    } finally {
      setSending(false);
    }
  };

  const toggleCitations = (msgId: string) => {
    setExpandedCitations((prev) => ({
      ...prev,
      [msgId]: !prev[msgId],
    }));
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-800/80 pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            Console de Chat Multi-Agente & Orquestração RAG
          </h1>
          <p className="text-xs text-[#A1A1AA] mt-0.5">
            Interaja com agentes especialistas, inspecione citações vetoriais
            recuperadas, latência real (`ms`) e guardrails de segurança.
          </p>
        </div>

        <button
          type="button"
          onClick={handleNewConversation}
          className="px-4 py-2 rounded-xl bg-[#DC2626] hover:bg-[#EF4444] text-xs font-semibold text-white flex items-center gap-1.5 transition-colors self-start sm:self-auto shadow-lg shadow-[#DC2626]/20"
        >
          <Plus className="w-4 h-4" />
          Nova Sessão de Chat
        </button>
      </div>

      {error && (
        <div className="p-3 rounded-xl bg-red-950/50 border border-red-800 text-red-200 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-[#EF4444] shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Layout Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left Control Column */}
        <div className="lg:col-span-4 space-y-4">
          {/* Active Agent Selector */}
          <div className="p-4 rounded-2xl bg-[#111114] border border-zinc-800/90 space-y-3">
            <label className="block text-xs font-semibold text-white flex items-center gap-2">
              <Bot className="w-4 h-4 text-[#EF4444]" />
              Agente Especialista Ativo
            </label>
            <select
              value={selectedAgentId}
              onChange={(e) => {
                setSelectedAgentId(e.target.value);
                setActiveConversationId(null);
                setMessages([]);
              }}
              className="w-full px-3 py-2.5 rounded-xl bg-[#09090B] border border-zinc-800 text-xs text-white focus:outline-none focus:border-[#DC2626]"
            >
              {agents.map((ag) => (
                <option key={ag.id} value={ag.id}>
                  {ag.slot_number !== null
                    ? `[SLOT #${String(ag.slot_number).padStart(2, '0')}] `
                    : '[DEMO] '}
                  {ag.name} ({ag.base_model})
                </option>
              ))}
            </select>

            {selectedAgent && (
              <div className="p-3 rounded-xl bg-[#09090B] border border-zinc-800/80 space-y-2 text-[11px]">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[#EF4444] font-semibold">
                    {selectedAgent.domain}
                  </span>
                  <span
                    className={`px-1.5 py-0.5 rounded font-mono text-[10px] ${
                      selectedAgent.is_demo
                        ? 'bg-amber-950/70 text-amber-300'
                        : 'bg-emerald-950/70 text-emerald-300'
                    }`}
                  >
                    {selectedAgent.is_demo ? '[DEMO]' : '[PRODUÇÃO]'}
                  </span>
                </div>
                <p className="text-[#A1A1AA] leading-relaxed">
                  {selectedAgent.purpose}
                </p>
                <div className="pt-1 flex flex-wrap gap-1.5 font-mono text-[10px] text-zinc-400">
                  <span className="px-2 py-0.5 rounded bg-[#18181C]">
                    Model: {selectedAgent.base_model}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-[#18181C]">
                    LoRA: {selectedAgent.lora_adapter.method.toUpperCase()}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Inference Parameters Control */}
          <div className="p-4 rounded-2xl bg-[#111114] border border-zinc-800/90 space-y-3.5 text-xs">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-white flex items-center gap-2">
                <Sliders className="w-4 h-4 text-[#EF4444]" />
                Parâmetros de Inferência
              </span>
              <label className="flex items-center gap-1.5 cursor-pointer text-[11px]">
                <input
                  type="checkbox"
                  checked={useRag}
                  onChange={(e) => setUseRag(e.target.checked)}
                  className="accent-[#DC2626]"
                />
                <span className="text-zinc-200 font-medium">Usar RAG</span>
              </label>
            </div>

            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span className="text-[#A1A1AA]">Temperature</span>
                <span className="font-mono text-white">
                  {temperature.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.05}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="w-full accent-[#DC2626]"
              />
            </div>

            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span className="text-[#A1A1AA]">Top-P (Nucleus Sampling)</span>
                <span className="font-mono text-white">{topP.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={1.0}
                step={0.05}
                value={topP}
                onChange={(e) => setTopP(Number(e.target.value))}
                className="w-full accent-[#DC2626]"
              />
            </div>
          </div>

          {/* Conversation History List */}
          <div className="p-4 rounded-2xl bg-[#111114] border border-zinc-800/90 space-y-3">
            <div className="text-xs font-semibold text-white flex items-center justify-between">
              <span className="flex items-center gap-2">
                <MessageSquare className="w-4 h-4 text-[#EF4444]" />
                Conversas Persistentes ({conversations.length})
              </span>
            </div>

            <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
              {conversations.length === 0 ? (
                <div className="text-xs text-[#A1A1AA] py-4 text-center">
                  Nenhuma conversa registrada ainda.
                </div>
              ) : (
                conversations.map((conv) => (
                  <div
                    key={conv.id}
                    onClick={() => handleSelectConversation(conv)}
                    className={`p-2.5 rounded-xl border text-xs cursor-pointer transition-all flex items-center justify-between gap-2 ${
                      activeConversationId === conv.id
                        ? 'bg-[#18181C] border-[#DC2626] text-white'
                        : 'bg-[#09090B] border-zinc-800/80 text-zinc-300 hover:border-zinc-700'
                    }`}
                  >
                    <div className="min-w-0">
                      <div className="font-medium truncate">{conv.title}</div>
                      <div className="text-[10px] text-[#A1A1AA] font-mono truncate">
                        {conv.agent_name || 'Agente'} • {conv.message_count} msgs
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={(e) => handleDeleteConversation(e, conv.id)}
                      className="p-1 text-zinc-500 hover:text-red-400 shrink-0"
                      title="Excluir conversa"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Right Chat Stream Column */}
        <div className="lg:col-span-8 bg-[#111114] border border-zinc-800/90 rounded-2xl flex flex-col h-[680px]">
          {/* Messages Container */}
          <div className="flex-1 p-5 overflow-y-auto space-y-4">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center max-w-lg mx-auto space-y-4">
                <div className="w-12 h-12 rounded-2xl bg-[#DC2626]/15 border border-[#DC2626]/30 flex items-center justify-center">
                  <Sparkles className="w-6 h-6 text-[#EF4444]" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-white">
                    Pronto para Orquestrar {selectedAgent?.name || 'o Agente'}
                  </h3>
                  <p className="text-xs text-[#A1A1AA] mt-1">
                    Envie uma pergunta ou clique em um dos cenários de teste
                    abaixo para validar o pipeline RAG, citações vetoriais e
                    telemetria de inferência.
                  </p>
                </div>

                {selectedAgent && selectedAgent.test_cases.length > 0 && (
                  <div className="w-full space-y-2 pt-2">
                    <div className="text-[11px] font-mono uppercase text-[#A1A1AA]">
                      Prompts de Validação Sugeridos:
                    </div>
                    {selectedAgent.test_cases.map((tc, i) => (
                      <button
                        key={i}
                        type="button"
                        onClick={() =>
                          void handleSendMessage(undefined, tc.input_prompt)
                        }
                        className="w-full text-left p-3 rounded-xl bg-[#09090B] hover:bg-[#18181C] border border-zinc-800 hover:border-[#DC2626]/50 text-xs text-zinc-200 transition-all"
                      >
                        <div className="font-semibold text-[#EF4444] text-[11px] mb-0.5">
                          {tc.name}
                        </div>
                        <div>{tc.input_prompt}</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              messages.map((msg) => {
                const isUser = msg.role === 'user';
                const citationsOpen = !!expandedCitations[msg.id];

                return (
                  <div
                    key={msg.id}
                    className={`flex flex-col ${
                      isUser ? 'items-end' : 'items-start'
                    }`}
                  >
                    <div
                      className={`max-w-3xl w-full rounded-2xl p-4 space-y-3 ${
                        isUser
                          ? 'bg-[#DC2626] text-white ml-auto sm:w-auto sm:min-w-[280px]'
                          : 'bg-[#09090B] border border-zinc-800 text-zinc-100'
                      }`}
                    >
                      {/* Role & Provider Telemetry Bar */}
                      <div className="flex items-center justify-between gap-3 flex-wrap border-b border-white/10 pb-2 text-[11px]">
                        <span className="font-semibold flex items-center gap-1.5">
                          {isUser ? (
                            <>
                              <UserIcon className="w-3.5 h-3.5" />
                              Você
                            </>
                          ) : (
                            <>
                              <Bot className="w-3.5 h-3.5 text-[#EF4444]" />
                              {selectedAgent?.name || 'Agente I.See'}
                            </>
                          )}
                        </span>

                        {!isUser && (
                          <div className="flex items-center gap-1.5 flex-wrap font-mono text-[10px]">
                            <span
                              className={`px-2 py-0.5 rounded font-semibold ${
                                msg.is_demo_response
                                  ? 'bg-amber-950/80 text-amber-300 border border-amber-700/60'
                                  : 'bg-emerald-950/80 text-emerald-300 border border-emerald-700/60'
                              }`}
                            >
                              {msg.is_demo_response
                                ? 'MODO DEMO (Simulado)'
                                : 'Ollama (Real LLM)'}
                            </span>

                            <span className="px-2 py-0.5 rounded bg-[#18181C] text-zinc-300 flex items-center gap-1">
                              <Clock className="w-3 h-3 text-[#EF4444]" />
                              {msg.latency_ms} ms
                            </span>

                            <span className="px-2 py-0.5 rounded bg-[#18181C] text-zinc-300 flex items-center gap-1">
                              <Zap className="w-3 h-3 text-amber-400" />
                              {msg.tokens_per_second} tok/s
                            </span>
                          </div>
                        )}
                      </div>

                      {/* Security Guardrail Flags if triggered */}
                      {msg.security_flags && msg.security_flags.length > 0 && (
                        <div className="p-2.5 rounded-lg bg-red-950/70 border border-red-700 text-red-200 text-xs flex items-center gap-2 font-mono">
                          <ShieldAlert className="w-4 h-4 text-[#EF4444] shrink-0" />
                          <span>
                            Guardrail Acionado: {msg.security_flags.join(', ')}
                          </span>
                        </div>
                      )}

                      {/* Content */}
                      <div className="text-xs sm:text-sm whitespace-pre-wrap leading-relaxed">
                        {msg.content}
                      </div>

                      {/* Expandable RAG Citations */}
                      {!isUser &&
                        msg.citations &&
                        msg.citations.length > 0 && (
                          <div className="pt-2 border-t border-zinc-800/90">
                            <button
                              type="button"
                              onClick={() => toggleCitations(msg.id)}
                              className="w-full flex items-center justify-between text-xs font-mono text-[#EF4444] hover:text-red-300 py-1"
                            >
                              <span className="flex items-center gap-1.5">
                                <Database className="w-3.5 h-3.5" />
                                {msg.citations.length} Citações Vetoriais RAG
                                Recuperadas
                              </span>
                              {citationsOpen ? (
                                <ChevronUp className="w-4 h-4" />
                              ) : (
                                <ChevronDown className="w-4 h-4" />
                              )}
                            </button>

                            {citationsOpen && (
                              <div className="mt-2 space-y-2">
                                {msg.citations.map((cit) => (
                                  <div
                                    key={cit.chunk_id}
                                    className="p-2.5 rounded-xl bg-[#111114] border border-zinc-800 text-[11px] space-y-1"
                                  >
                                    <div className="flex items-center justify-between font-mono text-zinc-300">
                                      <span className="flex items-center gap-1 text-white font-semibold">
                                        <FileText className="w-3 h-3 text-[#EF4444]" />
                                        {cit.filename} (Chunk #{cit.chunk_index})
                                      </span>
                                      <span className="text-emerald-400">
                                        Score:{' '}
                                        {(cit.similarity_score * 100).toFixed(1)}
                                        %
                                      </span>
                                    </div>
                                    <p className="text-[#A1A1AA] line-clamp-3">
                                      {cit.excerpt}
                                    </p>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                    </div>
                  </div>
                );
              })
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Bar */}
          <form
            onSubmit={(e) => void handleSendMessage(e)}
            className="p-4 border-t border-zinc-800 bg-[#09090B]/60 rounded-b-2xl flex items-center gap-3"
          >
            <input
              type="text"
              value={inputPrompt}
              onChange={(e) => setInputPrompt(e.target.value)}
              disabled={sending}
              placeholder={
                selectedAgent
                  ? `Enviar comando ou consulta para ${selectedAgent.name}...`
                  : 'Selecione um agente para iniciar...'
              }
              className="flex-1 px-4 py-3 rounded-xl bg-[#09090B] border border-zinc-800 text-xs sm:text-sm text-white focus:outline-none focus:border-[#DC2626]"
            />
            <button
              type="submit"
              disabled={sending || !inputPrompt.trim()}
              className="px-5 py-3 rounded-xl bg-[#DC2626] hover:bg-[#EF4444] disabled:opacity-50 text-white text-xs font-semibold flex items-center gap-2 transition-colors shadow-lg shadow-[#DC2626]/20"
            >
              <Send className="w-4 h-4" />
              {sending ? 'Gerando...' : 'Enviar'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

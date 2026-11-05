import React, { useState } from 'react';
import {
  ShieldCheck,
  Terminal,
  KeyRound,
  UserPlus,
  LogIn,
  Sparkles,
  AlertCircle,
  Cpu,
  Database,
  Layers,
  ExternalLink,
} from 'lucide-react';
import { api } from '../services/api';
import type { UserRead, UserRole } from '../types/api';

interface AuthScreenProps {
  onAuthenticated: (user: UserRead) => void;
}

export const AuthScreen: React.FC<AuthScreenProps> = ({ onAuthenticated }) => {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('admin@iseetechnologies.com.br');
  const [password, setPassword] = useState('ISeeAdmin#2026');
  const [fullName, setFullName] = useState('Júlio César — I.See Engineering');
  const [role, setRole] = useState<UserRole>('operator');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleQuickFillDemo = () => {
    setMode('login');
    setEmail('admin@iseetechnologies.com.br');
    setPassword('ISeeAdmin#2026');
    setError(null);
  };

  const handleQuickLoginDemo = async () => {
    setMode('login');
    setEmail('admin@iseetechnologies.com.br');
    setPassword('ISeeAdmin#2026');
    setError(null);
    setLoading(true);
    try {
      const res = await api.login({
        email: 'admin@iseetechnologies.com.br',
        password: 'ISeeAdmin#2026',
      });
      onAuthenticated(res.user);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro ao autenticar.');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (mode === 'login') {
        const res = await api.login({ email, password });
        onAuthenticated(res.user);
      } else {
        const res = await api.register({
          email,
          full_name: fullName,
          password,
          role,
        });
        onAuthenticated(res.user);
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Não foi possível concluir a autenticação.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#09090B] text-[#FAFAFA] flex flex-col justify-between relative overflow-hidden">
      {/* Subtle architectural grid background */}
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.04]"
        style={{
          backgroundImage:
            'linear-gradient(to right, #FAFAFA 1px, transparent 1px), linear-gradient(to bottom, #FAFAFA 1px, transparent 1px)',
          backgroundSize: '48px 48px',
        }}
      />
      <div className="pointer-events-none absolute -top-40 left-1/2 -translate-x-1/2 w-[680px] h-[280px] bg-[#DC2626]/15 blur-[130px] rounded-full" />

      {/* Top Header */}
      <header className="relative z-10 border-b border-zinc-800/80 bg-[#09090B]/90 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-[#DC2626] flex items-center justify-center shadow-lg shadow-[#DC2626]/25">
              <Terminal className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold tracking-tight text-white text-base">
                  I.See Agents
                </span>
                <span className="text-[10px] font-mono uppercase tracking-widest px-2 py-0.5 rounded bg-[#DC2626]/15 text-[#EF4444] border border-[#DC2626]/30">
                  v2.0 Enterprise
                </span>
              </div>
              <p className="text-xs text-[#A1A1AA]">
                I.See Technologies — Multi-Agent AI & RAG Platform
              </p>
            </div>
          </div>

          <div className="hidden sm:flex items-center gap-4 text-xs text-[#A1A1AA]">
            <a
              href="https://iseetechnologies.com.br/"
              target="_blank"
              rel="noreferrer"
              className="hover:text-white transition-colors flex items-center gap-1"
            >
              iseetechnologies.com.br
              <ExternalLink className="w-3 h-3" />
            </a>
            <span className="text-zinc-700">|</span>
            <span className="font-mono text-zinc-400">
              Dev: Júlio César (securityMaker488)
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Left Architectural Pitch */}
          <div className="lg:col-span-7 space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#18181C] border border-zinc-800 text-xs text-[#A1A1AA]">
              <span className="w-2 h-2 rounded-full bg-[#DC2626] animate-pulse" />
              <span>
                Arquitetura Multi-Agente Modular • Ollama Local • LoRA / QLoRA
              </span>
            </div>

            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-white leading-[1.12]">
              Orquestração de Agentes Especializados &{' '}
              <span className="text-[#EF4444]">Pipeline RAG Vetorial</span>
            </h1>

            <p className="text-[#A1A1AA] text-sm sm:text-base leading-relaxed max-w-xl">
              Plataforma corporativa desenvolvida pela{' '}
              <strong className="text-white">I.See Technologies</strong> para
              execução local de LLMs via Ollama, indexação segura de bases de
              conhecimento com busca semântica vetorial, governança RBAC e
              arquitetura preparada para{' '}
              <strong className="text-white">
                12 slots de agentes especializados
              </strong>
              .
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
              <div className="p-4 rounded-xl bg-[#111114] border border-zinc-800/90">
                <Cpu className="w-5 h-5 text-[#EF4444] mb-2" />
                <div className="text-xs font-semibold text-white">
                  Dual-Mode Runtime
                </div>
                <p className="text-[11px] text-[#A1A1AA] mt-1">
                  Conexão direta com Ollama local ou fallback determinístico
                  transparente `[DEMO]`.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-[#111114] border border-zinc-800/90">
                <Database className="w-5 h-5 text-[#EF4444] mb-2" />
                <div className="text-xs font-semibold text-white">
                  RAG & Citações Reais
                </div>
                <p className="text-[11px] text-[#A1A1AA] mt-1">
                  Ingestão de documentos/imagens com hash SHA-256, chunking e
                  similaridade cosseno.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-[#111114] border border-zinc-800/90">
                <Layers className="w-5 h-5 text-[#EF4444] mb-2" />
                <div className="text-xs font-semibold text-white">
                  12 Slots + LoRA/QLoRA
                </div>
                <p className="text-[11px] text-[#A1A1AA] mt-1">
                  Registro versionado com suporte a adaptadores PEFT 4-bit NF4 e
                  import/export JSON.
                </p>
              </div>
            </div>
          </div>

          {/* Right Auth Card */}
          <div className="lg:col-span-5">
            <div className="bg-[#111114] border border-zinc-800 rounded-2xl p-6 sm:p-8 shadow-2xl">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <ShieldCheck className="w-5 h-5 text-[#DC2626]" />
                    Controle de Acesso JWT
                  </h2>
                  <p className="text-xs text-[#A1A1AA] mt-0.5">
                    Autentique-se para acessar o console operacional
                  </p>
                </div>
              </div>

              {/* Mode Switcher */}
              <div className="grid grid-cols-2 gap-1 p-1 bg-[#09090B] rounded-lg border border-zinc-800 mb-5">
                <button
                  type="button"
                  onClick={() => {
                    setMode('login');
                    setError(null);
                  }}
                  className={`py-2 text-xs font-medium rounded-md transition-all flex items-center justify-center gap-1.5 ${
                    mode === 'login'
                      ? 'bg-[#DC2626] text-white shadow'
                      : 'text-[#A1A1AA] hover:text-white'
                  }`}
                >
                  <LogIn className="w-3.5 h-3.5" />
                  Entrar (Login)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setMode('register');
                    setError(null);
                  }}
                  className={`py-2 text-xs font-medium rounded-md transition-all flex items-center justify-center gap-1.5 ${
                    mode === 'register'
                      ? 'bg-[#DC2626] text-white shadow'
                      : 'text-[#A1A1AA] hover:text-white'
                  }`}
                >
                  <UserPlus className="w-3.5 h-3.5" />
                  Criar Conta
                </button>
              </div>

              {/* Quick Demo Credentials Box */}
              <div className="mb-5 p-3.5 rounded-xl bg-[#18181C] border border-[#DC2626]/30">
                <div className="flex items-center justify-between gap-2 mb-2">
                  <span className="text-[11px] font-mono uppercase tracking-wider text-[#EF4444] flex items-center gap-1.5 font-semibold">
                    <Sparkles className="w-3.5 h-3.5" />
                    Credenciais Padrão (Admin Demo)
                  </span>
                  <button
                    type="button"
                    onClick={handleQuickFillDemo}
                    className="text-[11px] text-zinc-300 hover:text-white underline font-mono"
                  >
                    Preencher campos
                  </button>
                </div>
                <div className="text-xs font-mono text-[#A1A1AA] space-y-0.5 mb-3">
                  <div>
                    E-mail:{' '}
                    <span className="text-white">
                      admin@iseetechnologies.com.br
                    </span>
                  </div>
                  <div>
                    Senha: <span className="text-white">ISeeAdmin#2026</span>
                  </div>
                </div>
                <button
                  type="button"
                  disabled={loading}
                  onClick={handleQuickLoginDemo}
                  className="w-full py-2 px-3 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-white text-xs font-medium transition-colors flex items-center justify-center gap-2 border border-zinc-700"
                >
                  <KeyRound className="w-3.5 h-3.5 text-[#EF4444]" />
                  Acesso Rápido com Credencial Admin Padrão
                </button>
              </div>

              {error && (
                <div className="mb-4 p-3 rounded-lg bg-red-950/50 border border-red-800 text-red-200 text-xs flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 text-[#EF4444] shrink-0 mt-0.5" />
                  <span>{error}</span>
                </div>
              )}

              <form onSubmit={handleSubmit} className="space-y-4">
                {mode === 'register' && (
                  <div>
                    <label className="block text-xs font-medium text-zinc-300 mb-1">
                      Nome Completo
                    </label>
                    <input
                      type="text"
                      required
                      minLength={2}
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      className="w-full px-3.5 py-2.5 rounded-lg bg-[#09090B] border border-zinc-800 text-sm text-white focus:outline-none focus:border-[#DC2626]"
                      placeholder="Seu nome completo"
                    />
                  </div>
                )}

                <div>
                  <label className="block text-xs font-medium text-zinc-300 mb-1">
                    E-mail Corporativo
                  </label>
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full px-3.5 py-2.5 rounded-lg bg-[#09090B] border border-zinc-800 text-sm text-white font-mono focus:outline-none focus:border-[#DC2626]"
                    placeholder="usuario@iseetechnologies.com.br"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-zinc-300 mb-1">
                    Senha
                  </label>
                  <input
                    type="password"
                    required
                    minLength={8}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full px-3.5 py-2.5 rounded-lg bg-[#09090B] border border-zinc-800 text-sm text-white font-mono focus:outline-none focus:border-[#DC2626]"
                    placeholder="Mínimo 8 caracteres"
                  />
                </div>

                {mode === 'register' && (
                  <div>
                    <label className="block text-xs font-medium text-zinc-300 mb-1">
                      Papel RBAC (Permissão)
                    </label>
                    <select
                      value={role}
                      onChange={(e) => setRole(e.target.value as UserRole)}
                      className="w-full px-3.5 py-2.5 rounded-lg bg-[#09090B] border border-zinc-800 text-sm text-white focus:outline-none focus:border-[#DC2626]"
                    >
                      <option value="operator">
                        Operator (Criar Agentes, KBs e Executar Chat)
                      </option>
                      <option value="admin">
                        Admin (Controle Total do Sistema e Slots)
                      </option>
                      <option value="viewer">
                        Viewer (Somente Leitura e Consulta)
                      </option>
                    </select>
                  </div>
                )}

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full py-2.5 px-4 rounded-lg bg-[#DC2626] hover:bg-[#EF4444] disabled:opacity-50 text-white font-semibold text-sm transition-colors shadow-lg shadow-[#DC2626]/20 flex items-center justify-center gap-2"
                >
                  {loading
                    ? 'Processando...'
                    : mode === 'login'
                      ? 'Autenticar no Console I.See'
                      : 'Registrar e Iniciar Sessão'}
                </button>
              </form>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-zinc-900 bg-[#09090B] py-4 px-6 text-center text-xs text-[#A1A1AA]">
        I.See Technologies © 2026 — Desenvolvido por Júlio César •{' '}
        <span className="font-mono text-zinc-400">
          securityMaker488/isee-agents
        </span>
      </footer>
    </div>
  );
};

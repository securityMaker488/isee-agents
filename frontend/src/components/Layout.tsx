import React from 'react';
import {
  LayoutDashboard,
  Bot,
  MessageSquare,
  Database,
  Sliders,
  LogOut,
  Terminal,
  Shield,
  Radio,
  ExternalLink,
} from 'lucide-react';
import type { OllamaStatusSchema, UserRead } from '../types/api';

export type ActiveTab =
  | 'dashboard'
  | 'agents'
  | 'chat'
  | 'knowledge'
  | 'models';

interface LayoutProps {
  user: UserRead;
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  onLogout: () => void;
  ollamaStatus: OllamaStatusSchema | null;
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({
  user,
  activeTab,
  onSelectTab,
  onLogout,
  ollamaStatus,
  children,
}) => {
  const navItems: Array<{
    id: ActiveTab;
    label: string;
    shortLabel: string;
    icon: React.FC<{ className?: string }>;
    badge?: string;
  }> = [
    {
      id: 'dashboard',
      label: 'Dashboard & Telemetria',
      shortLabel: 'Dashboard',
      icon: LayoutDashboard,
    },
    {
      id: 'agents',
      label: 'Catálogo & 12 Slots',
      shortLabel: 'Agentes (12 Slots)',
      icon: Bot,
      badge: '12 SLOTS',
    },
    {
      id: 'chat',
      label: 'Chat Multi-Agente',
      shortLabel: 'Chat',
      icon: MessageSquare,
    },
    {
      id: 'knowledge',
      label: 'Bases RAG & Vetores',
      shortLabel: 'RAG / KBs',
      icon: Database,
    },
    {
      id: 'models',
      label: 'Modelos LoRA & Auditoria',
      shortLabel: 'LoRA & Logs',
      icon: Sliders,
    },
  ];

  const isOllamaLive = ollamaStatus?.reachable === true;

  return (
    <div className="min-h-screen bg-[#09090B] text-[#FAFAFA] flex flex-col lg:flex-row">
      {/* Sidebar Desktop */}
      <aside className="w-full lg:w-64 bg-[#111114] border-b lg:border-b-0 lg:border-r border-zinc-800/90 flex flex-col justify-between shrink-0">
        <div>
          {/* Brand Header */}
          <div className="p-5 border-b border-zinc-800/80 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-[#DC2626] flex items-center justify-center shadow-md shadow-[#DC2626]/25">
                <Terminal className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-1.5">
                  <span className="font-bold text-sm tracking-tight text-white">
                    I.See Agents
                  </span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#DC2626]/20 text-[#EF4444] border border-[#DC2626]/30">
                    v2.0
                  </span>
                </div>
                <p className="text-[11px] text-[#A1A1AA]">I.See Technologies</p>
              </div>
            </div>
          </div>

          {/* Runtime Mode Banner */}
          <div className="px-4 pt-4">
            <div
              className={`p-3 rounded-xl border text-xs ${
                isOllamaLive
                  ? 'bg-emerald-950/30 border-emerald-800/60 text-emerald-200'
                  : 'bg-[#18181C] border-amber-700/50 text-amber-200'
              }`}
            >
              <div className="flex items-center justify-between font-mono text-[11px] font-semibold">
                <span className="flex items-center gap-1.5">
                  <Radio
                    className={`w-3.5 h-3.5 ${
                      isOllamaLive ? 'text-emerald-400' : 'text-amber-400'
                    }`}
                  />
                  {isOllamaLive ? 'OLLAMA LIVE' : 'MODO DEMO ATIVO'}
                </span>
                <span className="text-[10px] opacity-80">
                  {isOllamaLive ? 'REAL LLM' : 'FALLBACK'}
                </span>
              </div>
              <p className="text-[11px] text-[#A1A1AA] mt-1 leading-snug">
                {isOllamaLive
                  ? `Conectado a ${ollamaStatus?.base_url} (${ollamaStatus?.installed_models.length || 0} modelos)`
                  : 'Ollama offline. Respostas demonstrativas sinalizadas como [DEMO].'}
              </p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="p-3 space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onSelectTab(item.id)}
                  className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all ${
                    active
                      ? 'bg-[#DC2626] text-white shadow-md shadow-[#DC2626]/20'
                      : 'text-[#A1A1AA] hover:text-white hover:bg-[#18181C]'
                  }`}
                >
                  <span className="flex items-center gap-2.5">
                    <Icon className="w-4 h-4 shrink-0" />
                    <span>{item.label}</span>
                  </span>
                  {item.badge && (
                    <span
                      className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                        active
                          ? 'bg-black/25 text-white'
                          : 'bg-zinc-800 text-zinc-300'
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* User RBAC Profile & Logout */}
        <div className="p-4 border-t border-zinc-800/80 space-y-3 bg-[#09090B]/40">
          <div className="flex items-center justify-between">
            <div className="min-w-0">
              <div className="text-xs font-semibold text-white truncate">
                {user.full_name}
              </div>
              <div className="text-[11px] text-[#A1A1AA] font-mono truncate">
                {user.email}
              </div>
            </div>
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-[#DC2626]/15 text-[#EF4444] border border-[#DC2626]/30 shrink-0">
              <Shield className="w-3 h-3" />
              {user.role}
            </span>
          </div>

          <button
            type="button"
            onClick={onLogout}
            className="w-full py-2 px-3 rounded-lg bg-[#18181C] hover:bg-zinc-800 text-xs text-zinc-300 hover:text-white border border-zinc-800 transition-colors flex items-center justify-center gap-2"
          >
            <LogOut className="w-3.5 h-3.5 text-[#EF4444]" />
            Encerrar Sessão
          </button>

          <div className="pt-1 flex items-center justify-between text-[10px] text-zinc-500 font-mono">
            <span>Dev: Júlio César</span>
            <a
              href="https://iseetechnologies.com.br/"
              target="_blank"
              rel="noreferrer"
              className="hover:text-zinc-300 flex items-center gap-0.5"
            >
              iseetechnologies.com.br
              <ExternalLink className="w-2.5 h-2.5" />
            </a>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <main className="flex-1 p-5 sm:p-7 max-w-[1440px] w-full mx-auto">
          {children}
        </main>
      </div>
    </div>
  );
};

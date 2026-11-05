import React, { useEffect, useState } from 'react';
import { AuthScreen } from './components/AuthScreen';
import { Layout, type ActiveTab } from './components/Layout';
import { DashboardView } from './views/DashboardView';
import { AgentsView } from './views/AgentsView';
import { ChatView } from './views/ChatView';
import { KnowledgeView } from './views/KnowledgeView';
import { ModelsAuditView } from './views/ModelsAuditView';
import { api, getStoredToken } from './services/api';
import type { OllamaStatusSchema, UserRead } from './types/api';

export const App: React.FC = () => {
  const [user, setUser] = useState<UserRead | null>(null);
  const [initializing, setInitializing] = useState(true);
  const [activeTab, setActiveTab] = useState<ActiveTab>('dashboard');
  const [selectedChatAgentId, setSelectedChatAgentId] = useState<string | null>(
    null
  );
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatusSchema | null>(
    null
  );

  useEffect(() => {
    const verifySession = async () => {
      const token = getStoredToken();
      if (!token) {
        setInitializing(false);
        return;
      }
      try {
        const currentUser = await api.getMe();
        setUser(currentUser);
        const status = await api.getOllamaStatus();
        setOllamaStatus(status);
      } catch {
        api.logout();
        setUser(null);
      } finally {
        setInitializing(false);
      }
    };

    void verifySession();
  }, []);

  const handleAuthenticated = async (authenticatedUser: UserRead) => {
    setUser(authenticatedUser);
    try {
      const status = await api.getOllamaStatus();
      setOllamaStatus(status);
    } catch {
      // ignore if ollama status fails initially
    }
  };

  const handleLogout = () => {
    api.logout();
    setUser(null);
    setActiveTab('dashboard');
  };

  const handleOpenChatWithAgent = (agentId: string) => {
    setSelectedChatAgentId(agentId);
    setActiveTab('chat');
  };

  if (initializing) {
    return (
      <div className="min-h-screen bg-[#09090B] text-[#FAFAFA] flex items-center justify-center text-sm font-mono">
        Inicializando I.See Agents Console...
      </div>
    );
  }

  if (!user) {
    return <AuthScreen onAuthenticated={handleAuthenticated} />;
  }

  return (
    <Layout
      user={user}
      activeTab={activeTab}
      onSelectTab={setActiveTab}
      onLogout={handleLogout}
      ollamaStatus={ollamaStatus}
    >
      {activeTab === 'dashboard' && (
        <DashboardView
          onNavigate={setActiveTab}
          onOllamaStatusUpdated={setOllamaStatus}
        />
      )}
      {activeTab === 'agents' && (
        <AgentsView onOpenChatWithAgent={handleOpenChatWithAgent} />
      )}
      {activeTab === 'chat' && (
        <ChatView initialAgentId={selectedChatAgentId} />
      )}
      {activeTab === 'knowledge' && <KnowledgeView />}
      {activeTab === 'models' && <ModelsAuditView />}
    </Layout>
  );
};

export default App;

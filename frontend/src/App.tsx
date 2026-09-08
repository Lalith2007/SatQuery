import React, { useState } from 'react';
import { Navbar, NavigationTab } from './components/layout/Navbar';
import { Footer } from './components/layout/Footer';
import { LandingPage } from './pages/LandingPage';
import { WorkspacePage } from './pages/WorkspacePage';
import { EvaluationPage } from './pages/EvaluationPage';
import { ReportsPage } from './pages/ReportsPage';
import { SystemPage } from './pages/SystemPage';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavigationTab>('workspace');

  return (
    <div className="min-h-screen flex flex-col bg-background text-text-primary">
      {/* Top Navigation */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main View Area */}
      <main className="flex-1">
        {activeTab === 'landing' && (
          <LandingPage
            onOpenWorkspace={() => setActiveTab('workspace')}
            onExploreDemo={() => setActiveTab('workspace')}
          />
        )}
        {activeTab === 'workspace' && <WorkspacePage />}
        {activeTab === 'evaluation' && <EvaluationPage />}
        {activeTab === 'reports' && <ReportsPage />}
        {activeTab === 'system' && <SystemPage />}
      </main>

      {/* Footer */}
      <Footer />
    </div>
  );
};

export default App;

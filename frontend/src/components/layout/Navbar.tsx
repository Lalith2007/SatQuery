import React from 'react';
import {
  Satellite,
  Layers,
  BarChart3,
  FileText,
  Cpu,
  Activity,
  Globe2,
} from 'lucide-react';
import { useAnalysis } from '../../context/AnalysisContext';
import { Badge } from '../common/Badge';

export type NavigationTab = 'workspace' | 'evaluation' | 'reports' | 'system' | 'landing';

interface NavbarProps {
  activeTab: NavigationTab;
  setActiveTab: (tab: NavigationTab) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab }) => {
  const { systemHealth } = useAnalysis();

  const navItems: { id: NavigationTab; label: string; icon: React.ReactNode }[] = [
    { id: 'workspace', label: 'Analysis Workspace', icon: <Layers className="w-4 h-4" /> },
    { id: 'evaluation', label: 'Evaluation', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'reports', label: 'Reports', icon: <FileText className="w-4 h-4" /> },
    { id: 'system', label: 'Specialist Registry', icon: <Cpu className="w-4 h-4" /> },
  ];

  return (
    <header className="sticky top-0 z-40 w-full border-b border-border-subtle bg-background-secondary/95 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div
          onClick={() => setActiveTab('landing')}
          className="flex items-center gap-3 cursor-pointer group"
        >
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 p-0.5 shadow-glow-cyan flex items-center justify-center transition group-hover:scale-105">
            <div className="w-full h-full bg-background-secondary rounded-[10px] flex items-center justify-center">
              <Satellite className="w-5 h-5 text-cyan-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold tracking-wider text-base text-text-primary uppercase">
                SATQUERY <span className="text-cyan-400">AI</span>
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950/80 border border-cyan-800/60 text-cyan-400 font-mono">
                v0.1.0
              </span>
            </div>
            <p className="text-[11px] text-text-muted hidden sm:block">
              Multimodal Remote Sensing Intelligence
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 bg-background-surface/70 p-1 rounded-xl border border-border-subtle">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 shadow-sm'
                    : 'text-text-secondary hover:text-text-primary hover:bg-background-elevated/50'
                }`}
              >
                {item.icon}
                <span className="hidden md:inline">{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* System Health Badge */}
        <div className="flex items-center gap-3">
          {systemHealth ? (
            <Badge
              variant={systemHealth.status === 'healthy' ? 'emerald' : 'amber'}
              size="sm"
              dot
              className="cursor-pointer"
            >
              <Activity className="w-3 h-3" />
              <span className="hidden sm:inline">
                {systemHealth.status === 'healthy' ? 'System Operational' : 'Degraded Mode'}
              </span>
              <span className="font-mono text-[10px] text-text-muted">
                ({systemHealth.registered_tools_count} tools)
              </span>
            </Badge>
          ) : (
            <Badge variant="neutral" size="sm" dot>
              Connecting...
            </Badge>
          )}

          <button
            onClick={() => setActiveTab('landing')}
            className={`p-2 rounded-lg text-xs border transition ${
              activeTab === 'landing'
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                : 'text-text-muted hover:text-text-primary border-border-subtle hover:bg-background-surface'
            }`}
            title="Overview / Landing"
          >
            <Globe2 className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
};

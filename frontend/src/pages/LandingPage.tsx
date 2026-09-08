import React from 'react';
import {
  Satellite,
  ArrowRight,
  Eye,
  Crosshair,
  GitCompare,
  Layers,
  FileText,
  BarChart3,
  ShieldCheck,
  Cpu,
  Sparkles,
} from 'lucide-react';
import { Badge } from '../components/common/Badge';

interface LandingPageProps {
  onOpenWorkspace: () => void;
  onExploreDemo: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({
  onOpenWorkspace,
  onExploreDemo,
}) => {
  const capabilities = [
    { title: 'VQA', desc: 'Remote Sensing Visual QA', icon: <Eye className="w-4 h-4 text-cyan-400" /> },
    { title: 'Grounding', desc: 'Spatial Bounding-Box Localization', icon: <Crosshair className="w-4 h-4 text-emerald-400" /> },
    { title: 'Change Intelligence', desc: 'Bi-Temporal Surface Difference', icon: <GitCompare className="w-4 h-4 text-purple-400" /> },
    { title: 'Cross-Modal Fusion', desc: 'Optical + SAR Radar Integration', icon: <Layers className="w-4 h-4 text-blue-400" /> },
    { title: 'Evidence Layer', desc: '3-Panel Storyboards & Heatmaps', icon: <Sparkles className="w-4 h-4 text-amber-400" /> },
    { title: 'Reports & Audits', desc: 'HTML, Markdown, and JSON Dossiers', icon: <FileText className="w-4 h-4 text-rose-400" /> },
  ];

  return (
    <div className="relative overflow-hidden pt-8 pb-16">
      {/* Background Radial Glow & Radar Lines */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-7xl h-96 bg-radial-gradient pointer-events-none opacity-80" />
      <div className="absolute inset-0 bg-grid-pattern opacity-20 pointer-events-none" />

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 space-y-16">
        {/* Hero Section */}
        <div className="text-center space-y-6 max-w-4xl mx-auto pt-6">
          <div className="flex justify-center">
            <Badge variant="cyan" size="md" dot>
              Agentic Multimodal Remote Sensing
            </Badge>
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-text-primary leading-tight">
            SATQUERY <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">AI</span>
          </h1>

          <p className="text-lg sm:text-xl font-medium text-text-secondary">
            An Interactive Vision-Language Assistant for Multimodal Remote Sensing
          </p>

          <p className="text-sm sm:text-base text-text-muted max-w-2xl mx-auto leading-relaxed">
            Ask questions about high-resolution satellite imagery. Detect complex changes across time. Receive verifiable, evidence-grounded answers without hallucination.
          </p>

          {/* Action CTAs */}
          <div className="flex flex-wrap items-center justify-center gap-4 pt-4">
            <button
              onClick={onOpenWorkspace}
              className="flex items-center gap-2.5 px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white text-sm font-semibold shadow-glow-cyan transition-all active:scale-95"
            >
              <span>Open Analysis Workspace</span>
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              onClick={onExploreDemo}
              className="flex items-center gap-2 px-6 py-3 rounded-xl glass-card hover:bg-background-elevated text-text-primary text-sm font-semibold border border-border-strong transition-all"
            >
              <span>Explore Presets & Demos</span>
            </button>
          </div>
        </div>

        {/* Capability Chips Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {capabilities.map((c, i) => (
            <div
              key={i}
              className="glass-card rounded-xl p-3.5 border border-border-subtle hover:border-cyan-500/50 transition flex flex-col items-center text-center space-y-2"
            >
              <div className="w-8 h-8 rounded-lg bg-background-elevated flex items-center justify-center">
                {c.icon}
              </div>
              <div>
                <h4 className="text-xs font-bold text-text-primary">{c.title}</h4>
                <p className="text-[10px] text-text-muted mt-0.5">{c.desc}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Architecture Highlights */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4">
          <div className="glass-panel rounded-xl p-5 border border-border-strong space-y-2">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-bold text-text-primary">Agent & Orchestration Engine</h3>
            </div>
            <p className="text-xs text-text-muted leading-relaxed">
              Autonomous query intent resolution, dynamic specialist routing, multi-step execution graphs, and end-to-end operational observability.
            </p>
          </div>

          <div className="glass-panel rounded-xl p-5 border border-border-strong space-y-2">
            <div className="flex items-center gap-2">
              <Eye className="w-4 h-4 text-emerald-400" />
              <h3 className="text-sm font-bold text-text-primary">Multimodal Neural Specialists</h3>
            </div>
            <p className="text-xs text-text-muted leading-relaxed">
              Fine-tuned PaliGemma vision-language models for optical VQA and grounding, paired with real TinyCD Siamese neural change detection.
            </p>
          </div>

          <div className="glass-panel rounded-xl p-5 border border-border-strong space-y-2">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-purple-400" />
              <h3 className="text-sm font-bold text-text-primary">Evidence Verification & Dossiers</h3>
            </div>
            <p className="text-xs text-text-muted leading-relaxed">
              Multi-panel visual storyboards, calibrated confidence, structured Markdown/HTML dossiers, and standardized scientific evaluation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

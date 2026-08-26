import React from 'react';
import { Layers, UploadCloud, Sparkles, Image as ImageIcon, AlertCircle } from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';
import { ImageUploader } from '../components/analysis/ImageUploader';
import { PresetSelector } from '../components/analysis/PresetSelector';
import { QueryComposer } from '../components/analysis/QueryComposer';
import { AnswerPanel } from '../components/analysis/AnswerPanel';
import { DecisionCard } from '../components/analysis/DecisionCard';
import { WorkflowVisualizer } from '../components/analysis/WorkflowVisualizer';
import { EvidenceViewer } from '../components/evidence/EvidenceViewer';
import { EvidenceGallery } from '../components/evidence/EvidenceGallery';
import { GroundingList } from '../components/evidence/GroundingList';
import { ExecutionTrace } from '../components/trace/ExecutionTrace';
import { LoadingPipeline } from '../components/common/LoadingPipeline';

export const WorkspacePage: React.FC = () => {
  const { inputMode, setInputMode, isLoading, loadingStage, pipelineStages, error, result } =
    useAnalysis();

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Loading Overlay */}
      {isLoading && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in">
          <div className="w-full max-w-md">
            <LoadingPipeline stageText={loadingStage} stages={pipelineStages} />
          </div>
        </div>
      )}

      {/* Main 2-Column Split: Input Panel (Left) vs Evidence Workspace (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Input, Upload, Presets & Query (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Mode Selector Tabs */}
          <div className="glass-panel rounded-xl p-1.5 flex items-center gap-1 border border-border-subtle">
            <button
              onClick={() => setInputMode('preset')}
              className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-xs font-semibold transition ${
                inputMode === 'preset'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-text-muted hover:text-text-primary hover:bg-background-elevated/40'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Curated Presets</span>
            </button>
            <button
              onClick={() => setInputMode('upload')}
              className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-xs font-semibold transition ${
                inputMode === 'upload'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-text-muted hover:text-text-primary hover:bg-background-elevated/40'
              }`}
            >
              <UploadCloud className="w-3.5 h-3.5" />
              <span>Upload Rasters</span>
            </button>
          </div>

          {/* Preset / Upload Component */}
          {inputMode === 'preset' ? <PresetSelector /> : <ImageUploader />}

          {/* Query Composer */}
          <QueryComposer />

          {/* Error Notice if any */}
          {error && (
            <div className="p-3.5 rounded-xl bg-rose-950/50 border border-rose-800/80 text-xs text-rose-300 flex items-start gap-2 animate-fade-in">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Right Column: Visual Evidence & Interactive Canvas (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <EvidenceViewer />
          <EvidenceGallery />
          <GroundingList />
        </div>
      </div>

      {/* Full-Width Bottom Section: Answers, Routing Decisions & Execution Traces */}
      {result && (
        <div className="space-y-4 pt-2 border-t border-border-subtle/80 animate-slide-up">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            <div className="lg:col-span-7">
              <AnswerPanel />
            </div>
            <div className="lg:col-span-5">
              <DecisionCard />
            </div>
          </div>

          <WorkflowVisualizer />
          <ExecutionTrace />
        </div>
      )}
    </div>
  );
};

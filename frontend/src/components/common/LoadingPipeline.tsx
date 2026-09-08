import React from 'react';
import { Loader2, CheckCircle2, AlertCircle, Circle } from 'lucide-react';

interface LoadingPipelineProps {
  stageText: string;
  stages: { stage: string; status: 'pending' | 'active' | 'completed' | 'failed' }[];
}

export const LoadingPipeline: React.FC<LoadingPipelineProps> = ({ stageText, stages }) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 glass-panel rounded-xl border border-cyan-800/40 shadow-glow-cyan">
      <div className="relative mb-6 flex items-center justify-center">
        <div className="absolute w-20 h-20 rounded-full bg-cyan-500/10 animate-ping" />
        <div className="w-16 h-16 rounded-full border-2 border-cyan-500/20 border-t-cyan-400 animate-spin flex items-center justify-center">
          <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
        </div>
      </div>

      <h4 className="text-base font-semibold text-text-primary text-center mb-1">
        Executing Agentic Orchestration
      </h4>
      <p className="text-xs text-accent-cyan font-mono text-center mb-6">{stageText}</p>

      {/* Pipeline checklist */}
      <div className="w-full max-w-md space-y-2.5 bg-background/80 p-4 rounded-lg border border-border-subtle">
        {stages.map((st, idx) => (
          <div key={idx} className="flex items-center gap-3 text-xs">
            {st.status === 'completed' ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            ) : st.status === 'active' ? (
              <Loader2 className="w-4 h-4 text-cyan-400 animate-spin shrink-0" />
            ) : st.status === 'failed' ? (
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            ) : (
              <Circle className="w-4 h-4 text-text-muted shrink-0" />
            )}
            <span
              className={
                st.status === 'completed'
                  ? 'text-text-secondary line-through'
                  : st.status === 'active'
                  ? 'text-cyan-300 font-medium'
                  : 'text-text-muted'
              }
            >
              {st.stage}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

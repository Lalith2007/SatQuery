import React from 'react';
import { Sparkles, CheckCircle, Cpu, ShieldCheck, HelpCircle, Layers } from 'lucide-react';
import { useAnalysis } from '../../context/AnalysisContext';
import { Badge } from '../common/Badge';

export const AnswerPanel: React.FC = () => {
  const { result } = useAnalysis();

  if (!result) return null;

  const conf = result.confidence !== null && result.confidence !== undefined ? result.confidence : null;
  const confPercent = conf !== null ? Math.round(conf * 100) : null;

  // Determine confidence variant
  let confVariant: 'emerald' | 'amber' | 'rose' | 'neutral' = 'neutral';
  let confLabel = 'CALIBRATED';
  if (confPercent !== null) {
    if (confPercent >= 80) {
      confVariant = 'emerald';
      confLabel = `HIGH CONFIDENCE (${confPercent}%)`;
    } else if (confPercent >= 60) {
      confVariant = 'amber';
      confLabel = `MODERATE CONFIDENCE (${confPercent}%)`;
    } else {
      confVariant = 'rose';
      confLabel = `LOW CONFIDENCE (${confPercent}%)`;
    }
  }

  // Determine model execution badge
  const specialist = result.agent_decision?.selected_specialist || result.selected_tools[0] || '';
  const isChangeModel = specialist.includes('temporal') || specialist.includes('change');
  
  return (
    <div className="glass-panel rounded-xl p-5 border border-border-strong space-y-4 shadow-xl animate-fade-in">
      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border-subtle pb-3">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md bg-cyan-500/10 flex items-center justify-center text-cyan-400">
            <Sparkles className="w-3.5 h-3.5" />
          </div>
          <span className="text-xs font-bold uppercase tracking-wider text-text-primary">
            Agent Intelligence Response
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {confPercent !== null && (
            <Badge variant={confVariant} size="sm" dot>
              {confLabel}
            </Badge>
          )}

          {/* Real Neural vs Synthesizer Badge */}
          {isChangeModel ? (
            <Badge variant="purple" size="sm">
              <Cpu className="w-3 h-3" />
              Neural Model: TinyCD (MPS/CUDA)
            </Badge>
          ) : (
            <Badge variant="blue" size="sm">
              <Cpu className="w-3 h-3" />
              PaliGemma RS Specialist
            </Badge>
          )}

          <Badge variant="neutral" size="sm">
            <Layers className="w-3 h-3" />
            {result.resolved_task}
          </Badge>
        </div>
      </div>

      {/* Main Answer Content */}
      <div className="p-4 rounded-lg bg-background/90 border border-border-subtle/80">
        <p className="text-sm sm:text-base leading-relaxed text-text-primary font-medium">
          {result.answer}
        </p>
      </div>

      {/* Footer Info & Verification */}
      <div className="flex flex-wrap items-center justify-between text-xs text-text-muted pt-1">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          <span>
            Resolved via <strong className="text-text-secondary">{result.agent_decision?.task_display_name || result.resolved_task}</strong>
          </span>
        </div>

        <span className="font-mono text-[11px] text-text-muted">
          Request ID: {result.request_id.slice(0, 8)}...
        </span>
      </div>
    </div>
  );
};

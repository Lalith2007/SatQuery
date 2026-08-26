import React from 'react';
import { Compass, CheckCircle2, ArrowRight, ShieldCheck } from 'lucide-react';
import { useAnalysis } from '../../context/AnalysisContext';
import { Badge } from '../common/Badge';

export const DecisionCard: React.FC = () => {
  const { result } = useAnalysis();

  if (!result || !result.agent_decision) return null;

  const decision = result.agent_decision;

  return (
    <div className="glass-card rounded-xl p-4 border border-border-subtle space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Compass className="w-4 h-4 text-cyan-400" />
          <h4 className="text-xs font-semibold text-text-primary uppercase tracking-wider">
            Agent Routing Decision
          </h4>
        </div>
        <Badge variant="cyan" size="sm">
          {decision.selected_specialist}
        </Badge>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
        <div className="bg-background-surface/80 p-2 rounded-lg border border-border-subtle">
          <span className="text-[10px] text-text-muted uppercase block">Task Intent</span>
          <span className="font-semibold text-text-primary truncate block">{decision.task_display_name}</span>
        </div>
        <div className="bg-background-surface/80 p-2 rounded-lg border border-border-subtle">
          <span className="text-[10px] text-text-muted uppercase block">Rasters Processed</span>
          <span className="font-semibold text-text-primary">{decision.image_count} {decision.image_count === 1 ? 'file' : 'files'}</span>
        </div>
        <div className="bg-background-surface/80 p-2 rounded-lg border border-border-subtle">
          <span className="text-[10px] text-text-muted uppercase block">Modalities</span>
          <span className="font-semibold text-cyan-300 capitalize">{decision.detected_modalities.join(', ')}</span>
        </div>
        <div className="bg-background-surface/80 p-2 rounded-lg border border-border-subtle">
          <span className="text-[10px] text-text-muted uppercase block">Execution Status</span>
          <span className="font-semibold text-emerald-400 capitalize">{result.status}</span>
        </div>
      </div>

      {decision.why_this_tool && (
        <div className="text-xs text-text-secondary bg-background/60 p-2.5 rounded-lg border border-border-subtle/80 flex items-start gap-2">
          <ShieldCheck className="w-3.5 h-3.5 text-cyan-400 shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <strong className="text-text-primary">Selection Rationale:</strong> {decision.why_this_tool}
          </p>
        </div>
      )}
    </div>
  );
};

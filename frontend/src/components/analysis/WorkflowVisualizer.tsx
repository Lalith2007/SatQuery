import React from 'react';
import { GitBranch, ArrowRight, CheckCircle, Clock } from 'lucide-react';
import { useAnalysis } from '../../context/AnalysisContext';
import { Badge } from '../common/Badge';

export const WorkflowVisualizer: React.FC = () => {
  const { result } = useAnalysis();

  if (!result || !result.task_plan || !result.task_plan.is_multi_step) return null;

  const plan = result.task_plan;

  return (
    <div className="glass-panel rounded-xl p-4 border border-cyan-900/40 space-y-3 shadow-glow-cyan/20">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-cyan-400" />
          <h4 className="text-xs font-semibold text-text-primary uppercase tracking-wider">
            Composite Multi-Specialist Workflow
          </h4>
        </div>
        <Badge variant="purple" size="sm">
          Sequential Pipeline
        </Badge>
      </div>

      <p className="text-xs text-text-muted">{plan.goal}</p>

      {/* Pipeline Steps Graph */}
      <div className="flex flex-col md:flex-row items-center gap-3 pt-2">
        {plan.steps.map((step, idx) => (
          <React.Fragment key={step.step_id || idx}>
            <div className="flex-1 w-full bg-background-surface p-3 rounded-lg border border-border-strong relative">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-800/60">
                  Step {step.step_index + 1}
                </span>
                <span className="flex items-center gap-1 text-[10px] text-emerald-400">
                  <CheckCircle className="w-3 h-3" />
                  {step.status}
                </span>
              </div>

              <h5 className="text-xs font-semibold text-text-primary">
                {step.tool_name === 'bitemporal_change_specialist'
                  ? 'Change Intelligence (TinyCD)'
                  : step.tool_name === 'single_image_rs_specialist'
                  ? 'Scene Intelligence (PaliGemma)'
                  : step.tool_name}
              </h5>
              <p className="text-[11px] text-text-muted mt-1 leading-snug">{step.purpose}</p>
            </div>

            {idx < plan.steps.length - 1 && (
              <div className="text-cyan-400 hidden md:block">
                <ArrowRight className="w-4 h-4" />
              </div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

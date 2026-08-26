import React, { useState } from 'react';
import { Activity, Clock, CheckCircle2, ChevronDown, ChevronUp, Terminal } from 'lucide-react';
import { useAnalysis } from '../../context/AnalysisContext';
import { Badge } from '../common/Badge';

export const ExecutionTrace: React.FC = () => {
  const { result } = useAnalysis();
  const [isExpanded, setIsExpanded] = useState<boolean>(false);
  const [selectedEntry, setSelectedEntry] = useState<number | null>(null);

  if (!result || !result.execution_trace || result.execution_trace.length === 0) {
    return null;
  }

  const totalDuration = result.execution_trace.reduce(
    (acc, t) => acc + (t.duration_ms || 0),
    0
  );

  return (
    <div className="glass-panel rounded-xl p-4 border border-border-subtle space-y-3">
      {/* Header */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between cursor-pointer group"
      >
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <h4 className="text-xs font-bold text-text-primary uppercase tracking-wider">
            Operational Execution Trace ({result.execution_trace.length} stages)
          </h4>
          <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/40">
            {totalDuration.toFixed(1)} ms
          </span>
        </div>

        <div className="flex items-center gap-2 text-xs text-text-muted group-hover:text-text-primary">
          <span>{isExpanded ? 'Collapse' : 'Expand Details'}</span>
          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </div>

      {/* Timeline Steps Preview (horizontal pills when collapsed) */}
      {!isExpanded && (
        <div className="flex flex-wrap gap-1.5 pt-1">
          {result.execution_trace.map((entry, idx) => (
            <div
              key={idx}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-background text-[10px] font-mono border border-border-subtle text-text-secondary"
            >
              <CheckCircle2 className="w-3 h-3 text-emerald-400" />
              <span>{entry.stage}</span>
              {entry.duration_ms !== null && entry.duration_ms !== undefined && (
                <span className="text-text-muted">({entry.duration_ms.toFixed(1)}ms)</span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Detailed Timeline Table (when expanded) */}
      {isExpanded && (
        <div className="space-y-2 pt-2 animate-fade-in">
          <div className="space-y-1.5 max-h-80 overflow-y-auto pr-1">
            {result.execution_trace.map((entry, idx) => (
              <div
                key={idx}
                onClick={() => setSelectedEntry(selectedEntry === idx ? null : idx)}
                className="cursor-pointer bg-background/80 hover:bg-background rounded-lg p-2.5 border border-border-subtle transition"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    <span className="text-xs font-semibold text-text-primary font-mono">
                      {entry.stage}
                    </span>
                    <span className="text-[11px] text-text-muted">by {entry.component}</span>
                  </div>

                  <div className="flex items-center gap-2">
                    {entry.duration_ms !== null && entry.duration_ms !== undefined && (
                      <span className="text-[11px] text-cyan-400 font-mono">
                        {entry.duration_ms.toFixed(2)} ms
                      </span>
                    )}
                    <Badge variant="neutral" size="sm">
                      {entry.status}
                    </Badge>
                  </div>
                </div>

                {/* Expanded Stage Metadata */}
                {selectedEntry === idx && entry.details && Object.keys(entry.details).length > 0 && (
                  <div className="mt-2 p-2 rounded bg-background-surface/90 border border-border-subtle text-[11px] font-mono text-text-secondary">
                    <pre className="overflow-x-auto text-[10px]">
                      {JSON.stringify(entry.details, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

import React from 'react';
import { Satellite, ShieldCheck, Terminal, Heart } from 'lucide-react';
import { useAnalysis } from '../../context/AnalysisContext';

export const Footer: React.FC = () => {
  const { systemHealth, result } = useAnalysis();

  const latency = result?.execution_trace?.reduce((acc, t) => acc + (t.duration_ms || 0), 0);

  return (
    <footer className="w-full border-t border-border-subtle bg-background-secondary/80 py-4 px-6 text-xs text-text-muted">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5 text-text-secondary">
            <Satellite className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-semibold text-text-primary">SatQuery AI</span>
          </div>
          <span>•</span>
          <span className="flex items-center gap-1">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            Zero-Hallucination Evidence Grounded
          </span>
          {latency !== undefined && (
            <>
              <span>•</span>
              <span className="flex items-center gap-1 text-accent-amber font-mono">
                <Terminal className="w-3.5 h-3.5" />
                Execution: {latency.toFixed(1)} ms
              </span>
            </>
          )}
        </div>

        <div className="flex items-center gap-4 text-[11px]">
          <span>
            Environment:{' '}
            <strong className="text-cyan-400">{systemHealth?.environment || 'development'}</strong>
          </span>
          <span>•</span>
          <span>Engine: PyTorch 2.6 + Apple Silicon (MPS) / CUDA</span>
        </div>
      </div>
    </footer>
  );
};

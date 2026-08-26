import React from 'react';
import { Crosshair, MapPin, Hash, Sparkles } from 'lucide-react';
import { useAnalysis } from '../../context/AnalysisContext';
import { Badge } from '../common/Badge';

export const GroundingList: React.FC = () => {
  const { result } = useAnalysis();

  if (!result || !result.evidence || result.evidence.length === 0) {
    return null;
  }

  return (
    <div className="glass-panel rounded-xl p-4 border border-border-subtle space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-bold text-text-secondary uppercase tracking-wider flex items-center gap-2">
          <Crosshair className="w-3.5 h-3.5 text-emerald-400" />
          Spatial Grounding & Evidence Items ({result.evidence.length})
        </h4>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
        {result.evidence.map((ev, idx) => {
          const bbox = ev.data?.bbox;
          const conf = ev.confidence !== null && ev.confidence !== undefined ? Math.round(ev.confidence * 100) : null;

          return (
            <div
              key={ev.id || idx}
              className="bg-background-surface/80 rounded-lg p-3 border border-border-subtle flex flex-col justify-between gap-2"
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <span className="text-[10px] text-cyan-400 font-mono uppercase block">
                    {ev.type}
                  </span>
                  <h5 className="text-xs font-semibold text-text-primary">{ev.label}</h5>
                </div>
                {conf !== null && (
                  <Badge variant="emerald" size="sm">
                    {conf}% Conf
                  </Badge>
                )}
              </div>

              {bbox && (
                <div className="bg-background/90 p-2 rounded border border-border-subtle/80 font-mono text-[11px] text-text-secondary flex items-center gap-1.5">
                  <MapPin className="w-3 h-3 text-cyan-400 shrink-0" />
                  <span>
                    BBox: [{bbox.map((v) => (typeof v === 'number' ? v.toFixed(3) : v)).join(', ')}]
                  </span>
                </div>
              )}

              {ev.data?.changed_pixel_ratio !== undefined && (
                <div className="text-[11px] text-text-muted flex items-center gap-2">
                  <span>
                    Changed Area: <strong className="text-purple-400">{(ev.data.changed_pixel_ratio * 100).toFixed(1)}%</strong>
                  </span>
                  {ev.data?.num_regions !== undefined && (
                    <span>• Clusters: <strong className="text-text-primary">{ev.data.num_regions}</strong></span>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

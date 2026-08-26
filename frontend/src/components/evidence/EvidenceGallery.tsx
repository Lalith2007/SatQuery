import React from 'react';
import { Layers, Download, Check, FileImage, ExternalLink } from 'lucide-react';
import { useAnalysis } from '../../context/AnalysisContext';
import { api } from '../../services/api';
import { Badge } from '../common/Badge';

export const EvidenceGallery: React.FC = () => {
  const { result, activeArtifact, setActiveArtifact } = useAnalysis();

  if (!result || !result.artifacts || result.artifacts.length === 0) {
    return null;
  }

  const getArtifactBadge = (name: string) => {
    const n = name.toLowerCase();
    if (n.includes('composite')) return <Badge variant="purple" size="sm">3-Panel Storyboard</Badge>;
    if (n.includes('grounding') || n.includes('annotated')) return <Badge variant="emerald" size="sm">Grounding Overlay</Badge>;
    if (n.includes('heatmap')) return <Badge variant="amber" size="sm">Heatmap</Badge>;
    if (n.includes('crop')) return <Badge variant="cyan" size="sm">ROI Crop</Badge>;
    if (n.includes('mask')) return <Badge variant="neutral" size="sm">Binary Mask</Badge>;
    return <Badge variant="blue" size="sm">Artifact</Badge>;
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-bold text-text-secondary uppercase tracking-wider flex items-center gap-2">
          <Layers className="w-3.5 h-3.5 text-cyan-400" />
          Generated Artifact Gallery ({result.artifacts.length})
        </h4>
        <span className="text-[11px] text-text-muted">Click any artifact to inspect in main viewport</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {result.artifacts.map((art) => {
          const isSelected = activeArtifact?.artifact_id === art.artifact_id;
          const artUrl = api.getArtifactUrl(art.artifact_id || art.name);

          return (
            <div
              key={art.artifact_id}
              onClick={() => setActiveArtifact(art)}
              className={`group cursor-pointer rounded-xl p-2.5 border transition-all duration-200 flex flex-col justify-between ${
                isSelected
                  ? 'bg-cyan-950/40 border-cyan-500 shadow-glow-cyan'
                  : 'glass-card hover:border-cyan-500/50'
              }`}
            >
              {/* Thumbnail */}
              <div className="w-full h-24 rounded-lg bg-background overflow-hidden border border-border-subtle relative flex items-center justify-center mb-2">
                <img
                  src={artUrl}
                  alt={art.name}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                  onError={(e) => {
                    (e.target as HTMLElement).style.display = 'none';
                  }}
                />
                <div className="absolute top-1.5 right-1.5">
                  {isSelected && (
                    <span className="w-5 h-5 rounded-full bg-cyan-500 text-black flex items-center justify-center shadow">
                      <Check className="w-3 h-3 stroke-[3]" />
                    </span>
                  )}
                </div>
              </div>

              {/* Title & Badge */}
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  {getArtifactBadge(art.name)}
                  <a
                    href={artUrl}
                    download
                    onClick={(e) => e.stopPropagation()}
                    className="text-text-muted hover:text-cyan-300 p-1 rounded hover:bg-background-elevated transition"
                    title="Download"
                  >
                    <Download className="w-3 h-3" />
                  </a>
                </div>
                <p className="text-xs font-semibold text-text-primary truncate" title={art.name}>
                  {art.name}
                </p>
                {art.description && (
                  <p className="text-[10px] text-text-muted line-clamp-1">{art.description}</p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

import React from 'react';
import {
  Eye,
  Crosshair,
  GitCompare,
  Layers,
  Workflow,
  CheckCircle2,
  ArrowRight,
  Sparkles,
} from 'lucide-react';
import { PRESET_SCENARIOS, useAnalysis } from '../../context/AnalysisContext';
import { api } from '../../services/api';
import { Badge } from '../common/Badge';

export const PresetSelector: React.FC = () => {
  const { selectedPreset, selectPreset, inputMode } = useAnalysis();

  const getIcon = (name: string) => {
    switch (name) {
      case 'Eye':
        return <Eye className="w-4 h-4 text-cyan-400" />;
      case 'Crosshair':
        return <Crosshair className="w-4 h-4 text-emerald-400" />;
      case 'GitCompare':
        return <GitCompare className="w-4 h-4 text-purple-400" />;
      case 'Layers':
        return <Layers className="w-4 h-4 text-blue-400" />;
      case 'Workflow':
        return <Workflow className="w-4 h-4 text-amber-400" />;
      default:
        return <Eye className="w-4 h-4 text-cyan-400" />;
    }
  };

  return (
    <div className="space-y-2.5">
      {PRESET_SCENARIOS.map((preset) => {
        const isSelected = inputMode === 'preset' && selectedPreset?.id === preset.id;
        return (
          <div
            key={preset.id}
            onClick={() => selectPreset(preset)}
            className={`cursor-pointer rounded-xl p-3 border transition-all duration-200 ${
              isSelected
                ? 'bg-cyan-950/40 border-cyan-500/80 shadow-glow-cyan'
                : 'glass-card hover:border-cyan-500/40'
            }`}
          >
            {/* Header info */}
            <div className="flex items-start justify-between gap-2 mb-2">
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-lg bg-background-elevated flex items-center justify-center shrink-0 border border-border-subtle">
                  {getIcon(preset.iconName)}
                </div>
                <div>
                  <h4 className="text-xs font-bold text-text-primary flex items-center gap-1.5">
                    {preset.name}
                    {isSelected && <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400" />}
                  </h4>
                  <p className="text-[11px] text-text-muted mt-0.5">{preset.subtitle}</p>
                </div>
              </div>

              <Badge variant="cyan" size="sm">
                {preset.sensor}
              </Badge>
            </div>

            {/* Visual Real Satellite Thumbnail Preview Strip */}
            <div className="grid grid-cols-2 gap-2 my-2">
              {preset.images.map((img, i) => {
                const isTiff = img.name.endsWith('.tif') || img.name.endsWith('.tiff');
                const imgUrl = api.getArtifactUrl(img.name);

                return (
                  <div
                    key={i}
                    className={`relative rounded-lg overflow-hidden border border-border-subtle bg-background ${
                      preset.images.length === 1 ? 'col-span-2 h-20' : 'h-16'
                    }`}
                  >
                    {!isTiff ? (
                      <img
                        src={imgUrl}
                        alt={img.role}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                        onError={(e) => {
                          (e.target as HTMLElement).style.opacity = '0.5';
                        }}
                      />
                    ) : (
                      <div className="w-full h-full bg-gradient-to-tr from-slate-900 via-slate-800 to-cyan-950/40 flex items-center justify-center p-2 text-center">
                        <span className="text-[10px] font-mono text-cyan-300">
                          SAR Radar GeoTIFF
                        </span>
                      </div>
                    )}
                    <div className="absolute bottom-1 left-1.5 px-1.5 py-0.5 rounded bg-black/75 backdrop-blur-xs text-[9px] font-mono text-text-secondary border border-white/10">
                      {img.role}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Footer Metadata & CTA */}
            <div className="flex items-center justify-between text-[10px] pt-1.5 border-t border-border-subtle/60 text-text-muted">
              <span>Source: <strong className="text-text-secondary">{preset.source}</strong></span>
              <span className={`font-semibold flex items-center gap-1 ${isSelected ? 'text-cyan-400' : 'text-text-muted'}`}>
                {isSelected ? 'Active Preset' : 'Select Preset'} <ArrowRight className="w-3 h-3" />
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
};

import React, { useState, useEffect } from 'react';
import {
  Cpu,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Layers,
  ArrowLeftRight,
  Activity,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Eye,
  GitCompare,
  Terminal,
  ShieldCheck,
} from 'lucide-react';
import { api } from '../../services/api';
import { SystemHealth, ToolMetadata } from '../../types/api';
import { Badge } from '../common/Badge';

interface ProductToolMapping {
  displayName: string;
  category: 'production' | 'development' | 'fallback';
  modelTitle: string;
  capabilities: string[];
  modalitiesDisplay: string;
  imageInputDisplay: string;
  description: string;
}

const TOOL_DISPLAY_MAP: Record<string, ProductToolMapping> = {
  single_image_rs_specialist: {
    displayName: 'Single-Image Intelligence',
    category: 'production',
    modelTitle: 'PaliGemma 3B + SatQuery LoRA',
    capabilities: ['Visual Question Answering', 'Spatial Grounding', 'Scene Captioning'],
    modalitiesDisplay: 'Optical · Multispectral · SAR',
    imageInputDisplay: '1 raster',
    description:
      'Adapted vision-language foundation model for remote sensing, delivering multi-class VQA, bounding box object grounding, and comprehensive scene description.',
  },
  bitemporal_change_specialist: {
    displayName: 'Bi-Temporal Change Intelligence',
    category: 'production',
    modelTitle: 'TinyCD Neural Change Detector',
    capabilities: ['Change Detection', 'Change VQA', 'Spatial Localization'],
    modalitiesDisplay: 'Optical · Multispectral',
    imageInputDisplay: '2 rasters (T0 & T1)',
    description:
      'Siamese convolutional neural network trained on the LEVIR-CD benchmark for sub-meter surface change detection, multi-cluster spatial localization, and temporal difference analysis.',
  },
  optical_sar_cross_modal_mock: {
    displayName: 'Optical-SAR Intelligence',
    category: 'development',
    modelTitle: 'Cross-Modal Optical + SAR Interface',
    capabilities: ['Optical-SAR Analysis', 'Cloud Penetration', 'Radar Fusion'],
    modalitiesDisplay: 'Optical · SAR',
    imageInputDisplay: '2 rasters (Optical + GeoTIFF SAR)',
    description:
      'Cross-modal fusion pipeline ingesting co-registered optical RGB imagery and Sentinel-1 C-Band SAR amplitude rasters for cloud-penetrating structural analysis.',
  },
  single_image_vqa_mock: {
    displayName: 'Standard VQA Backend',
    category: 'development',
    modelTitle: 'Single-Image VQA Service Backend',
    capabilities: ['Visual Question Answering'],
    modalitiesDisplay: 'Optical · Multispectral · SAR',
    imageInputDisplay: '1 raster',
    description:
      'Deterministic VQA service backend used for standard benchmarking, automated testing, and offline test environments.',
  },
  single_image_caption_mock: {
    displayName: 'Scene Captioning Backend',
    category: 'development',
    modelTitle: 'Remote-Sensing Caption Generator',
    capabilities: ['Scene Captioning'],
    modalitiesDisplay: 'Optical · Multispectral · SAR',
    imageInputDisplay: '1 raster',
    description:
      'Semantic captioning service backend for automated remote-sensing overview generation and textual scene summaries.',
  },
  single_image_grounding_mock: {
    displayName: 'Spatial Grounding Backend',
    category: 'development',
    modelTitle: 'Spatial Grounding & Bounding Box Engine',
    capabilities: ['Visual Grounding', 'Bounding Box Extraction'],
    modalitiesDisplay: 'Optical · Multispectral · SAR',
    imageInputDisplay: '1 raster',
    description:
      'Feature localization backend returning normalized coordinates for runways, taxiways, water bodies, and structural assets.',
  },
  bi_temporal_change_mock: {
    displayName: 'Bi-Temporal Difference Mock',
    category: 'development',
    modelTitle: 'Pixel-Difference Change Service',
    capabilities: ['Change Detection', 'Change VQA'],
    modalitiesDisplay: 'Optical · Multispectral · SAR',
    imageInputDisplay: '2 rasters',
    description:
      'Deterministic baseline change detection backend for unit test verification and comparative evaluation.',
  },
  alternate_single_image_vqa_mock: {
    displayName: 'Alternate VQA Demonstration',
    category: 'development',
    modelTitle: 'Alternate Experimental VQA Service',
    capabilities: ['Experimental VQA'],
    modalitiesDisplay: 'Optical · Multispectral',
    imageInputDisplay: '1 raster',
    description:
      'Dynamic alternate VQA backend used to demonstrate zero-downtime runtime specialist swapping in the registry.',
  },
};

export const SystemStatusView: React.FC = () => {
  const [tools, setTools] = useState<ToolMetadata[]>([]);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [swapLoading, setSwapLoading] = useState<boolean>(false);
  const [swapMessage, setSwapMessage] = useState<string | null>(null);
  const [useAlternate, setUseAlternate] = useState<boolean>(false);
  const [expandedDetails, setExpandedDetails] = useState<Record<string, boolean>>({});

  useEffect(() => {
    loadStatus();
  }, []);

  const loadStatus = async () => {
    setIsLoading(true);
    try {
      const [toolsRes, healthRes] = await Promise.all([api.getTools(), api.getHealth()]);
      setTools(toolsRes);
      setHealth(healthRes);
    } catch (e) {
      console.error('Failed to load status:', e);
    } finally {
      setIsLoading(false);
    }
  };

  const handleToolSwap = async () => {
    setSwapLoading(true);
    setSwapMessage(null);
    try {
      const newAlternate = !useAlternate;
      const res = await api.swapTool(
        newAlternate ? 'single_image_vqa_mock' : 'alternate_single_image_vqa_mock',
        newAlternate
      );
      setUseAlternate(newAlternate);
      setSwapMessage(res.message || 'Specialist backend swapped successfully.');
      await loadStatus();
    } catch (err: any) {
      setSwapMessage(err.response?.data?.detail || err.message || 'Tool swap failed.');
    } finally {
      setSwapLoading(false);
    }
  };

  const toggleDetails = (toolName: string) => {
    setExpandedDetails((prev) => ({ ...prev, [toolName]: !prev[toolName] }));
  };

  // Group tools into Production vs Development / Mock
  const productionTools = tools.filter((t) => !t.name.includes('mock'));
  const devTools = tools.filter((t) => t.name.includes('mock'));

  const renderToolCard = (tool: ToolMetadata) => {
    const isHealthy = health?.tools_health[tool.name] ?? true;
    const mapping = TOOL_DISPLAY_MAP[tool.name] || {
      displayName: tool.name.replace(/_/g, ' ').toUpperCase(),
      category: !tool.name.includes('mock') ? 'production' : 'development',
      modelTitle: tool.metadata?.model_name || 'Specialist Backend',
      capabilities: tool.supported_tasks.map((t) => t.replace(/_/g, ' ')),
      modalitiesDisplay: tool.required_modalities.join(' · '),
      imageInputDisplay: `${tool.min_images} to ${tool.max_images} raster(s)`,
      description: tool.description,
    };

    const isExpanded = !!expandedDetails[tool.name];

    return (
      <div
        key={tool.name}
        className={`glass-card rounded-xl p-5 border transition flex flex-col justify-between space-y-4 ${
          mapping.category === 'production'
            ? 'border-cyan-500/40 shadow-glow-cyan/10 hover:border-cyan-400/70'
            : 'border-border-subtle hover:border-purple-500/40'
        }`}
      >
        <div>
          {/* Header */}
          <div className="flex items-start justify-between gap-3 mb-2">
            <div>
              <h4 className="text-sm font-bold text-text-primary flex items-center gap-2">
                {mapping.displayName}
              </h4>
              <p className="text-xs font-semibold text-cyan-400 mt-0.5">
                {mapping.modelTitle}
              </p>
            </div>

            <div className="flex flex-col items-end gap-1 shrink-0">
              {mapping.category === 'production' ? (
                <Badge variant="cyan" size="sm" dot>
                  Production
                </Badge>
              ) : (
                <Badge variant="neutral" size="sm">
                  Development / Mock
                </Badge>
              )}

              <Badge variant={isHealthy ? 'emerald' : 'rose'} size="sm">
                {isHealthy ? '● Online' : '● Offline'}
              </Badge>
            </div>
          </div>

          <p className="text-xs text-text-secondary leading-relaxed mt-2.5">
            {mapping.description}
          </p>

          {/* Capabilities & Modalities Grid */}
          <div className="grid grid-cols-2 gap-2 mt-3.5 pt-3 border-t border-border-subtle/70 text-[11px]">
            <div className="bg-background/80 p-2 rounded-lg border border-border-subtle">
              <span className="text-[10px] text-text-muted uppercase font-mono block mb-1">
                Modalities
              </span>
              <span className="font-medium text-text-primary capitalize">
                {mapping.modalitiesDisplay}
              </span>
            </div>

            <div className="bg-background/80 p-2 rounded-lg border border-border-subtle">
              <span className="text-[10px] text-text-muted uppercase font-mono block mb-1">
                Image Input
              </span>
              <span className="font-medium text-text-primary">
                {mapping.imageInputDisplay}
              </span>
            </div>
          </div>

          {/* Capabilities Badges */}
          <div className="mt-3">
            <span className="text-[10px] text-text-muted uppercase font-mono block mb-1.5">
              Capabilities
            </span>
            <div className="flex flex-wrap gap-1.5">
              {mapping.capabilities.map((cap, idx) => (
                <span
                  key={idx}
                  className="text-[10px] px-2 py-0.5 rounded bg-background text-cyan-300 font-mono border border-border-subtle/90 flex items-center gap-1"
                >
                  <span className="text-cyan-400 font-bold">•</span> {cap}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Expandable Technical Details */}
        <div className="pt-2 border-t border-border-subtle/60">
          <button
            onClick={() => toggleDetails(tool.name)}
            className="w-full flex items-center justify-between text-[11px] text-text-muted hover:text-text-primary transition py-1"
          >
            <span className="font-mono flex items-center gap-1">
              <Terminal className="w-3 h-3 text-cyan-400" /> Technical Details
            </span>
            {isExpanded ? (
              <ChevronUp className="w-3.5 h-3.5" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" />
            )}
          </button>

          {isExpanded && (
            <div className="mt-2 p-2.5 rounded-lg bg-background/90 border border-border-subtle font-mono text-[10px] space-y-1 text-text-muted">
              <div>
                Tool Identifier: <span className="text-cyan-300">{tool.name}</span>
              </div>
              <div>
                Version: <span className="text-text-secondary">v{tool.version}</span>
              </div>
              <div>
                Tasks: <span className="text-text-secondary">{tool.supported_tasks.join(', ')}</span>
              </div>
              {tool.metadata && Object.keys(tool.metadata).length > 0 && (
                <div className="pt-1 border-t border-border-subtle/40 text-[9px] text-text-muted truncate">
                  Metadata: {JSON.stringify(tool.metadata)}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* System Health Overview Card */}
      <div className="glass-panel rounded-xl p-6 border border-border-strong space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 text-cyan-400 flex items-center justify-center border border-cyan-800/40">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-text-primary flex items-center gap-2">
                Specialist Tool Registry & Runtime Status
                {health && (
                  <Badge variant={health.status === 'healthy' ? 'emerald' : 'amber'} size="sm" dot>
                    {health.status.toUpperCase()}
                  </Badge>
                )}
              </h3>
              <p className="text-xs text-text-muted mt-0.5">
                Real-time registry of multi-modal vision-language agents and remote sensing neural models.
              </p>
            </div>
          </div>

          <button
            onClick={loadStatus}
            disabled={isLoading}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-background-elevated hover:bg-background-elevated/80 text-text-secondary hover:text-text-primary text-xs font-medium border border-border-subtle transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh Registry</span>
          </button>
        </div>

        {health && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-border-subtle">
            <div className="bg-background/80 p-2.5 rounded-lg border border-border-subtle">
              <span className="text-[10px] text-text-muted uppercase font-mono block">App Version</span>
              <span className="text-xs font-semibold text-text-primary">{health.version}</span>
            </div>
            <div className="bg-background/80 p-2.5 rounded-lg border border-border-subtle">
              <span className="text-[10px] text-text-muted uppercase font-mono block">Active Tools</span>
              <span className="text-xs font-semibold text-cyan-400">{health.registered_tools_count} Registered</span>
            </div>
            <div className="bg-background/80 p-2.5 rounded-lg border border-border-subtle">
              <span className="text-[10px] text-text-muted uppercase font-mono block">Environment</span>
              <span className="text-xs font-semibold text-purple-400 capitalize">{health.environment}</span>
            </div>
            <div className="bg-background/80 p-2.5 rounded-lg border border-border-subtle">
              <span className="text-[10px] text-text-muted uppercase font-mono block">Last Ping</span>
              <span className="text-[11px] font-mono text-text-secondary truncate block">
                {new Date(health.timestamp).toLocaleTimeString()}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Dev Tool Swap Demonstration Card */}
      <div className="glass-card rounded-xl p-5 border border-purple-900/40 space-y-3 shadow-glow-blue/20">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <ArrowLeftRight className="w-4 h-4 text-purple-400" />
            <div>
              <h4 className="text-xs font-bold text-text-primary uppercase tracking-wider">
                Specialist Tool Swapping Demo (Development Mode)
              </h4>
              <p className="text-[11px] text-text-muted">
                Demonstrates dynamic runtime tool replacement in the registry without modifying agent core logic.
              </p>
            </div>
          </div>

          <button
            onClick={handleToolSwap}
            disabled={swapLoading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold transition active:scale-95 disabled:opacity-50 shrink-0"
          >
            <ArrowLeftRight className={`w-3.5 h-3.5 ${swapLoading ? 'animate-spin' : ''}`} />
            <span>{useAlternate ? 'Restore Standard Backend' : 'Swap to Alternate VQA Demonstration'}</span>
          </button>
        </div>

        {swapMessage && (
          <div className="p-2.5 rounded-lg bg-purple-950/40 border border-purple-800/60 text-xs font-mono text-purple-300">
            {swapMessage}
          </div>
        )}
      </div>

      {/* SECTION 1: PRODUCTION INTELLIGENCE */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4" /> Production Intelligence ({productionTools.length})
          </h4>
          <span className="text-[11px] text-text-muted">Verified Neural Weights & LoRA Adapters</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {productionTools.map(renderToolCard)}
        </div>
      </div>

      {/* SECTION 2: DEVELOPMENT & FALLBACK BACKENDS */}
      <div className="space-y-3 pt-2">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-bold text-text-secondary uppercase tracking-wider flex items-center gap-1.5">
            <Layers className="w-4 h-4 text-purple-400" /> Development & Fallback Backends ({devTools.length})
          </h4>
          <span className="text-[11px] text-text-muted">Specialist Mock Backends & Evaluation Harnesses</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {devTools.map(renderToolCard)}
        </div>
      </div>
    </div>
  );
};

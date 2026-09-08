import React, { useState } from 'react';
import {
  Maximize2,
  Download,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Layers,
  Image as ImageIcon,
  AlertCircle,
  FileCheck,
} from 'lucide-react';
import { useAnalysis } from '../../context/AnalysisContext';
import { api } from '../../services/api';
import { Badge } from '../common/Badge';
import { Modal } from '../common/Modal';

export const EvidenceViewer: React.FC = () => {
  const { result, activeArtifact, setActiveArtifact, selectedPreset, inputMode, uploadedFiles } =
    useAnalysis();
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [imageError, setImageError] = useState<boolean>(false);

  // Fallback image source if no analysis has run yet
  let fallbackSrc = '';
  let fallbackTitle = 'Awaiting Analysis';

  if (!result) {
    if (inputMode === 'upload' && uploadedFiles.length > 0) {
      fallbackSrc = uploadedFiles[0].previewUrl;
      fallbackTitle = uploadedFiles[0].file.name;
    } else if (selectedPreset && selectedPreset.images.length > 0) {
      fallbackSrc = api.getArtifactUrl(selectedPreset.images[0].name);
      fallbackTitle = selectedPreset.name;
    }
  }

  const currentImageSrc = activeArtifact
    ? api.getArtifactUrl(activeArtifact.artifact_id || activeArtifact.name)
    : fallbackSrc;

  const currentTitle = activeArtifact ? activeArtifact.name : fallbackTitle;

  return (
    <div className="glass-panel rounded-xl flex flex-col h-full border border-border-strong overflow-hidden shadow-2xl">
      {/* Top Controls Toolbar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-subtle bg-background-secondary/90">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-6 h-6 rounded-md bg-cyan-500/10 flex items-center justify-center text-cyan-400 shrink-0">
            <Layers className="w-3.5 h-3.5" />
          </div>
          <div className="truncate">
            <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider truncate">
              {currentTitle}
            </h3>
            {activeArtifact && (
              <p className="text-[10px] text-cyan-400 font-mono truncate">
                {activeArtifact.type || 'Visual Evidence'}
              </p>
            )}
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-1.5 shrink-0">
          {currentImageSrc && (
            <>
              <button
                onClick={() => setZoomLevel((z) => Math.min(z + 0.25, 2.5))}
                className="p-1.5 rounded-lg text-text-muted hover:text-text-primary hover:bg-background-elevated transition"
                title="Zoom In"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                onClick={() => setZoomLevel((z) => Math.max(z - 0.25, 0.5))}
                className="p-1.5 rounded-lg text-text-muted hover:text-text-primary hover:bg-background-elevated transition"
                title="Zoom Out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <button
                onClick={() => setZoomLevel(1)}
                className="p-1.5 rounded-lg text-text-muted hover:text-text-primary hover:bg-background-elevated transition"
                title="Reset Zoom"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
              <a
                href={currentImageSrc}
                download
                target="_blank"
                rel="noreferrer"
                className="p-1.5 rounded-lg text-text-muted hover:text-cyan-300 hover:bg-cyan-500/10 transition"
                title="Download Artifact"
              >
                <Download className="w-4 h-4" />
              </a>
              <button
                onClick={() => setIsModalOpen(true)}
                className="p-1.5 rounded-lg text-text-muted hover:text-cyan-300 hover:bg-cyan-500/10 transition ml-1"
                title="Fullscreen Inspection"
              >
                <Maximize2 className="w-4 h-4" />
              </button>
            </>
          )}
        </div>
      </div>

      {/* Main Viewport Canvas */}
      <div className="flex-1 bg-background relative flex items-center justify-center p-4 overflow-hidden min-h-[360px] max-h-[500px]">
        {/* Radar background grid */}
        <div className="absolute inset-0 bg-grid-pattern opacity-40 pointer-events-none" />

        {currentImageSrc && !imageError ? (
          <div className="relative z-10 max-h-full max-w-full overflow-auto flex items-center justify-center">
            <img
              src={currentImageSrc}
              alt={currentTitle}
              onError={() => setImageError(true)}
              style={{
                transform: `scale(${zoomLevel})`,
                transformOrigin: 'center center',
              }}
              className="max-h-[380px] max-w-full rounded-lg shadow-2xl object-contain border border-border-strong/80 transition-transform duration-200"
            />
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center text-center p-8 z-10">
            <div className="w-14 h-14 rounded-2xl bg-background-surface border border-border-subtle flex items-center justify-center text-text-muted mb-3">
              <ImageIcon className="w-7 h-7" />
            </div>
            <h4 className="text-sm font-semibold text-text-secondary">
              {imageError ? 'Evidence Rendering Unavailable' : 'No Visual Evidence Selected'}
            </h4>
            <p className="text-xs text-text-muted max-w-xs mt-1">
              {imageError
                ? 'The requested artifact could not be loaded.'
                : 'Execute an analysis query to generate evidence-grounded spatial storyboards.'}
            </p>
          </div>
        )}

        {/* Active Artifact Badge Floating in Corner */}
        {activeArtifact && (
          <div className="absolute bottom-3 left-3 z-20 flex items-center gap-2">
            <Badge variant="cyan" size="sm">
              <FileCheck className="w-3 h-3" />
              {activeArtifact.name}
            </Badge>
          </div>
        )}
      </div>

      {/* Fullscreen Lightbox Modal */}
      <Modal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title={currentTitle}
        subtitle={activeArtifact?.description || 'Remote Sensing Visual Evidence Inspection'}
        imageSrc={currentImageSrc}
      />
    </div>
  );
};

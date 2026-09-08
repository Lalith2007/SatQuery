import React, { useEffect } from 'react';
import { X, ZoomIn, ZoomOut, RotateCcw, Download } from 'lucide-react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  imageSrc?: string;
  children?: React.ReactNode;
}

export const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  subtitle,
  imageSrc,
  children,
}) => {
  const [zoom, setZoom] = React.useState<number>(1);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md animate-fade-in">
      <div className="relative w-full max-w-5xl max-h-[90vh] flex flex-col bg-background-surface border border-border-strong rounded-xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border-subtle bg-background-secondary/90">
          <div>
            <h3 className="text-lg font-semibold text-text-primary flex items-center gap-2">
              <span>🔍</span> {title}
            </h3>
            {subtitle && <p className="text-xs text-text-secondary mt-0.5">{subtitle}</p>}
          </div>

          <div className="flex items-center gap-2">
            {imageSrc && (
              <>
                <button
                  onClick={() => setZoom((z) => Math.min(z + 0.25, 3))}
                  className="p-1.5 rounded-lg text-text-secondary hover:text-text-primary hover:bg-background-elevated transition"
                  title="Zoom In"
                >
                  <ZoomIn className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setZoom((z) => Math.max(z - 0.25, 0.5))}
                  className="p-1.5 rounded-lg text-text-secondary hover:text-text-primary hover:bg-background-elevated transition"
                  title="Zoom Out"
                >
                  <ZoomOut className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setZoom(1)}
                  className="p-1.5 rounded-lg text-text-secondary hover:text-text-primary hover:bg-background-elevated transition"
                  title="Reset Zoom"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
                <a
                  href={imageSrc}
                  download
                  target="_blank"
                  rel="noreferrer"
                  className="p-1.5 rounded-lg text-accent-cyan hover:bg-accent-cyan/10 transition"
                  title="Download Image"
                >
                  <Download className="w-4 h-4" />
                </a>
              </>
            )}
            <button
              onClick={onClose}
              className="p-1.5 ml-2 rounded-lg text-text-secondary hover:text-text-primary hover:bg-rose-500/20 transition"
            >
              <X className="w-5 h-5 text-rose-400" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6 flex items-center justify-center bg-background min-h-[400px]">
          {imageSrc ? (
            <div className="overflow-auto max-h-[70vh] flex items-center justify-center">
              <img
                src={imageSrc}
                alt={title}
                style={{ transform: `scale(${zoom})`, transformOrigin: 'center center' }}
                className="max-h-[68vh] max-w-full rounded-lg shadow-lg object-contain transition-transform duration-150"
              />
            </div>
          ) : (
            children
          )}
        </div>
      </div>
    </div>
  );
};

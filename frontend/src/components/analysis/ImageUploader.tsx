import React, { useRef } from 'react';
import { Upload, X, Image as ImageIcon, Plus, AlertCircle, FileCode } from 'lucide-react';
import { useAnalysis } from '../../context/AnalysisContext';
import { Badge } from '../common/Badge';

export const ImageUploader: React.FC = () => {
  const {
    uploadedFiles,
    addUploadedFiles,
    removeUploadedFile,
    updateFileRole,
    updateFileModality,
    clearUploads,
  } = useAnalysis();

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addUploadedFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      addUploadedFiles(Array.from(e.target.files));
    }
  };

  return (
    <div className="space-y-4">
      {/* Upload Dropzone */}
      <div
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className="border-2 border-dashed border-border-strong hover:border-accent-cyan/60 bg-background-surface/40 hover:bg-background-surface/80 transition-all rounded-xl p-5 text-center cursor-pointer flex flex-col items-center justify-center gap-2 group"
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".png,.jpg,.jpeg,.tif,.tiff"
          onChange={handleFileSelect}
          className="hidden"
        />
        <div className="w-10 h-10 rounded-full bg-cyan-500/10 text-cyan-400 flex items-center justify-center transition group-hover:scale-110">
          <Upload className="w-5 h-5" />
        </div>
        <div>
          <p className="text-xs font-semibold text-text-primary">
            Drop raster rasters or <span className="text-cyan-400 underline">browse</span>
          </p>
          <p className="text-[11px] text-text-muted mt-0.5">
            Supported: GeoTIFF (.tif, .tiff), PNG, JPEG (1 image for VQA, 2 for Change/Fusion)
          </p>
        </div>
      </div>

      {/* Selected Uploads List */}
      {uploadedFiles.length > 0 && (
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
              Selected Rasters ({uploadedFiles.length})
            </span>
            <button
              onClick={clearUploads}
              className="text-[11px] text-rose-400 hover:text-rose-300 transition hover:underline"
            >
              Clear All
            </button>
          </div>

          <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
            {uploadedFiles.map((item, idx) => {
              const isTiff = item.file.name.endsWith('.tif') || item.file.name.endsWith('.tiff');
              return (
                <div
                  key={item.id}
                  className="glass-card rounded-lg p-2.5 flex items-center gap-3 border border-border-subtle hover:border-cyan-500/30"
                >
                  {/* Thumbnail / Icon */}
                  <div className="w-12 h-12 rounded bg-background shrink-0 border border-border-subtle overflow-hidden flex items-center justify-center">
                    {isTiff ? (
                      <FileCode className="w-6 h-6 text-accent-cyan" />
                    ) : (
                      <img
                        src={item.previewUrl}
                        alt={item.file.name}
                        className="w-full h-full object-cover"
                      />
                    )}
                  </div>

                  {/* Details & Controls */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-xs font-medium text-text-primary truncate" title={item.file.name}>
                        {item.file.name}
                      </p>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          removeUploadedFile(item.id);
                        }}
                        className="text-text-muted hover:text-rose-400 transition"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    <div className="flex items-center gap-2 mt-1.5">
                      {/* Role Input */}
                      <input
                        type="text"
                        value={item.role}
                        onChange={(e) => updateFileRole(item.id, e.target.value)}
                        className="bg-background-secondary border border-border-subtle rounded px-1.5 py-0.5 text-[10px] text-cyan-300 w-28 focus:outline-none focus:border-cyan-500 font-mono"
                        placeholder="Role label"
                      />

                      {/* Modality Selector */}
                      <select
                        value={item.modality}
                        onChange={(e) =>
                          updateFileModality(
                            item.id,
                            e.target.value as 'optical' | 'sar' | 'multispectral'
                          )
                        }
                        className="bg-background-secondary border border-border-subtle rounded px-1.5 py-0.5 text-[10px] text-text-secondary focus:outline-none focus:border-cyan-500 font-mono"
                      >
                        <option value="optical">Optical</option>
                        <option value="sar">SAR</option>
                        <option value="multispectral">Multispectral</option>
                      </select>

                      <span className="text-[10px] text-text-muted ml-auto font-mono">
                        {(item.file.size / 1024).toFixed(0)} KB
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="text-[11px] text-emerald-400/90 bg-emerald-950/40 border border-emerald-800/40 rounded-lg p-2 flex items-center gap-1.5">
            <span className="text-xs">✓</span>
            <span>
              Images will be uploaded to <code className="font-mono text-emerald-300">/api/v1/query/multipart</code> for pipeline execution.
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

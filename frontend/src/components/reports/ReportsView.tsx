import React, { useState } from 'react';
import { FileText, Download, Eye, FileCode, CheckCircle, Loader2, Sparkles } from 'lucide-react';
import { useAnalysis } from '../../context/AnalysisContext';
import { api } from '../../services/api';
import { ReportGenerationResponse } from '../../types/api';
import { Badge } from '../common/Badge';

export const ReportsView: React.FC = () => {
  const { result } = useAnalysis();
  const [format, setFormat] = useState<'html' | 'markdown' | 'json'>('html');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generatedReport, setGeneratedReport] = useState<ReportGenerationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!result) return;
    setIsGenerating(true);
    setError(null);
    try {
      const res = await api.generateReport({
        response: result,
        format: format,
      });
      setGeneratedReport(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Report generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  if (!result) {
    return (
      <div className="glass-panel rounded-xl p-12 text-center max-w-2xl mx-auto space-y-4 border border-border-subtle my-8">
        <div className="w-14 h-14 rounded-2xl bg-background-surface border border-border-subtle flex items-center justify-center text-text-muted mx-auto">
          <FileText className="w-7 h-7" />
        </div>
        <h3 className="text-base font-semibold text-text-primary">
          No Query Result Available for Report Generation
        </h3>
        <p className="text-xs text-text-muted max-w-md mx-auto">
          Execute an analysis query in the Analysis Workspace to generate a structured intelligence report with grounded visual evidence.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Header card */}
      <div className="glass-panel rounded-xl p-6 border border-border-strong space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-bold text-text-primary flex items-center gap-2">
              <FileText className="w-5 h-5 text-cyan-400" />
              Intelligence Report Generator
            </h3>
            <p className="text-xs text-text-muted mt-1">
              Generate persistent, audit-ready remote-sensing intelligence dossiers with embedded evidence.
            </p>
          </div>

          {/* Format selector */}
          <div className="flex items-center gap-1.5 bg-background p-1 rounded-xl border border-border-subtle">
            {(['html', 'markdown', 'json'] as const).map((fmt) => (
              <button
                key={fmt}
                onClick={() => setFormat(fmt)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium uppercase tracking-wider transition ${
                  format === fmt
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'text-text-muted hover:text-text-primary'
                }`}
              >
                {fmt}
              </button>
            ))}
          </div>
        </div>

        {/* Action Button */}
        <div className="flex items-center justify-between pt-4 border-t border-border-subtle">
          <div className="text-xs text-text-muted">
            Target Query: <strong className="text-text-secondary truncate">{result.query}</strong>
          </div>

          <button
            onClick={handleGenerate}
            disabled={isGenerating}
            className="flex items-center gap-2 px-5 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-medium text-xs shadow-glow-cyan transition active:scale-95 disabled:opacity-50"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Generating {format.toUpperCase()}...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-3.5 h-3.5" />
                <span>Generate {format.toUpperCase()} Report</span>
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-xs text-rose-300">
          {error}
        </div>
      )}

      {/* Generated Report Card */}
      {generatedReport && (
        <div className="glass-panel rounded-xl p-6 border border-emerald-800/60 space-y-4 animate-fade-in">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border-subtle pb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center border border-emerald-800/40">
                <CheckCircle className="w-5 h-5" />
              </div>
              <div>
                <h4 className="text-sm font-semibold text-text-primary flex items-center gap-2">
                  {generatedReport.file_name}
                  <Badge variant="emerald" size="sm">
                    Ready
                  </Badge>
                </h4>
                <p className="text-xs font-mono text-cyan-400">
                  Report ID: {generatedReport.report_id}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <a
                href={api.getReportDownloadUrl(generatedReport.report_id)}
                download={generatedReport.file_name}
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-sm transition"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download Report ({generatedReport.format.toUpperCase()})</span>
              </a>
            </div>
          </div>

          {/* Preview window */}
          <div className="space-y-2">
            <h5 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
              Document Preview (First 500 characters)
            </h5>
            <div className="bg-background p-4 rounded-lg border border-border-subtle overflow-x-auto">
              <pre className="text-xs font-mono text-text-secondary whitespace-pre-wrap">
                {generatedReport.content_preview}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

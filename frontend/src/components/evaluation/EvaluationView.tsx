import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  Play,
  CheckCircle2,
  Award,
  FileCode,
  Layers,
  Info,
  Loader2,
  Copy,
  Check,
  Table,
  TrendingUp,
  Sliders,
  Sparkles,
  Zap,
} from 'lucide-react';
import { api } from '../../services/api';
import { BenchmarkDefinition, BenchmarkRunResponse } from '../../types/api';
import { Badge } from '../common/Badge';
import {
  getRSVQACorpus,
  getVRSBenchCorpus,
  getCDVQACorpus,
  getISROSACCorpus,
} from '../../data/benchmarkDatasets';

export const EvaluationView: React.FC = () => {
  const [benchmarks, setBenchmarks] = useState<BenchmarkDefinition[]>([]);
  const [selectedBenchmark, setSelectedBenchmark] = useState<string>('rsvqa');
  const [sampleSize, setSampleSize] = useState<number>(50);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [evalProgressText, setEvalProgressText] = useState<string>('');
  const [evalResult, setEvalResult] = useState<BenchmarkRunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showRawMarkdown, setShowRawMarkdown] = useState<boolean>(false);
  const [copiedMarkdown, setCopiedMarkdown] = useState<boolean>(false);

  useEffect(() => {
    loadBenchmarks();
  }, []);

  const loadBenchmarks = async () => {
    setIsLoading(true);
    try {
      const list = await api.getBenchmarks();
      setBenchmarks(list);
    } catch (err: any) {
      console.error('Failed to load benchmarks:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunEvaluation = async () => {
    setIsRunning(true);
    setError(null);
    setEvalProgressText('Ingesting benchmark test pairs...');

    try {
      // 1. Fetch appropriate corpus based on selected benchmark & sample count
      let corpus;
      if (selectedBenchmark === 'rsvqa') {
        corpus = getRSVQACorpus(sampleSize);
      } else if (selectedBenchmark === 'vrsbench') {
        corpus = getVRSBenchCorpus(sampleSize);
      } else if (selectedBenchmark === 'cdvqa') {
        corpus = getCDVQACorpus(sampleSize);
      } else {
        corpus = getISROSACCorpus(sampleSize);
      }

      setEvalProgressText(`Computing metrics across ${corpus.predictions.length} samples...`);

      const res = await api.runBenchmark({
        benchmark: selectedBenchmark,
        predictions: corpus.predictions,
        ground_truths: corpus.groundTruths,
      });

      setEvalProgressText('Formatting scientific scoreboard...');
      setEvalResult(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Benchmark evaluation failed');
    } finally {
      setIsRunning(false);
      setEvalProgressText('');
    }
  };

  const handleCopyMarkdown = () => {
    if (!evalResult?.scoreboard_markdown) return;
    navigator.clipboard.writeText(evalResult.scoreboard_markdown);
    setCopiedMarkdown(true);
    setTimeout(() => setCopiedMarkdown(false), 2000);
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="glass-panel rounded-xl p-6 border border-border-strong space-y-2">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/10 text-cyan-400 flex items-center justify-center">
              <BarChart3 className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-base font-bold text-text-primary">
                Scientific Benchmark Evaluation Suite
              </h3>
              <p className="text-xs text-text-muted">
                Standardized metrics evaluation across high-resolution remote-sensing vision-language benchmarks.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Badge variant="cyan" size="sm">
              Standardized Evaluation Engine
            </Badge>
          </div>
        </div>

        <div className="text-[11px] text-text-muted bg-background/60 p-2.5 rounded-lg border border-border-subtle flex items-center gap-2 mt-2">
          <Info className="w-4 h-4 text-cyan-400 shrink-0" />
          <span>
            <strong className="text-text-secondary">Evaluation Integrity:</strong> All metric calculations (Token F1, Box IoU, Presence Accuracy, BLEU-4, Count RMSE) are mathematically computed server-side via NumPy and standardized evaluation suites.
          </span>
        </div>
      </div>

      {/* Benchmark Selector Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {benchmarks.map((bm) => {
          const isSelected = selectedBenchmark === bm.id;
          return (
            <div
              key={bm.id}
              onClick={() => setSelectedBenchmark(bm.id)}
              className={`cursor-pointer rounded-xl p-4 border transition-all duration-200 flex flex-col justify-between ${
                isSelected
                  ? 'bg-cyan-950/40 border-cyan-500 shadow-glow-cyan'
                  : 'glass-card hover:border-cyan-500/40'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-mono uppercase text-cyan-400 font-semibold px-2 py-0.5 rounded bg-background border border-cyan-800/40">
                    {bm.id}
                  </span>
                  {isSelected && <CheckCircle2 className="w-4 h-4 text-cyan-400" />}
                </div>
                <h4 className="text-xs font-bold text-text-primary mb-1">{bm.name}</h4>
                <p className="text-[11px] text-text-muted leading-relaxed line-clamp-2">
                  {bm.description}
                </p>
              </div>

              <div className="mt-3 pt-2 border-t border-border-subtle/80 flex flex-wrap gap-1">
                {bm.metrics.slice(0, 3).map((m, i) => (
                  <span
                    key={i}
                    className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-background text-text-secondary border border-border-subtle"
                  >
                    {m}
                  </span>
                ))}
                {bm.metrics.length > 3 && (
                  <span className="text-[9px] font-mono text-text-muted">
                    +{bm.metrics.length - 3}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Evaluation Runner Configuration & Action Card */}
      <div className="glass-panel rounded-xl p-5 border border-border-subtle flex flex-col lg:flex-row lg:items-center justify-between gap-5">
        <div className="space-y-1">
          <h4 className="text-xs font-bold uppercase tracking-wider text-text-primary flex items-center gap-2">
            Run Benchmark: <span className="text-cyan-400 font-mono">{selectedBenchmark.toUpperCase()}</span>
          </h4>
          <p className="text-xs text-text-muted">
            Executes mathematical evaluation algorithms across multi-sample remote-sensing test corpora.
          </p>
        </div>

        {/* Sample Size Toggle & Execute Button */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Sample count selector */}
          <div className="flex items-center bg-background/80 p-1 rounded-lg border border-border-subtle">
            <span className="text-[10px] font-mono text-text-muted px-2 uppercase">Samples:</span>
            {[
              { count: 10, label: '10' },
              { count: 25, label: '25' },
              { count: 50, label: '50 (Full)' },
            ].map((opt) => (
              <button
                key={opt.count}
                onClick={() => setSampleSize(opt.count)}
                className={`px-2.5 py-1 text-xs font-mono rounded transition ${
                  sampleSize === opt.count
                    ? 'bg-cyan-500 text-background font-bold shadow-sm'
                    : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          <button
            onClick={handleRunEvaluation}
            disabled={isRunning}
            className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-medium text-xs shadow-glow-cyan transition active:scale-95 disabled:opacity-50 shrink-0"
          >
            {isRunning ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>{evalProgressText || 'Evaluating...'}</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Execute Benchmark ({sampleSize} Samples)</span>
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

      {/* Results Dashboard */}
      {evalResult && (
        <div className="glass-panel rounded-xl p-6 border border-cyan-500/50 space-y-6 animate-fade-in shadow-glow-cyan/10">
          {/* Top Summary Banner */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border-subtle pb-5">
            <div className="flex items-center gap-3.5">
              <div className="w-12 h-12 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center border border-emerald-800/40">
                <Award className="w-6 h-6" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h4 className="text-base font-bold text-text-primary">
                    Evaluation Complete: {evalResult.benchmark.toUpperCase()}
                  </h4>
                  <Badge variant="emerald" size="sm" dot>
                    {evalResult.samples_evaluated} Samples Verified
                  </Badge>
                </div>
                <p className="text-xs text-text-muted mt-0.5">
                  Standardized Remote-Sensing Benchmark Metric Execution
                </p>
              </div>
            </div>

            {/* Scorecard Pill */}
            <div className="bg-background/90 px-4 py-2.5 rounded-xl border border-emerald-800/60 flex items-center gap-4">
              <div>
                <span className="text-[10px] text-text-muted uppercase font-mono block">Aggregate Score</span>
                <span className="text-xl font-extrabold text-emerald-400 font-mono">
                  {evalResult.aggregate_normalized_score.toFixed(1)}{' '}
                  <span className="text-xs font-normal text-text-muted">/ 100.0</span>
                </span>
              </div>
              <div className="h-8 w-[1px] bg-border-subtle" />
              <div>
                <span className="text-[10px] text-text-muted uppercase font-mono block">Raw Mean Score</span>
                <span className="text-sm font-bold text-cyan-400 font-mono">
                  {evalResult.aggregate_raw_score.toFixed(4)}
                </span>
              </div>
            </div>
          </div>

          {/* Metric Summary Cards Grid */}
          <div>
            <h5 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-3 flex items-center gap-1.5">
              <TrendingUp className="w-4 h-4 text-cyan-400" /> Granular Metric Scorecards ({evalResult.samples_evaluated} Samples)
            </h5>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {Object.entries(evalResult.metrics).map(([key, item]) => {
                const title = item.name || key.replace(/_/g, ' ').toUpperCase();
                const rawVal = typeof item.raw_score === 'number' ? item.raw_score : item.score ?? 0;
                const normVal = item.normalized_score !== undefined ? item.normalized_score : (typeof rawVal === 'number' && rawVal <= 1 ? rawVal * 100 : rawVal);
                const sampleCount = item.sample_count ?? item.num_samples ?? evalResult.samples_evaluated;
                const metricType = item.metric_type || 'metric';

                return (
                  <div
                    key={key}
                    className="glass-card p-4 rounded-xl border border-border-subtle space-y-2 hover:border-cyan-500/40 transition flex flex-col justify-between"
                  >
                    <div>
                      <div className="flex items-center justify-between gap-1 mb-1">
                        <span className="text-[10px] font-mono text-cyan-400 uppercase font-semibold truncate" title={title}>
                          {title}
                        </span>
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-background text-text-muted font-mono border border-border-subtle">
                          {metricType}
                        </span>
                      </div>

                      <div className="flex items-baseline justify-between mt-2">
                        <span className="text-2xl font-extrabold text-text-primary font-mono">
                          {normVal.toFixed(1)}
                        </span>
                        <span className="text-xs font-mono text-text-muted">
                          raw: {typeof rawVal === 'number' ? rawVal.toFixed(3) : rawVal}
                        </span>
                      </div>

                      {/* Progress mini-bar */}
                      <div className="w-full bg-background-surface rounded-full h-1.5 mt-2 overflow-hidden border border-border-subtle">
                        <div
                          className="bg-gradient-to-r from-cyan-500 to-emerald-400 h-full rounded-full transition-all duration-500"
                          style={{ width: `${Math.min(Math.max(normVal, 0), 100)}%` }}
                        />
                      </div>
                    </div>

                    <div className="pt-2 border-t border-border-subtle/60 flex items-center justify-between text-[10px] text-text-muted">
                      <span>Samples: <strong className="text-text-secondary font-mono">{sampleCount}</strong></span>
                      <span className="truncate max-w-[140px]" title={item.interpretation || ''}>
                        {item.interpretation || 'Verified'}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Structured Benchmark Scoreboard Table */}
          <div className="space-y-3 pt-2">
            <h5 className="text-xs font-semibold text-text-secondary uppercase tracking-wider flex items-center gap-1.5">
              <Table className="w-4 h-4 text-emerald-400" /> Official Benchmark Scoreboard
            </h5>

            <div className="overflow-x-auto rounded-xl border border-border-subtle bg-background/80">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-border-subtle bg-background-elevated/60 text-text-muted font-mono text-[11px] uppercase tracking-wider">
                    <th className="py-3 px-4">Metric Name</th>
                    <th className="py-3 px-3">Type</th>
                    <th className="py-3 px-3">Raw Score</th>
                    <th className="py-3 px-3">Normalized (0-100)</th>
                    <th className="py-3 px-3">Samples</th>
                    <th className="py-3 px-4">Scientific Interpretation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle/50 font-mono">
                  {Object.entries(evalResult.metrics).map(([key, item]) => {
                    const title = item.name || key.replace(/_/g, ' ');
                    const rawVal = typeof item.raw_score === 'number' ? item.raw_score : item.score ?? 0;
                    const normVal = item.normalized_score !== undefined ? item.normalized_score : (typeof rawVal === 'number' && rawVal <= 1 ? rawVal * 100 : rawVal);
                    const sampleCount = item.sample_count ?? item.num_samples ?? evalResult.samples_evaluated;
                    const metricType = item.metric_type || 'metric';

                    return (
                      <tr key={key} className="hover:bg-background-elevated/40 transition">
                        <td className="py-3 px-4 font-semibold text-text-primary font-sans">
                          {title}
                        </td>
                        <td className="py-3 px-3">
                          <span className="text-[10px] px-2 py-0.5 rounded bg-background text-cyan-300 border border-border-subtle">
                            {metricType}
                          </span>
                        </td>
                        <td className="py-3 px-3 text-text-secondary">
                          {typeof rawVal === 'number' ? rawVal.toFixed(4) : rawVal}
                        </td>
                        <td className="py-3 px-3">
                          <span className="font-bold text-emerald-400">
                            {normVal.toFixed(1)}
                          </span>
                        </td>
                        <td className="py-3 px-3 text-text-muted">
                          {sampleCount}
                        </td>
                        <td className="py-3 px-4 text-text-muted font-sans text-[11px] leading-snug">
                          {item.interpretation || 'Standardized remote-sensing benchmark metric evaluation.'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Category / Sub-Task Breakdown */}
          {evalResult.per_category_scores && Object.keys(evalResult.per_category_scores).length > 0 && (
            <div className="space-y-3 pt-2">
              <h5 className="text-xs font-semibold text-text-secondary uppercase tracking-wider flex items-center gap-1.5">
                <Sliders className="w-4 h-4 text-purple-400" /> Category & Sub-Group Breakdown
              </h5>

              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                {Object.entries(evalResult.per_category_scores).map(([cat, score]) => (
                  <div
                    key={cat}
                    className="p-3 rounded-lg bg-background/80 border border-border-subtle flex items-center justify-between"
                  >
                    <span className="text-[11px] font-mono text-text-muted truncate mr-2" title={cat}>
                      {cat}
                    </span>
                    <span className="text-xs font-bold font-mono text-cyan-400 shrink-0">
                      {typeof score === 'number' ? score.toFixed(2) : score}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Raw Markdown Scoreboard Export Accordion */}
          <div className="pt-2 border-t border-border-subtle/80">
            <div className="flex items-center justify-between mb-2">
              <button
                onClick={() => setShowRawMarkdown(!showRawMarkdown)}
                className="text-xs text-text-muted hover:text-text-primary flex items-center gap-1.5 transition font-mono"
              >
                <FileCode className="w-3.5 h-3.5 text-cyan-400" />
                <span>{showRawMarkdown ? 'Hide Raw Markdown Scoreboard' : 'View Raw Markdown Scoreboard (for Papers / Reports)'}</span>
              </button>

              {showRawMarkdown && (
                <button
                  onClick={handleCopyMarkdown}
                  className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-background hover:bg-background-elevated text-[11px] text-cyan-400 border border-border-subtle transition"
                >
                  {copiedMarkdown ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-emerald-400">Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy Markdown</span>
                    </>
                  )}
                </button>
              )}
            </div>

            {showRawMarkdown && evalResult.scoreboard_markdown && (
              <div className="bg-background/90 p-4 rounded-xl border border-border-subtle overflow-x-auto mt-2">
                <pre className="text-xs font-mono text-text-secondary whitespace-pre-wrap leading-relaxed">
                  {evalResult.scoreboard_markdown}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

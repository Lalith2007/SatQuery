import React, { useEffect, useRef } from 'react';
import { Send, Sparkles, Trash2, Command } from 'lucide-react';
import { useAnalysis } from '../../context/AnalysisContext';

export const QueryComposer: React.FC = () => {
  const { query, setQuery, executeAnalysis, isLoading, inputMode, selectedPreset } =
    useAnalysis();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!isLoading && query.trim()) {
        executeAnalysis();
      }
    }
  };

  const samplePrompts = [
    'What is the least land covered in this image?',
    'Where is the water?',
    'What changed between these two acquisition dates?',
    'Has there been any built-up expansion or construction?',
  ];

  return (
    <div className="glass-panel rounded-xl p-4 space-y-3 border border-border-strong shadow-lg">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold text-text-secondary uppercase tracking-wider flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          Natural-Language Query
        </label>
        {query && (
          <button
            onClick={() => setQuery('')}
            className="text-[11px] text-text-muted hover:text-rose-400 flex items-center gap-1 transition"
            title="Clear Query"
          >
            <Trash2 className="w-3 h-3" />
            Clear
          </button>
        )}
      </div>

      {/* Textarea */}
      <div className="relative">
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask any question about land cover, temporal change, spatial features, or cross-modal imagery..."
          rows={3}
          className="w-full bg-background border border-border-strong focus:border-accent-cyan rounded-lg p-3 text-sm text-text-primary placeholder:text-text-muted/60 focus:outline-none focus:ring-1 focus:ring-accent-cyan transition resize-none font-sans"
        />
      </div>

      {/* Suggested prompts */}
      <div className="space-y-1.5">
        <div className="text-[10px] text-text-muted font-medium">Quick Suggestions:</div>
        <div className="flex flex-wrap gap-1.5">
          {samplePrompts.map((prompt, idx) => (
            <button
              key={idx}
              onClick={() => setQuery(prompt)}
              className="text-[11px] px-2.5 py-1 rounded-md bg-background-elevated/60 hover:bg-background-elevated hover:text-cyan-300 text-text-secondary border border-border-subtle hover:border-cyan-500/30 transition text-left"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>

      {/* Action Footer */}
      <div className="flex items-center justify-between pt-2 border-t border-border-subtle">
        <div className="flex items-center gap-1.5 text-[11px] text-text-muted hidden sm:flex">
          <Command className="w-3 h-3" />
          <span>Press</span>
          <kbd className="px-1.5 py-0.5 rounded bg-background border border-border-subtle text-[10px] font-mono">
            ⌘+Enter
          </kbd>
          <span>to execute</span>
        </div>

        <button
          onClick={() => executeAnalysis()}
          disabled={isLoading || !query.trim()}
          className={`flex items-center gap-2 px-5 py-2 rounded-lg font-medium text-xs transition shadow-md ${
            isLoading || !query.trim()
              ? 'bg-cyan-950/50 text-cyan-700 cursor-not-allowed border border-cyan-900/40'
              : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-glow-cyan cursor-pointer active:scale-95'
          }`}
        >
          <Send className="w-3.5 h-3.5" />
          <span>{isLoading ? 'Analyzing Imagery...' : 'Run Agent Analysis'}</span>
        </button>
      </div>
    </div>
  );
};

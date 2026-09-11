import { useState } from 'react';
import { Zap, Calendar, CalendarRange, Share2, Check } from 'lucide-react';
import { PortalViewMode, DateTimeRange } from '../types';

interface ViewToggleProps {
  viewMode: PortalViewMode;
  onViewModeChange: (mode: PortalViewMode) => void;
  priceArea: 'DK1' | 'DK2';
  dateRange?: DateTimeRange;
}

export function ViewToggle({ viewMode, onViewModeChange, priceArea, dateRange }: ViewToggleProps) {
  const [copied, setCopied] = useState(false);

  const handleCopyLink = async () => {
    try {
      const url = new URL(window.location.href);
      if (viewMode === 'today') {
        url.pathname = '/all';
        url.searchParams.delete('view');
        url.searchParams.delete('start');
        url.searchParams.delete('end');
      } else if (viewMode === 'range' && dateRange) {
        url.pathname = '/all';
        url.searchParams.set('view', 'range');
        if (dateRange.start) url.searchParams.set('start', dateRange.start);
        if (dateRange.end) url.searchParams.set('end', dateRange.end);
      } else {
        url.pathname = '/';
        url.searchParams.delete('view');
        url.searchParams.delete('start');
        url.searchParams.delete('end');
      }

      if (priceArea === 'DK2') {
        url.searchParams.set('zone', 'DK2');
      } else {
        url.searchParams.delete('zone');
      }

      await navigator.clipboard.writeText(url.toString());
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback if clipboard API fails
      setCopied(false);
    }
  };

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-3 w-full bg-slate-800/80 p-2 sm:p-2.5 rounded-xl border border-slate-700/80 backdrop-blur-sm">
      {/* Mode Buttons */}
      <div className="flex items-center gap-1.5 p-1 bg-slate-900/90 rounded-lg border border-slate-700/60 w-full sm:w-auto flex-wrap">
        <button
          type="button"
          onClick={() => onViewModeChange('live')}
          className={`flex-1 sm:flex-none flex items-center justify-center gap-2 px-3.5 py-2 rounded-md text-xs sm:text-sm font-medium transition-all cursor-pointer ${
            viewMode === 'live'
              ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
          }`}
        >
          <Zap className="w-4 h-4 text-amber-400" />
          <span>Live (Next 4)</span>
        </button>

        <button
          type="button"
          onClick={() => onViewModeChange('today')}
          className={`flex-1 sm:flex-none flex items-center justify-center gap-2 px-3.5 py-2 rounded-md text-xs sm:text-sm font-medium transition-all cursor-pointer ${
            viewMode === 'today'
              ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
          }`}
        >
          <Calendar className="w-4 h-4 text-emerald-400" />
          <span>Today (All 96 Quarters)</span>
        </button>

        <button
          type="button"
          onClick={() => onViewModeChange('range')}
          className={`flex-1 sm:flex-none flex items-center justify-center gap-2 px-3.5 py-2 rounded-md text-xs sm:text-sm font-medium transition-all cursor-pointer ${
            viewMode === 'range'
              ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
          }`}
        >
          <CalendarRange className="w-4 h-4 text-cyan-400" />
          <span>Custom Range</span>
        </button>
      </div>

      {/* Share / Copy Direct Link */}
      <button
        type="button"
        onClick={handleCopyLink}
        className={`flex items-center justify-center gap-2 px-3.5 py-2 rounded-lg text-xs sm:text-sm font-medium border transition-all cursor-pointer ${
          copied
            ? 'bg-emerald-600/20 text-emerald-300 border-emerald-500/40'
            : 'bg-slate-900/80 text-slate-300 border-slate-700 hover:bg-slate-750 hover:text-white'
        }`}
        title="Copy direct shareable link for this view"
      >
        {copied ? (
          <>
            <Check className="w-4 h-4 text-emerald-400" />
            <span>Link Copied!</span>
          </>
        ) : (
          <>
            <Share2 className="w-4 h-4 text-indigo-400" />
            <span>Copy Shareable Link</span>
          </>
        )}
      </button>
    </div>
  );
}

import { useState, useEffect } from 'react';
import { Calendar, Clock, MapPin, Filter, RotateCcw, Share2, Check, TrendingUp } from 'lucide-react';
import { DateTimeRange } from '../types';
import { getTodayDateString } from '../hooks/usePredictions';

interface RangeFilterBarProps {
  priceArea: 'DK1' | 'DK2';
  onPriceAreaChange: (zone: 'DK1' | 'DK2') => void;
  marketMode: 'INTRADAY_D0' | 'DAY_AHEAD_D1';
  onMarketModeChange: (mode: 'INTRADAY_D0' | 'DAY_AHEAD_D1') => void;
  dateRange: DateTimeRange;
  onRangeApply: (range: DateTimeRange) => void;
  resultCount: number;
  loading: boolean;
}

export function RangeFilterBar({
  priceArea,
  onPriceAreaChange,
  marketMode,
  onMarketModeChange,
  dateRange,
  onRangeApply,
  resultCount,
  loading,
}: RangeFilterBarProps) {
  const [startVal, setStartVal] = useState(dateRange.start);
  const [endVal, setEndVal] = useState(dateRange.end);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setStartVal(dateRange.start);
    setEndVal(dateRange.end);
  }, [dateRange.start, dateRange.end]);

  const handleApply = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    onRangeApply({ start: startVal, end: endVal });
  };

  // Helper to format ISO or local Date to YYYY-MM-DDTHH:mm
  const formatDateTimeLocal = (d: Date) => {
    const pad = (n: number) => n.toString().padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
  };

  const applyPreset = (preset: 'today' | 'next24' | 'past24' | 'morning' | 'evening') => {
    const now = new Date();
    let s = new Date();
    let e = new Date();

    if (preset === 'today') {
      const today = getTodayDateString();
      const newStart = `${today}T00:00`;
      const newEnd = `${today}T23:45`;
      setStartVal(newStart);
      setEndVal(newEnd);
      onRangeApply({ start: newStart, end: newEnd });
      return;
    } else if (preset === 'next24') {
      const startIso = formatDateTimeLocal(now);
      const endD = new Date(now.getTime() + 24 * 60 * 60 * 1000);
      const endIso = formatDateTimeLocal(endD);
      setStartVal(startIso);
      setEndVal(endIso);
      onRangeApply({ start: startIso, end: endIso });
      return;
    } else if (preset === 'past24') {
      s = new Date(now.getTime() - 24 * 60 * 60 * 1000);
    } else if (preset === 'morning') {
      s.setHours(6, 0, 0, 0);
      e.setHours(12, 0, 0, 0);
    } else if (preset === 'evening') {
      s.setHours(17, 0, 0, 0);
      e.setHours(21, 0, 0, 0);
    }

    const newStart = formatDateTimeLocal(s);
    const newEnd = formatDateTimeLocal(e);
    setStartVal(newStart);
    setEndVal(newEnd);
    onRangeApply({ start: newStart, end: newEnd });
  };

  const handleCopyShareLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // fallback
    }
  };

  return (
    <div className="w-full bg-slate-800 p-4 sm:p-5 rounded-xl border border-slate-700 shadow-xl flex flex-col gap-4">
      {/* Top bar: Market Mode + Price Area Selector + Presets */}
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 pb-4 border-b border-slate-700/80">
        <div className="flex items-center gap-3.5 flex-wrap w-full lg:w-auto">
          {/* Market Mode Toggle */}
          <div className="flex items-center gap-2">
            <span className="text-slate-300 text-xs sm:text-sm font-semibold whitespace-nowrap">Market:</span>
            <div className="flex bg-slate-900 p-1 rounded-lg border border-slate-700 shadow-sm">
              <button
                type="button"
                onClick={() => onMarketModeChange('INTRADAY_D0')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-semibold text-xs sm:text-sm transition-all cursor-pointer ${
                  marketMode === 'INTRADAY_D0'
                    ? 'bg-blue-600 text-white shadow-md'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                }`}
              >
                <TrendingUp className="w-3.5 h-3.5 text-blue-300" />
                <span>Intraday Continuous</span>
              </button>

              <button
                type="button"
                onClick={() => onMarketModeChange('DAY_AHEAD_D1')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-semibold text-xs sm:text-sm transition-all cursor-pointer ${
                  marketMode === 'DAY_AHEAD_D1'
                    ? 'bg-amber-600 text-white shadow-md'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                }`}
              >
                <Clock className="w-3.5 h-3.5 text-amber-300" />
                <span>Day-Ahead Auction</span>
              </button>
            </div>
          </div>

          {/* Price Area Selector (DK1 / DK2) */}
          <div className="flex items-center gap-2">
            <span className="text-slate-300 text-xs sm:text-sm font-semibold whitespace-nowrap flex items-center gap-1">
              <MapPin className="w-3.5 h-3.5 text-indigo-400" />
              Area:
            </span>

            <div className="flex bg-slate-900 p-1 rounded-lg border border-slate-700 shadow-sm">
              {(['DK1', 'DK2'] as const).map((zone) => (
                <button
                  key={zone}
                  type="button"
                  onClick={() => onPriceAreaChange(zone)}
                  className={`px-3 py-1.5 rounded-md font-semibold text-xs sm:text-sm transition-all cursor-pointer ${
                    priceArea === zone
                      ? 'bg-indigo-600 text-white shadow-md'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                  }`}
                >
                  {zone}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Quick Presets */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-xs text-slate-400 mr-1 flex items-center gap-1">
            <Clock className="w-3 h-3 text-slate-500" />
            Presets:
          </span>
          <button
            type="button"
            onClick={() => applyPreset('today')}
            className="px-2.5 py-1 rounded bg-slate-900 text-slate-300 hover:bg-slate-750 hover:text-white border border-slate-700 text-xs font-medium transition-colors cursor-pointer"
          >
            Today (00-24h)
          </button>
          <button
            type="button"
            onClick={() => applyPreset('next24')}
            className="px-2.5 py-1 rounded bg-slate-900 text-slate-300 hover:bg-slate-750 hover:text-white border border-slate-700 text-xs font-medium transition-colors cursor-pointer"
          >
            Next 96Q
          </button>
          <button
            type="button"
            onClick={() => applyPreset('morning')}
            className="px-2.5 py-1 rounded bg-slate-900 text-slate-300 hover:bg-slate-750 hover:text-white border border-slate-700 text-xs font-medium transition-colors cursor-pointer"
          >
            Morning (06-12)
          </button>
          <button
            type="button"
            onClick={() => applyPreset('evening')}
            className="px-2.5 py-1 rounded bg-slate-900 text-slate-300 hover:bg-slate-750 hover:text-white border border-slate-700 text-xs font-medium transition-colors cursor-pointer"
          >
            Evening (17-21)
          </button>
        </div>
      </div>

      {/* Bottom Bar: Datetime Range Inputs & Apply */}
      <form onSubmit={handleApply} className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3">
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 flex-1">
          {/* Start Datetime */}
          <div className="flex items-center gap-2 bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-700 flex-1">
            <Calendar className="w-4 h-4 text-emerald-400 shrink-0" />
            <div className="flex flex-col flex-1">
              <label className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">Start DateTime</label>
              <input
                type="datetime-local"
                value={startVal}
                onChange={(e) => setStartVal(e.target.value)}
                className="bg-transparent text-slate-100 text-xs sm:text-sm font-mono focus:outline-none cursor-pointer"
              />
            </div>
          </div>

          <span className="text-slate-500 font-bold hidden sm:inline">→</span>

          {/* End Datetime */}
          <div className="flex items-center gap-2 bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-700 flex-1">
            <Calendar className="w-4 h-4 text-rose-400 shrink-0" />
            <div className="flex flex-col flex-1">
              <label className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">End DateTime</label>
              <input
                type="datetime-local"
                value={endVal}
                onChange={(e) => setEndVal(e.target.value)}
                className="bg-transparent text-slate-100 text-xs sm:text-sm font-mono focus:outline-none cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Action Buttons & Counters */}
        <div className="flex items-center gap-2 justify-end flex-wrap">
          <button
            type="submit"
            disabled={loading}
            className="flex items-center justify-center gap-1.5 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs sm:text-sm font-semibold transition-all shadow-md shadow-indigo-600/30 cursor-pointer disabled:opacity-50"
          >
            <Filter className="w-4 h-4" />
            <span>{loading ? 'Filtering...' : 'Apply Filter'}</span>
          </button>

          <button
            type="button"
            onClick={() => applyPreset('today')}
            className="p-2 bg-slate-900 text-slate-400 hover:text-white rounded-lg border border-slate-700 transition-colors cursor-pointer"
            title="Reset to Today"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          <button
            type="button"
            onClick={handleCopyShareLink}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold border transition-colors cursor-pointer ${
              copied
                ? 'bg-emerald-600/20 text-emerald-300 border-emerald-500/40'
                : 'bg-slate-900 text-slate-300 border-slate-700 hover:bg-slate-750 hover:text-white'
            }`}
            title="Copy shareable direct link for this date range"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span>Copied</span>
              </>
            ) : (
              <>
                <Share2 className="w-3.5 h-3.5 text-indigo-400" />
                <span>Share</span>
              </>
            )}
          </button>

          <div className="text-xs font-mono px-3 py-2 bg-slate-900 rounded-lg border border-slate-700 text-slate-300 whitespace-nowrap">
            Quarters: <strong className="text-indigo-400">{resultCount}</strong>
          </div>
        </div>
      </form>
    </div>
  );
}

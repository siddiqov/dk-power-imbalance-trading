import { useState, useEffect } from 'react';
import { Calendar, Clock, MapPin, Filter, RotateCcw } from 'lucide-react';
import { DateTimeRange } from '../types';

interface RangeFilterBarProps {
  priceArea: 'DK1' | 'DK2';
  onPriceAreaChange: (zone: 'DK1' | 'DK2') => void;
  dateRange: DateTimeRange;
  onRangeApply: (range: DateTimeRange) => void;
  resultCount: number;
  loading: boolean;
}

export function RangeFilterBar({
  priceArea,
  onPriceAreaChange,
  dateRange,
  onRangeApply,
  resultCount,
  loading,
}: RangeFilterBarProps) {
  const [startVal, setStartVal] = useState(dateRange.start);
  const [endVal, setEndVal] = useState(dateRange.end);

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
      s.setHours(0, 0, 0, 0);
      e.setHours(23, 45, 0, 0);
    } else if (preset === 'next24') {
      e = new Date(now.getTime() + 24 * 60 * 60 * 1000);
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

  return (
    <div className="w-full bg-slate-800 p-4 sm:p-5 rounded-xl border border-slate-700 shadow-xl flex flex-col gap-4">
      {/* Top bar: Area Dropdown & Preset shortcuts */}
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 pb-4 border-b border-slate-700/80">
        {/* Area Dropdown */}
        <div className="flex items-center gap-3 w-full sm:w-auto">
          <div className="flex items-center gap-2 text-slate-300 text-sm font-semibold whitespace-nowrap">
            <MapPin className="w-4 h-4 text-indigo-400" />
            <span>Price Area:</span>
          </div>

          <div className="relative w-full sm:w-64">
            <select
              value={priceArea}
              onChange={(e) => onPriceAreaChange(e.target.value as 'DK1' | 'DK2')}
              className="w-full appearance-none bg-slate-900 text-slate-100 border border-slate-700 rounded-lg px-3.5 py-2 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 cursor-pointer pr-8"
            >
              <option value="DK1">DK1 — West Denmark (Jutland / Funen)</option>
              <option value="DK2">DK2 — East Denmark (Zealand / CPH)</option>
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2.5 text-slate-400">
              ▼
            </div>
          </div>
        </div>

        {/* Quick Presets */}
        <div className="flex items-center gap-1.5 flex-wrap w-full lg:w-auto">
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
            Next 24h
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
        <div className="flex items-center gap-2 justify-end">
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

          <div className="text-xs font-mono px-3 py-2 bg-slate-900 rounded-lg border border-slate-700 text-slate-300 whitespace-nowrap">
            Quarters: <strong className="text-indigo-400">{resultCount}</strong>
          </div>
        </div>
      </form>
    </div>
  );
}

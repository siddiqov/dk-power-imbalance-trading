import { Calendar, TrendingUp, TrendingDown, MinusCircle, Layers, BarChart3 } from 'lucide-react';
import { DaySummary } from '../types';

interface DailySummaryBannerProps {
  summary: DaySummary | null;
  dateStr: string;
  priceArea: 'DK1' | 'DK2';
  title?: string;
  subtitle?: React.ReactNode;
}

export function DailySummaryBanner({
  summary,
  dateStr,
  priceArea,
  title = 'Full Settlement Day Overview',
  subtitle,
}: DailySummaryBannerProps) {
  if (!summary) return null;

  const formatCurrency = (val: number) => `€ ${val.toFixed(2)}`;
  const formatSpread = (val: number) => {
    const formatted = formatCurrency(Math.abs(val));
    return val >= 0 ? `+${formatted}` : `-${formatted}`;
  };

  return (
    <div className="w-full bg-slate-800 p-5 rounded-xl border border-slate-700 shadow-lg flex flex-col gap-4">
      {/* Top Header Row: Date & Area Status */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-700/70">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-emerald-500/20 text-emerald-400 rounded-lg border border-emerald-500/30">
            <Calendar className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-semibold text-white">{title}</h2>
              <span className="text-xs px-2 py-0.5 rounded-full font-mono font-medium bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                {priceArea}
              </span>
            </div>
            <p className="text-xs text-slate-400">
              {subtitle || (
                <>
                  Delivery Date: <span className="font-mono text-slate-200">{dateStr}</span> ({summary.totalQuarters} 15-minute QH settlement intervals)
                </>
              )}
            </p>
          </div>
        </div>

        {/* Signal Counts Pills */}
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-950/60 border border-emerald-700/60 text-emerald-400 text-xs font-medium">
            <TrendingUp className="w-3.5 h-3.5" />
            <span>BUY: <strong className="font-bold text-white">{summary.buyCount}</strong></span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-rose-950/60 border border-rose-700/60 text-rose-400 text-xs font-medium">
            <TrendingDown className="w-3.5 h-3.5" />
            <span>SELL: <strong className="font-bold text-white">{summary.sellCount}</strong></span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-700 text-slate-400 text-xs font-medium">
            <MinusCircle className="w-3.5 h-3.5 text-slate-500" />
            <span>HOLD: <strong className="font-bold text-slate-200">{summary.holdCount}</strong></span>
          </div>
        </div>
      </div>

      {/* Metric Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <div className="bg-slate-900/70 p-3 rounded-lg border border-slate-750">
          <div className="text-[11px] font-medium text-slate-400 flex items-center gap-1">
            <Layers className="w-3.5 h-3.5 text-slate-500" />
            Total Quarters
          </div>
          <div className="text-lg font-bold font-mono text-slate-100 mt-1">
            {summary.totalQuarters} <span className="text-xs font-normal text-slate-400">/ 96</span>
          </div>
        </div>

        <div className="bg-slate-900/70 p-3 rounded-lg border border-slate-750">
          <div className="text-[11px] font-medium text-slate-400 flex items-center gap-1">
            <BarChart3 className="w-3.5 h-3.5 text-slate-500" />
            Avg Spot
          </div>
          <div className="text-lg font-bold font-mono text-slate-100 mt-1">
            {formatCurrency(summary.avgSpot)}
          </div>
        </div>

        <div className="bg-slate-900/70 p-3 rounded-lg border border-slate-750">
          <div className="text-[11px] font-medium text-slate-400 flex items-center gap-1">
            <BarChart3 className="w-3.5 h-3.5 text-indigo-400" />
            Avg Pred Imb
          </div>
          <div className="text-lg font-bold font-mono text-indigo-300 mt-1">
            {formatCurrency(summary.avgImbalance)}
          </div>
        </div>

        <div className="bg-slate-900/70 p-3 rounded-lg border border-slate-750">
          <div className="text-[11px] font-medium text-slate-400 flex items-center gap-1">
            <BarChart3 className="w-3.5 h-3.5 text-emerald-400" />
            Avg Spread
          </div>
          <div className={`text-lg font-bold font-mono mt-1 ${summary.avgSpread >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {formatSpread(summary.avgSpread)}
          </div>
        </div>

        <div className="bg-slate-900/70 p-3 rounded-lg border border-slate-750 col-span-2 sm:col-span-1">
          <div className="text-[11px] font-medium text-slate-400">
            Total Trade Vol
          </div>
          <div className="text-lg font-bold font-mono text-slate-100 mt-1">
            {summary.totalVolume.toFixed(1)} <span className="text-xs font-normal text-slate-400">MWh</span>
          </div>
        </div>
      </div>
    </div>
  );
}

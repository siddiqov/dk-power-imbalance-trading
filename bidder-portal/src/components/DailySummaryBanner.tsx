import { useState } from 'react';
import {
  Calendar,
  TrendingUp,
  TrendingDown,
  MinusCircle,
  Layers,
  BarChart3,
  DollarSign,
  Target,
  Activity,
  LayoutGrid,
  Rows3,
} from 'lucide-react';
import { DaySummary, QuarterPrediction } from '../types';

interface DailySummaryBannerProps {
  summary: DaySummary | null;
  predictions?: QuarterPrediction[];
  dateStr: string;
  priceArea: 'DK1' | 'DK2';
  title?: string;
  subtitle?: React.ReactNode;
}

export function DailySummaryBanner({
  summary,
  predictions = [],
  dateStr,
  priceArea,
  title = 'Full Settlement Day Overview',
  subtitle,
}: DailySummaryBannerProps) {
  // Default is 'compact'
  const [layoutMode, setLayoutMode] = useState<'full' | 'compact'>(() => {
    try {
      const saved = localStorage.getItem('portal_summary_layout_v2');
      return saved === 'full' ? 'full' : 'compact';
    } catch {
      return 'compact';
    }
  });

  if (!summary) return null;

  const handleToggle = (mode: 'full' | 'compact') => {
    setLayoutMode(mode);
    try {
      localStorage.setItem('portal_summary_layout_v2', mode);
    } catch {
      // ignore
    }
  };

  const formatCurrency = (val: number) => `€ ${val.toFixed(2)}`;
  const formatSpread = (val: number) => {
    const formatted = formatCurrency(Math.abs(val));
    return val >= 0 ? `+${formatted}` : `-${formatted}`;
  };

  const formatPnl = (val: number) => {
    const abs = Math.abs(val).toFixed(2);
    return val >= 0 ? `+€ ${abs}` : `-€ ${abs}`;
  };

  // Performance calculations
  const realizedPnl = summary.realizedPnl ?? 0;
  const settledCount = summary.settledCount ?? 0;
  const totalQuarters = summary.totalQuarters;

  const winCount = predictions.filter(
    (p) => p.net_pnl_eur != null && Number(p.net_pnl_eur) > 0
  ).length;
  const winRate = settledCount > 0 ? (winCount / settledCount) * 100 : 0;

  const activeCount = predictions.filter((p) => {
    const dec = (p.decision || '').toUpperCase();
    return dec.includes('BUY') || dec.includes('SELL');
  }).length;

  const pnlColor =
    realizedPnl > 0
      ? 'text-emerald-400'
      : realizedPnl < 0
      ? 'text-rose-400'
      : 'text-slate-400';

  const pnlBgColor =
    realizedPnl > 0
      ? 'border-emerald-500/30 bg-emerald-950/30'
      : realizedPnl < 0
      ? 'border-rose-500/30 bg-rose-950/30'
      : 'border-slate-750 bg-slate-900/70';

  const winRateColor =
    winRate >= 60
      ? 'text-emerald-400'
      : winRate >= 45
      ? 'text-amber-400'
      : settledCount === 0
      ? 'text-slate-500'
      : 'text-rose-400';

  const progressPct = totalQuarters > 0 ? (settledCount / totalQuarters) * 100 : 0;

  return (
    <div className="w-full bg-slate-800 p-5 rounded-xl border border-slate-700 shadow-xl flex flex-col gap-4">
      {/* ── HEADER ROW: Title, Status, Signal Pills & View Toggle ── */}
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
            <p className="text-xs text-slate-400 mt-0.5">
              {subtitle || (
                <>
                  Delivery Date: <span className="font-mono text-slate-200">{dateStr}</span> ({summary.totalQuarters} 15-minute QH settlement intervals)
                </>
              )}
            </p>
          </div>
        </div>

        {/* Right side: Signals and Toggle */}
        <div className="flex items-center gap-2.5 flex-wrap">
          {/* Signal Counts Pills */}
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

          {/* Full / Compact Layout Toggle */}
          <div className="flex bg-slate-900 p-0.5 rounded-lg border border-slate-700 shadow-sm ml-1">
            <button
              type="button"
              onClick={() => handleToggle('full')}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer ${
                layoutMode === 'full'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Full detailed view with all KPI cards and market averages"
            >
              <LayoutGrid className="w-3.5 h-3.5" />
              <span>Full</span>
            </button>

            <button
              type="button"
              onClick={() => handleToggle('compact')}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer ${
                layoutMode === 'compact'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Compact horizontal layout to maximize table visibility"
            >
              <Rows3 className="w-3.5 h-3.5" />
              <span>Compact</span>
            </button>
          </div>
        </div>
      </div>

      {/* ── FULL DETAILED VIEW (DEFAULT) ── */}
      {layoutMode === 'full' && (
        <div className="flex flex-col gap-4">
          {/* TIER 1: Commercial Trading KPIs */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {/* Card 1: Net Realized PnL */}
            <div className={`p-4 rounded-xl border shadow-md ${pnlBgColor}`}>
              <div className="flex items-center gap-2 text-[11px] font-medium text-slate-400 uppercase tracking-wider">
                <DollarSign className="w-4 h-4 text-slate-500" />
                Net Realized PnL
              </div>
              <div className={`text-2xl font-bold font-mono mt-2 ${pnlColor}`}>
                {settledCount > 0 ? formatPnl(realizedPnl) : '—'}
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                {settledCount > 0
                  ? `From ${settledCount} settled quarter${settledCount !== 1 ? 's' : ''}`
                  : 'No settled quarters yet'}
              </div>
            </div>

            {/* Card 2: Win Rate */}
            <div className="p-4 rounded-xl border border-slate-750 bg-slate-900/70 shadow-md">
              <div className="flex items-center gap-2 text-[11px] font-medium text-slate-400 uppercase tracking-wider">
                <Target className="w-4 h-4 text-slate-500" />
                Win Rate
              </div>
              <div className={`text-2xl font-bold font-mono mt-2 ${winRateColor}`}>
                {settledCount > 0 ? `${winRate.toFixed(1)}%` : '—'}
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                {settledCount > 0
                  ? `${winCount} wins / ${settledCount} settled`
                  : 'Awaiting settlement'}
              </div>
            </div>

            {/* Card 3: Active Trades */}
            <div className="p-4 rounded-xl border border-slate-750 bg-slate-900/70 shadow-md">
              <div className="flex items-center gap-2 text-[11px] font-medium text-slate-400 uppercase tracking-wider">
                <Activity className="w-4 h-4 text-slate-500" />
                Active Trades
              </div>
              <div className="text-2xl font-bold font-mono mt-2 text-indigo-300">
                {activeCount}
                <span className="text-sm font-normal text-slate-500 ml-1">/ {totalQuarters}</span>
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                {summary.buyCount} BUY · {summary.sellCount} SELL · {summary.holdCount} HOLD
              </div>
            </div>

            {/* Card 4: Settlement Progress */}
            <div className="p-4 rounded-xl border border-slate-750 bg-slate-900/70 shadow-md">
              <div className="flex items-center gap-2 text-[11px] font-medium text-slate-400 uppercase tracking-wider">
                <BarChart3 className="w-4 h-4 text-slate-500" />
                Settlement Progress
              </div>
              <div className="text-2xl font-bold font-mono mt-2 text-slate-100">
                {settledCount}
                <span className="text-sm font-normal text-slate-500 ml-1">/ {totalQuarters}</span>
              </div>
              <div className="mt-2 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-full transition-all duration-500"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                {progressPct.toFixed(0)}% complete
              </div>
            </div>
          </div>

          {/* Section Divider */}
          <div className="flex items-center gap-3">
            <div className="h-px bg-slate-700/80 flex-1" />
            <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500 font-mono">
              Market & Price Averages
            </span>
            <div className="h-px bg-slate-700/80 flex-1" />
          </div>

          {/* TIER 2: Market & Pricing Averages */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-slate-900/70 p-3 rounded-lg border border-slate-750">
              <div className="text-[11px] font-medium text-slate-400 flex items-center gap-1">
                <BarChart3 className="w-3.5 h-3.5 text-slate-500" />
                Avg Spot Price
              </div>
              <div className="text-lg font-bold font-mono text-slate-100 mt-1">
                {formatCurrency(summary.avgSpot)}
              </div>
            </div>

            <div className="bg-slate-900/70 p-3 rounded-lg border border-slate-750">
              <div className="text-[11px] font-medium text-slate-400 flex items-center gap-1">
                <BarChart3 className="w-3.5 h-3.5 text-indigo-400" />
                Avg Pred Imbalance
              </div>
              <div className="text-lg font-bold font-mono text-indigo-300 mt-1">
                {formatCurrency(summary.avgImbalance)}
              </div>
            </div>

            <div className="bg-slate-900/70 p-3 rounded-lg border border-slate-750">
              <div className="text-[11px] font-medium text-slate-400 flex items-center gap-1">
                <BarChart3 className="w-3.5 h-3.5 text-emerald-400" />
                Avg Pred Spread
              </div>
              <div className={`text-lg font-bold font-mono mt-1 ${summary.avgSpread >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {formatSpread(summary.avgSpread)}
              </div>
            </div>

            <div className="bg-slate-900/70 p-3 rounded-lg border border-slate-750">
              <div className="text-[11px] font-medium text-slate-400 flex items-center gap-1">
                <Layers className="w-3.5 h-3.5 text-slate-500" />
                Total Trade Vol
              </div>
              <div className="text-lg font-bold font-mono text-slate-100 mt-1">
                {summary.totalVolume.toFixed(1)} <span className="text-xs font-normal text-slate-400">MWh</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── COMPACT VIEW (ULTRA-SLIM) ── */}
      {layoutMode === 'compact' && (
        <div className="bg-slate-900/80 p-3.5 rounded-xl border border-slate-750 flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-4">
          {/* Left: Net Realized PnL & Win Rate Hero */}
          <div className="flex items-center gap-6 flex-wrap">
            <div className="flex flex-col">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Net Realized PnL</span>
              <span className={`text-xl font-bold font-mono mt-0.5 ${pnlColor}`}>
                {settledCount > 0 ? formatPnl(realizedPnl) : '—'}
              </span>
            </div>

            <div className="h-8 w-px bg-slate-750 hidden sm:block" />

            <div className="flex flex-col">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Win Rate</span>
              <span className={`text-xl font-bold font-mono mt-0.5 ${winRateColor}`}>
                {settledCount > 0 ? `${winRate.toFixed(1)}%` : '—'}
              </span>
            </div>

            <div className="h-8 w-px bg-slate-750 hidden sm:block" />

            <div className="flex flex-col">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Settled Quarters</span>
              <div className="flex items-center gap-2 mt-0.5">
                <span className="text-sm font-bold font-mono text-slate-200">{settledCount} / {totalQuarters}</span>
                <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-400 rounded-full" style={{ width: `${progressPct}%` }} />
                </div>
              </div>
            </div>

            <div className="h-8 w-px bg-slate-750 hidden sm:block" />

            <div className="flex flex-col">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Active Trades</span>
              <span className="text-sm font-bold font-mono text-indigo-300 mt-0.5">
                {activeCount} / {totalQuarters}
              </span>
            </div>
          </div>

          {/* Right: Quick Market Averages in compact badge pills */}
          <div className="flex items-center gap-2 flex-wrap border-t lg:border-t-0 border-slate-800 pt-2 lg:pt-0">
            <div className="px-2.5 py-1.5 rounded-lg bg-slate-850 border border-slate-750 text-xs font-mono">
              <span className="text-slate-400 mr-1.5">Avg Spot:</span>
              <strong className="text-slate-100">{formatCurrency(summary.avgSpot)}</strong>
            </div>

            <div className="px-2.5 py-1.5 rounded-lg bg-slate-850 border border-slate-750 text-xs font-mono">
              <span className="text-slate-400 mr-1.5">Avg Pred:</span>
              <strong className="text-indigo-300">{formatCurrency(summary.avgImbalance)}</strong>
            </div>

            <div className="px-2.5 py-1.5 rounded-lg bg-slate-850 border border-slate-750 text-xs font-mono">
              <span className="text-slate-400 mr-1.5">Avg Spread:</span>
              <strong className={summary.avgSpread >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                {formatSpread(summary.avgSpread)}
              </strong>
            </div>

            <div className="px-2.5 py-1.5 rounded-lg bg-slate-850 border border-slate-750 text-xs font-mono">
              <span className="text-slate-400 mr-1.5">Total Vol:</span>
              <strong className="text-slate-200">{summary.totalVolume.toFixed(1)} MWh</strong>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

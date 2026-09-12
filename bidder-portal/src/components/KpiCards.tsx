import { DollarSign, Target, Activity, BarChart3 } from 'lucide-react';
import { QuarterPrediction, DaySummary } from '../types';

interface KpiCardsProps {
  predictions: QuarterPrediction[];
  daySummary: DaySummary | null;
}

export function KpiCards({ predictions, daySummary }: KpiCardsProps) {
  if (!daySummary) return null;

  const realizedPnl = daySummary.realizedPnl ?? 0;
  const settledCount = daySummary.settledCount ?? 0;
  const totalQuarters = daySummary.totalQuarters;

  // Win rate: quarters where net_pnl_eur > 0 among settled quarters
  const winCount = predictions.filter(
    (p) => p.net_pnl_eur != null && Number(p.net_pnl_eur) > 0
  ).length;
  const winRate = settledCount > 0 ? (winCount / settledCount) * 100 : 0;

  // Active trades: BUY or SELL (not HOLD)
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
      : 'border-slate-700 bg-slate-900/70';

  const winRateColor =
    winRate >= 60
      ? 'text-emerald-400'
      : winRate >= 45
      ? 'text-amber-400'
      : settledCount === 0
      ? 'text-slate-500'
      : 'text-rose-400';

  const progressPct = totalQuarters > 0 ? (settledCount / totalQuarters) * 100 : 0;

  const formatPnl = (val: number) => {
    const abs = Math.abs(val).toFixed(2);
    return val >= 0 ? `+€ ${abs}` : `-€ ${abs}`;
  };

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {/* Card 1: Net Realized PnL */}
      <div
        className={`p-4 rounded-xl border shadow-lg ${pnlBgColor}`}
      >
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
      <div className="p-4 rounded-xl border border-slate-700 bg-slate-900/70 shadow-lg">
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
      <div className="p-4 rounded-xl border border-slate-700 bg-slate-900/70 shadow-lg">
        <div className="flex items-center gap-2 text-[11px] font-medium text-slate-400 uppercase tracking-wider">
          <Activity className="w-4 h-4 text-slate-500" />
          Active Trades
        </div>
        <div className="text-2xl font-bold font-mono mt-2 text-indigo-300">
          {activeCount}
          <span className="text-sm font-normal text-slate-500 ml-1">/ {totalQuarters}</span>
        </div>
        <div className="text-[11px] text-slate-500 mt-1">
          {daySummary.buyCount} BUY · {daySummary.sellCount} SELL · {daySummary.holdCount} HOLD
        </div>
      </div>

      {/* Card 4: Settlement Progress */}
      <div className="p-4 rounded-xl border border-slate-700 bg-slate-900/70 shadow-lg">
        <div className="flex items-center gap-2 text-[11px] font-medium text-slate-400 uppercase tracking-wider">
          <BarChart3 className="w-4 h-4 text-slate-500" />
          Settlement Progress
        </div>
        <div className="text-2xl font-bold font-mono mt-2 text-slate-100">
          {settledCount}
          <span className="text-sm font-normal text-slate-500 ml-1">/ {totalQuarters}</span>
        </div>
        {/* Mini Progress Bar */}
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
  );
}

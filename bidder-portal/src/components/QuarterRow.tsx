import { ChevronRight, ChevronDown, Activity, DollarSign, Layers, ShieldAlert } from 'lucide-react';
import { QuarterPrediction } from '../types';
import { ConfidenceBar } from './ConfidenceBar';
import { DirectionBadge } from './DirectionBadge';
import { StatusBadge } from './StatusBadge';

interface QuarterRowProps {
  prediction: QuarterPrediction;
  isTodayMode?: boolean;
  showDate?: boolean;
  isAuditView?: boolean;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
}

export function QuarterRow({
  prediction,
  isTodayMode = false,
  showDate = false,
  isAuditView = false,
  isExpanded = false,
  onToggleExpand,
}: QuarterRowProps) {
  const formatTime = (quarterIndex: number) => {
    const hour = Math.floor((quarterIndex - 1) / 4).toString().padStart(2, '0');
    const minute = (((quarterIndex - 1) % 4) * 15).toString().padStart(2, '0');
    return `${hour}:${minute}`;
  };

  const formatCurrency = (val: number | null | undefined) => {
    if (val == null || isNaN(val)) return '—';
    return `€ ${Number(val).toFixed(2)}`;
  };

  const formatSpread = (val: number | null | undefined) => {
    if (val == null || isNaN(val)) return '—';
    const num = Number(val);
    const formatted = `€ ${Math.abs(num).toFixed(2)}`;
    return num > 0 ? `+${formatted}` : num < 0 ? `-${formatted}` : `€ 0.00`;
  };

  const isSpreadPositive = prediction.pred_spread_eur > 0;
  const isNetPnlPositive = prediction.net_pnl_eur != null && prediction.net_pnl_eur > 0;
  const isNetPnlNegative = prediction.net_pnl_eur != null && prediction.net_pnl_eur < 0;

  const now = Date.now();
  const deliveryStartTime = new Date(prediction.time_dk).getTime();
  const deliveryEndTime = deliveryStartTime + 15 * 60 * 1000;
  const isCurrent = isTodayMode && now >= deliveryStartTime && now < deliveryEndTime;
  const isPast = isTodayMode && now >= deliveryEndTime;

  return (
    <>
      <tr
        onClick={isAuditView ? onToggleExpand : undefined}
        className={`transition-colors ${isAuditView ? 'cursor-pointer' : ''} ${
          isCurrent
            ? 'bg-indigo-950/40 hover:bg-indigo-900/50 border-l-4 border-l-indigo-500'
            : isExpanded
            ? 'bg-slate-800/90 border-l-4 border-l-indigo-400'
            : isPast
            ? 'opacity-85 hover:bg-slate-750'
            : 'hover:bg-slate-750'
        }`}
      >
        {/* Quarter Identifier */}
        <td className="px-3.5 py-3 font-semibold text-slate-200">
          <div className="flex items-center gap-2 flex-wrap">
            {isAuditView && (
              <span className="text-slate-400 hover:text-indigo-300 transition-colors">
                {isExpanded ? (
                  <ChevronDown className="w-4 h-4 text-indigo-400" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-slate-500" />
                )}
              </span>
            )}
            {showDate && (
              <span className="text-[10px] text-indigo-300 bg-indigo-950/80 px-1.5 py-0.5 rounded border border-indigo-700/60 font-mono font-medium">
                {prediction.delivery_date}
              </span>
            )}
            <span className="font-mono text-sm">{prediction.quarter_label}</span>
            <span className="text-[11px] text-slate-400 bg-slate-900/80 px-1.5 py-0.5 rounded border border-slate-700 font-mono font-normal">
              QH-{prediction.quarter_index.toString().padStart(2, '0')}
            </span>
            {isCurrent && (
              <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 rounded-full animate-pulse">
                Active
              </span>
            )}
          </div>
        </td>

        {/* Time DK */}
        <td className="px-3.5 py-3 text-slate-300 font-mono text-xs">
          {formatTime(prediction.quarter_index)}
        </td>

        {/* Spot Price */}
        <td className="px-3.5 py-3 text-right text-slate-300 font-mono text-xs">
          {formatCurrency(prediction.spot_price_eur)}
        </td>

        {/* Pred Imbalance */}
        <td className="px-3.5 py-3 text-right text-slate-300 font-mono text-xs">
          {formatCurrency(prediction.pred_imbalance_eur)}
        </td>

        {/* Pred Spread */}
        <td
          className={`px-3.5 py-3 text-right font-mono text-xs font-semibold ${
            isSpreadPositive ? 'text-emerald-400' : 'text-rose-400'
          }`}
        >
          {formatSpread(prediction.pred_spread_eur)}
        </td>

        {/* Probabilities */}
        <td className="px-3.5 py-3 text-center">
          <ConfidenceBar pUp={prediction.p_up} pDown={prediction.p_down} />
        </td>

        {/* Decision Badge */}
        <td className="px-3.5 py-3">
          <DirectionBadge decision={prediction.decision} />
        </td>

        {/* Trade Volume */}
        <td className="px-3.5 py-3 text-right text-slate-300 font-medium font-mono text-xs">
          {prediction.volume_mwh.toFixed(1)} MWh
        </td>

        {/* ── AUDIT COLUMNS (Only shown in Audit View /all) ── */}
        {isAuditView && (
          <>
            {/* Settled Imbalance Price */}
            <td className="px-3.5 py-3 text-right font-mono text-xs">
              {prediction.actual_settled_eur != null ? (
                <span className="text-slate-100 font-medium">
                  {formatCurrency(prediction.actual_settled_eur)}
                </span>
              ) : (
                <span className="text-slate-600 italic">—</span>
              )}
            </td>

            {/* Net Realized PnL */}
            <td
              className={`px-3.5 py-3 text-right font-mono text-xs font-bold ${
                isNetPnlPositive
                  ? 'text-emerald-400'
                  : isNetPnlNegative
                  ? 'text-rose-400'
                  : 'text-slate-400'
              }`}
            >
              {prediction.net_pnl_eur != null ? (
                formatSpread(prediction.net_pnl_eur)
              ) : (
                <span className="text-slate-600 italic font-normal">—</span>
              )}
            </td>

            {/* Settlement Status Badge */}
            <td className="px-3.5 py-3 text-center">
              <StatusBadge status={prediction.status} />
            </td>
          </>
        )}
      </tr>

      {/* ── EXPANDABLE DETAIL DRAWER ── */}
      {isAuditView && isExpanded && (
        <tr className="bg-slate-900/95 border-b border-indigo-500/30">
          <td colSpan={11} className="p-4 pl-8">
            <div className="flex flex-col gap-4 text-xs">
              {/* Header with Title & Quick Info */}
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <div className="flex items-center gap-2 font-semibold text-indigo-300">
                  <Activity className="w-4 h-4 text-indigo-400" />
                  <span>Audit Breakdown: {prediction.quarter_label} (QH-{prediction.quarter_index}) — {prediction.model_name}</span>
                </div>
                <div className="text-[11px] text-slate-400 font-mono">
                  Delivery Time: {new Date(prediction.time_dk).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} CEST
                  {prediction.settled_at && ` · Settled at: ${new Date(prediction.settled_at).toLocaleTimeString()}`}
                </div>
              </div>

              {/* Quantiles & Cash Flow Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* 1. Quantiles & Forecast Details */}
                <div className="bg-slate-850 p-3.5 rounded-lg border border-slate-750 flex flex-col gap-2.5">
                  <div className="text-[11px] font-semibold text-slate-300 flex items-center gap-1.5 uppercase tracking-wider">
                    <Layers className="w-3.5 h-3.5 text-indigo-400" />
                    Probabilistic Quantiles (EUR/MWh)
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-center pt-1">
                    <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
                      <div className="text-[10px] text-slate-400">Q10 (Bearish)</div>
                      <div className="text-xs font-mono font-bold text-slate-200 mt-0.5">
                        {formatCurrency(prediction.q10_price_eur)}
                      </div>
                    </div>
                    <div className="bg-slate-900/80 p-2 rounded border border-indigo-900/40">
                      <div className="text-[10px] text-indigo-400">Q50 (Median)</div>
                      <div className="text-xs font-mono font-bold text-indigo-300 mt-0.5">
                        {formatCurrency(prediction.q50_price_eur)}
                      </div>
                    </div>
                    <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
                      <div className="text-[10px] text-slate-400">Q90 (Bullish)</div>
                      <div className="text-xs font-mono font-bold text-slate-200 mt-0.5">
                        {formatCurrency(prediction.q90_price_eur)}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800">
                    <span>Pred Imbalance: <strong className="text-slate-200 font-mono">{formatCurrency(prediction.pred_imbalance_eur)}</strong></span>
                    <span>Spread Risk: <strong className={`font-mono ${isSpreadPositive ? 'text-emerald-400' : 'text-rose-400'}`}>{formatSpread(prediction.pred_spread_eur)}</strong></span>
                  </div>
                </div>

                {/* 2. Settlement & Cash Flow Ledger */}
                <div className="bg-slate-850 p-3.5 rounded-lg border border-slate-750 flex flex-col gap-2.5">
                  <div className="text-[11px] font-semibold text-slate-300 flex items-center gap-1.5 uppercase tracking-wider">
                    <DollarSign className="w-3.5 h-3.5 text-emerald-400" />
                    Commercial Cash Flow & PnL Ledger
                  </div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px] pt-1 font-mono">
                    <div className="flex justify-between text-slate-400">
                      <span>DA Cash Flow:</span>
                      <span className="text-slate-200">{formatSpread(prediction.da_cash_flow_eur)}</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>Settle Cash Flow:</span>
                      <span className="text-slate-200">{formatSpread(prediction.settle_cash_flow_eur)}</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>Gross PnL:</span>
                      <span className={prediction.gross_pnl_eur && prediction.gross_pnl_eur > 0 ? 'text-emerald-400' : 'text-slate-200'}>
                        {formatSpread(prediction.gross_pnl_eur)}
                      </span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>Friction (€0.51/MWh):</span>
                      <span className="text-rose-400">{prediction.fees_eur != null ? `-€ ${prediction.fees_eur.toFixed(2)}` : '—'}</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>Corp Tax (22%):</span>
                      <span className="text-rose-400">{prediction.tax_eur != null ? `-€ ${prediction.tax_eur.toFixed(2)}` : '—'}</span>
                    </div>
                    <div className="flex justify-between font-bold border-t border-slate-750 pt-1">
                      <span className="text-slate-200">Net Realized PnL:</span>
                      <span className={isNetPnlPositive ? 'text-emerald-400' : isNetPnlNegative ? 'text-rose-400' : 'text-slate-300'}>
                        {formatSpread(prediction.net_pnl_eur)}
                      </span>
                    </div>
                  </div>
                  {prediction.status?.toUpperCase() !== 'SETTLED' && (
                    <div className="text-[10px] text-amber-400/80 bg-amber-950/30 px-2 py-1 rounded border border-amber-800/40 flex items-center gap-1.5">
                      <ShieldAlert className="w-3.5 h-3.5 shrink-0 text-amber-400" />
                      <span>Order locked at Gate Closure. Cash flows will be reconciled upon Energinet settlement publishing.</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

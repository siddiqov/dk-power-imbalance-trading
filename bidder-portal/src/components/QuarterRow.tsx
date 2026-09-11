import { QuarterPrediction } from '../types';
import { ConfidenceBar } from './ConfidenceBar';
import { DirectionBadge } from './DirectionBadge';

interface QuarterRowProps {
  prediction: QuarterPrediction;
  isTodayMode?: boolean;
  showDate?: boolean;
}

export function QuarterRow({ prediction, isTodayMode = false, showDate = false }: QuarterRowProps) {
  const formatTime = (quarterIndex: number) => {
    const hour = Math.floor((quarterIndex - 1) / 4).toString().padStart(2, '0');
    const minute = (((quarterIndex - 1) % 4) * 15).toString().padStart(2, '0');
    return `${hour}:${minute}`;
  };

  const formatCurrency = (val: number) => `€ ${val.toFixed(2)}`;
  
  const formatSpread = (val: number) => {
    const formatted = formatCurrency(Math.abs(val));
    return val > 0 ? `+${formatted}` : `-${formatted}`;
  };

  const isSpreadPositive = prediction.pred_spread_eur > 0;

  const now = Date.now();
  const deliveryStartTime = new Date(prediction.time_dk).getTime();
  const deliveryEndTime = deliveryStartTime + 15 * 60 * 1000;
  const isCurrent = isTodayMode && now >= deliveryStartTime && now < deliveryEndTime;
  const isPast = isTodayMode && now >= deliveryEndTime;

  return (
    <tr
      className={`transition-colors ${
        isCurrent
          ? 'bg-indigo-950/40 hover:bg-indigo-900/50 border-l-4 border-l-indigo-500'
          : isPast
          ? 'opacity-75 hover:bg-slate-750'
          : 'hover:bg-slate-750'
      }`}
    >
      <td className="px-4 py-4 font-semibold text-slate-200">
        <div className="flex items-center gap-2 flex-wrap">
          {showDate && (
            <span className="text-[10px] text-indigo-300 bg-indigo-950/80 px-1.5 py-0.5 rounded border border-indigo-700/60 font-mono font-medium">
              {prediction.delivery_date}
            </span>
          )}
          <span>{prediction.quarter_label}</span>
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
      <td className="px-4 py-4 text-slate-300 font-mono">
        {formatTime(prediction.quarter_index)}
      </td>
      <td className="px-4 py-4 text-right text-slate-300 font-mono">
        {formatCurrency(prediction.spot_price_eur)}
      </td>
      <td className="px-4 py-4 text-right text-slate-300 font-mono">
        {formatCurrency(prediction.pred_imbalance_eur)}
      </td>
      <td className={`px-4 py-4 text-right font-mono font-medium ${isSpreadPositive ? 'text-green-400' : 'text-red-400'}`}>
        {formatSpread(prediction.pred_spread_eur)}
      </td>
      <td className="px-4 py-4 text-center">
        <ConfidenceBar pUp={prediction.p_up} pDown={prediction.p_down} />
      </td>
      <td className="px-4 py-4">
        <DirectionBadge decision={prediction.decision} />
      </td>
      <td className="px-4 py-4 text-right text-slate-300 font-medium">
        {prediction.volume_mwh.toFixed(1)} MWh
      </td>
    </tr>
  );
}

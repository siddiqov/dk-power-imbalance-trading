import { QuarterPrediction } from '../types';
import { ConfidenceBar } from './ConfidenceBar';
import { DirectionBadge } from './DirectionBadge';

interface QuarterRowProps {
  prediction: QuarterPrediction;
}

export function QuarterRow({ prediction }: QuarterRowProps) {
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

  return (
    <tr className="hover:bg-slate-750 transition-colors">
      <td className="px-4 py-4 font-semibold text-slate-200">
        <div className="flex items-center gap-2">
          <span>{prediction.quarter_label}</span>
          <span className="text-[11px] text-slate-400 bg-slate-900/80 px-1.5 py-0.5 rounded border border-slate-700 font-mono font-normal">
            QH-{prediction.quarter_index.toString().padStart(2, '0')}
          </span>
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

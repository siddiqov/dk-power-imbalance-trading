import { useState, useMemo } from 'react';
import { Info, HelpCircle } from 'lucide-react';
import { QuarterPrediction } from '../types';
import { QuarterRow } from './QuarterRow';

interface QuarterTableProps {
  predictions: QuarterPrediction[];
  loading: boolean;
  error: string | null;
  isTodayMode?: boolean;
}

interface ColumnDef {
  label: string;
  tooltip: string;
  align: 'left' | 'center' | 'right';
}

const COLUMNS: ColumnDef[] = [
  {
    label: 'Quarter',
    tooltip: '15-minute delivery settlement interval (Q1 to Q96) matching Nord Pool QH contracts.',
    align: 'left',
  },
  {
    label: 'Time (DK)',
    tooltip: 'Start time of the 15-minute physical delivery period in local Danish time (CET/CEST).',
    align: 'left',
  },
  {
    label: 'Spot (€)',
    tooltip: 'Day-Ahead Nord Pool auction clearing price (€/MWh) for this settlement quarter.',
    align: 'right',
  },
  {
    label: 'Pred Imb (€)',
    tooltip: 'Transformer-TFT model forecast for the final Energinet imbalance settlement price (€/MWh).',
    align: 'right',
  },
  {
    label: 'Pred Spread (€)',
    tooltip: 'Pred Spread = Pred Imbalance − Spot Price (€/MWh). Positive spread suggests upward price risk.',
    align: 'right',
  },
  {
    label: 'P(Up) / P(Down)',
    tooltip: 'Directional probabilities: P(Up) is probability imbalance > spot; P(Down) is probability imbalance < spot.',
    align: 'center',
  },
  {
    label: 'Decision',
    tooltip: 'Recommended trading signal: BUY Spot (Long), SELL Spot (Short), or HOLD (including Spike Shield protection).',
    align: 'left',
  },
  {
    label: 'Volume',
    tooltip: 'Recommended trade size in MWh, sized dynamically according to signal confidence and market volatility.',
    align: 'right',
  },
];

export function QuarterTable({ predictions, loading, error, isTodayMode = false }: QuarterTableProps) {
  const [activeCol, setActiveCol] = useState<ColumnDef | null>(null);
  const [signalFilter, setSignalFilter] = useState<'ALL' | 'BUY' | 'SELL' | 'HOLD'>('ALL');

  const hasMultipleDates = useMemo(() => {
    return new Set(predictions.map((p) => p.delivery_date)).size > 1;
  }, [predictions]);

  const filteredPredictions = predictions.filter((p) => {
    if (signalFilter === 'ALL') return true;
    const dec = (p.decision || '').toUpperCase();
    if (signalFilter === 'BUY') return dec.includes('BUY');
    if (signalFilter === 'SELL') return dec.includes('SELL');
    if (signalFilter === 'HOLD') return !dec.includes('BUY') && !dec.includes('SELL');
    return true;
  });

  if (error) {
    return <div className="p-4 bg-red-900/50 text-red-200 border border-red-800 rounded-lg">Error: {error}</div>;
  }

  if (loading && predictions.length === 0) {
    return <div className="p-8 text-center text-slate-400">Loading predictions from Supabase...</div>;
  }

  if (!loading && predictions.length === 0) {
    return (
      <div className="p-8 text-center text-slate-400 bg-slate-800 rounded-xl border border-slate-700">
        {isTodayMode ? 'No quarter predictions found for today in Supabase.' : 'No upcoming quarters available.'}
      </div>
    );
  }

  return (
    <div className="w-full bg-slate-800 rounded-xl border border-slate-700 shadow-xl overflow-hidden flex flex-col">
      {/* Top Controls: Filter Pills for All 96 Quarters Mode + Definition Bar */}
      {isTodayMode && (
        <div className="px-5 py-3 bg-slate-900/60 border-b border-slate-700/80 flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400 font-medium mr-1">Filter Signals:</span>
            {(['ALL', 'BUY', 'SELL', 'HOLD'] as const).map((filter) => (
              <button
                key={filter}
                type="button"
                onClick={() => setSignalFilter(filter)}
                className={`px-2.5 py-1 rounded font-medium text-xs transition-colors cursor-pointer ${
                  signalFilter === filter
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'bg-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-750'
                }`}
              >
                {filter}
                {filter !== 'ALL' && (
                  <span className="ml-1 opacity-70">
                    (
                    {
                      predictions.filter((p) => {
                        const dec = (p.decision || '').toUpperCase();
                        if (filter === 'BUY') return dec.includes('BUY');
                        if (filter === 'SELL') return dec.includes('SELL');
                        return !dec.includes('BUY') && !dec.includes('SELL');
                      }).length
                    }
                    )
                  </span>
                )}
              </button>
            ))}
          </div>

          <div className="text-slate-400 text-xs font-mono">
            Showing {filteredPredictions.length} of {predictions.length} quarters
          </div>
        </div>
      )}

      {/* Interactive Definition Banner */}
      <div className="px-5 py-2.5 bg-slate-900/90 border-b border-slate-700/80 flex items-center gap-2.5 text-xs transition-colors duration-200">
        {activeCol ? (
          <div className="flex items-center gap-2 text-slate-200 animate-fadeIn">
            <span className="font-semibold text-indigo-400 flex items-center gap-1">
              <Info className="w-3.5 h-3.5 text-indigo-400" />
              {activeCol.label}:
            </span>
            <span className="text-slate-300 font-normal">{activeCol.tooltip}</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-slate-400">
            <HelpCircle className="w-3.5 h-3.5 text-slate-500" />
            <span>Hover over any column header to view its trading definition and formula.</span>
          </div>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse whitespace-nowrap">
          <thead>
            <tr className="bg-slate-900 text-slate-400 uppercase text-xs tracking-wider">
              {COLUMNS.map((col) => {
                const alignClass = {
                  left: 'text-left',
                  center: 'text-center',
                  right: 'text-right',
                }[col.align];

                const flexAlign = {
                  left: 'justify-start',
                  center: 'justify-center',
                  right: 'justify-end',
                }[col.align];

                const isHovered = activeCol?.label === col.label;

                return (
                  <th
                    key={col.label}
                    title={col.tooltip}
                    onMouseEnter={() => setActiveCol(col)}
                    onMouseLeave={() => setActiveCol(null)}
                    className={`px-4 py-3.5 font-medium border-b border-slate-700 cursor-pointer select-none transition-colors ${alignClass} ${
                      isHovered ? 'bg-slate-800/80 text-indigo-300' : 'hover:bg-slate-850 hover:text-slate-200'
                    }`}
                  >
                    <div className={`inline-flex items-center gap-1.5 ${flexAlign}`}>
                      <span>{col.label}</span>
                      <Info
                        className={`w-3.5 h-3.5 transition-colors ${
                          isHovered ? 'text-indigo-400' : 'text-slate-500 opacity-60'
                        }`}
                      />
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700">
            {filteredPredictions.map((prediction) => (
              <QuarterRow
                key={`${prediction.delivery_date}_${prediction.quarter_index}_${prediction.time_dk}`}
                prediction={prediction}
                isTodayMode={isTodayMode}
                showDate={hasMultipleDates}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

import { useState, useMemo } from 'react';
import { Info, HelpCircle, Copy, Check, FileSpreadsheet } from 'lucide-react';
import { QuarterPrediction } from '../types';
import { QuarterRow } from './QuarterRow';

interface QuarterTableProps {
  predictions: QuarterPrediction[];
  loading: boolean;
  error: string | null;
  isTodayMode?: boolean;
}

interface ColumnDef {
  key: string;
  label: string;
  tooltip: string;
  align: 'left' | 'center' | 'right';
}

const COLUMNS: ColumnDef[] = [
  {
    key: 'quarter',
    label: 'Quarter',
    tooltip: '15-minute delivery settlement interval (Q1 to Q96) matching Nord Pool QH contracts.',
    align: 'left',
  },
  {
    key: 'time',
    label: 'Time (DK)',
    tooltip: 'Start time of the 15-minute physical delivery period in local Danish time (CET/CEST).',
    align: 'left',
  },
  {
    key: 'spot',
    label: 'Spot (€)',
    tooltip: 'Day-Ahead Nord Pool auction clearing price (€/MWh) for this settlement quarter.',
    align: 'right',
  },
  {
    key: 'pred_imbalance',
    label: 'Pred Imb (€)',
    tooltip: 'Transformer-TFT model forecast for the final Energinet imbalance settlement price (€/MWh).',
    align: 'right',
  },
  {
    key: 'pred_spread',
    label: 'Pred Spread (€)',
    tooltip: 'Pred Spread = Pred Imbalance − Spot Price (€/MWh). Positive spread suggests upward price risk.',
    align: 'right',
  },
  {
    key: 'probabilities',
    label: 'P(Up) / P(Down)',
    tooltip: 'Directional probabilities: P(Up) is probability imbalance > spot; P(Down) is probability imbalance < spot.',
    align: 'center',
  },
  {
    key: 'decision',
    label: 'Decision',
    tooltip: 'Recommended trading signal: BUY Spot (Long), SELL Spot (Short), or HOLD (including Spike Shield protection).',
    align: 'left',
  },
  {
    key: 'volume',
    label: 'Volume',
    tooltip: 'Recommended trade size in MWh, sized dynamically according to signal confidence and market volatility.',
    align: 'right',
  },
];

const formatTime = (quarterIndex: number) => {
  const hour = Math.floor((quarterIndex - 1) / 4).toString().padStart(2, '0');
  const minute = (((quarterIndex - 1) % 4) * 15).toString().padStart(2, '0');
  return `${hour}:${minute}`;
};

function getColumnValue(prediction: QuarterPrediction, key: string, hasMultipleDates: boolean): string {
  switch (key) {
    case 'quarter':
      return hasMultipleDates ? `${prediction.delivery_date} ${prediction.quarter_label}` : prediction.quarter_label;
    case 'time':
      return formatTime(prediction.quarter_index);
    case 'spot':
      return prediction.spot_price_eur != null ? prediction.spot_price_eur.toFixed(2) : '';
    case 'pred_imbalance':
      return prediction.pred_imbalance_eur != null ? prediction.pred_imbalance_eur.toFixed(2) : '';
    case 'pred_spread':
      return prediction.pred_spread_eur != null ? prediction.pred_spread_eur.toFixed(2) : '';
    case 'probabilities':
      return `${((prediction.p_up ?? 0) * 100).toFixed(1)}% / ${((prediction.p_down ?? 0) * 100).toFixed(1)}%`;
    case 'decision':
      return prediction.decision || '';
    case 'volume':
      return prediction.volume_mwh != null ? prediction.volume_mwh.toFixed(1) : '';
    default:
      return '';
  }
}

const copyToClipboard = async (text: string) => {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fallback
  }
  const textArea = document.createElement('textarea');
  textArea.value = text;
  textArea.style.position = 'fixed';
  textArea.style.left = '-999999px';
  textArea.style.top = '-999999px';
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();
  const ok = document.execCommand('copy');
  textArea.remove();
  return ok;
};

export function QuarterTable({ predictions, loading, error, isTodayMode = false }: QuarterTableProps) {
  const [activeCol, setActiveCol] = useState<ColumnDef | null>(null);
  const [signalFilter, setSignalFilter] = useState<'ALL' | 'BUY' | 'SELL' | 'HOLD'>('ALL');
  const [copiedColKey, setCopiedColKey] = useState<string | null>(null);
  const [copyFeedback, setCopyFeedback] = useState<string | null>(null);
  const [includeHeader, setIncludeHeader] = useState<boolean>(true);

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

  const handleCopyColumn = async (col: ColumnDef, e?: React.MouseEvent) => {
    const values = filteredPredictions.map((p) => getColumnValue(p, col.key, hasMultipleDates));
    const withHeader = includeHeader || (e && e.shiftKey);
    const text = withHeader ? [col.label, ...values].join('\n') : values.join('\n');

    await copyToClipboard(text);
    setCopiedColKey(col.key);
    setCopyFeedback(
      `Copied ${values.length} values for "${col.label}"${withHeader ? ' (with header)' : ''} to clipboard — ready to paste in Excel`
    );

    setTimeout(() => {
      setCopiedColKey((curr) => (curr === col.key ? null : curr));
    }, 2500);

    setTimeout(() => {
      setCopyFeedback((curr) => (curr?.includes(col.label) ? null : curr));
    }, 4500);
  };

  const handleCopyEntireTable = async () => {
    const headers = COLUMNS.map((c) => c.label);
    const rows = filteredPredictions.map((p) =>
      COLUMNS.map((col) => getColumnValue(p, col.key, hasMultipleDates)).join('\t')
    );
    const tsv = [headers.join('\t'), ...rows].join('\n');

    await copyToClipboard(tsv);
    setCopiedColKey('__ALL__');
    setCopyFeedback(`Copied entire table (${rows.length} rows × ${headers.length} columns) in TSV format for Excel!`);

    setTimeout(() => {
      setCopiedColKey((curr) => (curr === '__ALL__' ? null : curr));
    }, 2500);

    setTimeout(() => {
      setCopyFeedback((curr) => (curr?.includes('entire table') ? null : curr));
    }, 4500);
  };

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

      {/* Interactive Definition & Excel Copy Toolbar */}
      <div className="px-5 py-2.5 bg-slate-900/90 border-b border-slate-700/80 flex flex-wrap items-center justify-between gap-3 text-xs transition-colors duration-200">
        <div className="flex items-center gap-2.5 min-w-0 flex-1">
          {copyFeedback ? (
            <div className="flex items-center gap-2 text-emerald-300 font-medium animate-fadeIn">
              <Check className="w-4 h-4 text-emerald-400 shrink-0" />
              <span className="truncate">{copyFeedback}</span>
            </div>
          ) : activeCol ? (
            <div className="flex items-center gap-2 text-slate-200 animate-fadeIn truncate">
              <span className="font-semibold text-indigo-400 flex items-center gap-1 shrink-0">
                <Info className="w-3.5 h-3.5 text-indigo-400" />
                {activeCol.label}:
              </span>
              <span className="text-slate-300 font-normal truncate">{activeCol.tooltip}</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-slate-400">
              <HelpCircle className="w-3.5 h-3.5 text-slate-500 shrink-0" />
              <span className="truncate">Click the copy icon on any column header to copy its values for Excel.</span>
            </div>
          )}
        </div>

        {/* Excel Copy Actions */}
        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={() => setIncludeHeader(!includeHeader)}
            className={`px-2 py-1 rounded text-[11px] font-mono border transition-colors cursor-pointer ${
              includeHeader
                ? 'bg-indigo-950/80 border-indigo-500 text-indigo-300'
                : 'bg-slate-800/80 border-slate-700 text-slate-400 hover:text-slate-300'
            }`}
            title="Toggle whether the column label is copied as the first row (useful when creating new Excel columns)"
          >
            Header: {includeHeader ? 'Included' : 'Values only'}
          </button>

          <button
            type="button"
            onClick={handleCopyEntireTable}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs border transition-colors cursor-pointer ${
              copiedColKey === '__ALL__'
                ? 'bg-emerald-900/50 border-emerald-500 text-emerald-200'
                : 'bg-slate-800 hover:bg-slate-750 border-slate-700 text-slate-300 hover:text-white'
            }`}
            title="Copy all columns as TSV to paste the entire table into Excel"
          >
            {copiedColKey === '__ALL__' ? (
              <Check className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400" />
            )}
            <span>Copy Entire Table</span>
          </button>
        </div>
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

                const isHovered = activeCol?.key === col.key;
                const isCopied = copiedColKey === col.key;

                return (
                  <th
                    key={col.key}
                    title={col.tooltip}
                    onMouseEnter={() => setActiveCol(col)}
                    onMouseLeave={() => setActiveCol(null)}
                    className={`px-3 py-3.5 font-medium border-b border-slate-700 select-none transition-colors group ${alignClass} ${
                      isHovered ? 'bg-slate-800/80 text-indigo-300' : 'hover:bg-slate-850 hover:text-slate-200'
                    }`}
                  >
                    <div className={`inline-flex items-center gap-1.5 ${flexAlign}`}>
                      <span>{col.label}</span>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleCopyColumn(col, e);
                        }}
                        title={`Copy ${col.label} column to clipboard (Excel format)`}
                        className={`p-1 rounded transition-all cursor-pointer ${
                          isCopied
                            ? 'bg-emerald-500/20 text-emerald-400 opacity-100 scale-110'
                            : 'text-slate-400 hover:text-indigo-300 hover:bg-slate-700/70 opacity-60 group-hover:opacity-100'
                        }`}
                        aria-label={`Copy ${col.label} column`}
                      >
                        {isCopied ? (
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                      </button>
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

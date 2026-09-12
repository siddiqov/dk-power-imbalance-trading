import { useState, useMemo } from 'react';
import { Info, HelpCircle, Copy, Check, FileSpreadsheet, Clock, ChevronDown, ChevronUp } from 'lucide-react';
import { QuarterPrediction } from '../types';
import { QuarterRow } from './QuarterRow';

interface QuarterTableProps {
  predictions: QuarterPrediction[];
  loading: boolean;
  error: string | null;
  isTodayMode?: boolean;
  isAuditView?: boolean;
}

interface ColumnDef {
  key: string;
  label: string;
  tooltip: string;
  align: 'left' | 'center' | 'right';
}

const BASE_COLUMNS: ColumnDef[] = [
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

const AUDIT_EXTRA_COLUMNS: ColumnDef[] = [
  {
    key: 'settled',
    label: 'Settled (€)',
    tooltip: 'Actual final Energinet Imbalance settlement clearing price (€/MWh) reconciled from Energi Data Service.',
    align: 'right',
  },
  {
    key: 'net_pnl',
    label: 'Net PnL (€)',
    tooltip: 'Realized net profit/loss after Nord Pool fees, TSO balancing fees, slippage (€0.51/MWh), and Danish corp tax (22%).',
    align: 'right',
  },
  {
    key: 'status',
    label: 'Status',
    tooltip: 'Settlement lifecycle: Settled (reconciled against official TSO data) or Pending (future/awaiting settlement).',
    align: 'center',
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
    case 'settled':
      return prediction.actual_settled_eur != null ? prediction.actual_settled_eur.toFixed(2) : '';
    case 'net_pnl':
      return prediction.net_pnl_eur != null ? prediction.net_pnl_eur.toFixed(2) : '';
    case 'status':
      return prediction.status || 'PENDING';
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

export function QuarterTable({
  predictions,
  loading,
  error,
  isTodayMode = false,
  isAuditView = false,
}: QuarterTableProps) {
  const [activeCol, setActiveCol] = useState<ColumnDef | null>(null);
  const [signalFilter, setSignalFilter] = useState<'ALL' | 'BUY' | 'SELL' | 'HOLD'>('ALL');
  const [copiedColKey, setCopiedColKey] = useState<string | null>(null);
  const [copyFeedback, setCopyFeedback] = useState<string | null>(null);
  const [includeHeader, setIncludeHeader] = useState<boolean>(true);
  const [expandedQuarterKey, setExpandedQuarterKey] = useState<string | null>(null);

  const activeColumns = useMemo(() => {
    return isAuditView ? [...BASE_COLUMNS, ...AUDIT_EXTRA_COLUMNS] : BASE_COLUMNS;
  }, [isAuditView]);

  const hasMultipleDates = useMemo(() => {
    return new Set(predictions.map((p) => p.delivery_date)).size > 1;
  }, [predictions]);

  const filteredPredictions = useMemo(() => {
    return predictions.filter((p) => {
      if (signalFilter === 'ALL') return true;
      const dec = (p.decision || '').toUpperCase();
      if (signalFilter === 'BUY') return dec.includes('BUY');
      if (signalFilter === 'SELL') return dec.includes('SELL');
      if (signalFilter === 'HOLD') return !dec.includes('BUY') && !dec.includes('SELL');
      return true;
    });
  }, [predictions, signalFilter]);

  // Index of first pending/upcoming quarter to insert the NOW separator
  const firstUpcomingIndex = useMemo(() => {
    if (!isTodayMode && !isAuditView) return -1;
    const now = Date.now();
    return filteredPredictions.findIndex((p) => {
      const deliveryEndTime = new Date(p.time_dk).getTime() + 15 * 60 * 1000;
      return now < deliveryEndTime && p.status?.toUpperCase() !== 'SETTLED';
    });
  }, [filteredPredictions, isTodayMode, isAuditView]);

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
    const headers = activeColumns.map((c) => c.label);
    const rows = filteredPredictions.map((p) =>
      activeColumns.map((col) => getColumnValue(p, col.key, hasMultipleDates)).join('\t')
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

  const toggleRowExpansion = (key: string) => {
    setExpandedQuarterKey((curr) => (curr === key ? null : key));
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
      {/* Top Controls: Filter Pills for All 96 Quarters Mode */}
      {isTodayMode && (
        <div className="px-5 py-3 bg-slate-900/60 border-b border-slate-700/80 flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-1.5 flex-wrap">
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

          <div className="flex items-center gap-3">
            {isAuditView && (
              <button
                type="button"
                onClick={() => {
                  if (expandedQuarterKey) {
                    setExpandedQuarterKey(null);
                  } else if (filteredPredictions.length > 0) {
                    const first = filteredPredictions[0];
                    setExpandedQuarterKey(`${first.delivery_date}_${first.quarter_index}`);
                  }
                }}
                className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors flex items-center gap-1 cursor-pointer font-medium"
              >
                {expandedQuarterKey ? (
                  <>
                    <ChevronUp className="w-3.5 h-3.5" />
                    <span>Collapse Open Details</span>
                  </>
                ) : (
                  <>
                    <ChevronDown className="w-3.5 h-3.5" />
                    <span>Click Row for Details</span>
                  </>
                )}
              </button>
            )}
            <div className="text-slate-400 text-xs font-mono">
              Showing {filteredPredictions.length} of {predictions.length} quarters
            </div>
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
              <span className="truncate">
                {isAuditView
                  ? 'Click any quarter row to inspect quantile forecast and settlement cash flows.'
                  : 'Click the copy icon on any column header to copy its values for Excel.'}
              </span>
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
            title="Toggle whether the column label is copied as the first row"
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

      {/* Scrollable Container with Sticky Table Header */}
      <div className={`overflow-x-auto ${isAuditView ? 'max-h-[75vh] overflow-y-auto' : ''}`}>
        <table className="w-full text-left border-collapse whitespace-nowrap">
          <thead className="sticky top-0 z-20 bg-slate-900 shadow-md">
            <tr className="bg-slate-900 text-slate-400 uppercase text-xs tracking-wider border-b border-slate-700">
              {activeColumns.map((col) => {
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
                    className={`px-3 py-3 font-medium select-none transition-colors group ${alignClass} ${
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
            {filteredPredictions.map((prediction, idx) => {
              const rowKey = `${prediction.delivery_date}_${prediction.quarter_index}`;
              const isExpanded = expandedQuarterKey === rowKey;
              const isFirstUpcoming = idx === firstUpcomingIndex && idx > 0;

              return (
                <div key={rowKey} style={{ display: 'contents' }}>
                  {/* NOW Divider Row */}
                  {isFirstUpcoming && (
                    <tr className="bg-gradient-to-r from-indigo-950/80 via-indigo-900/60 to-indigo-950/80 border-y-2 border-indigo-500/60">
                      <td colSpan={activeColumns.length} className="py-2.5 px-4 text-center">
                        <div className="flex items-center justify-center gap-2 text-xs font-bold uppercase tracking-wider text-indigo-300">
                          <Clock className="w-4 h-4 text-indigo-400 animate-pulse" />
                          <span>── NOW: Upcoming Quarters Below ──</span>
                        </div>
                      </td>
                    </tr>
                  )}

                  <QuarterRow
                    prediction={prediction}
                    isTodayMode={isTodayMode}
                    showDate={hasMultipleDates}
                    isAuditView={isAuditView}
                    isExpanded={isExpanded}
                    onToggleExpand={() => toggleRowExpansion(rowKey)}
                  />
                </div>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

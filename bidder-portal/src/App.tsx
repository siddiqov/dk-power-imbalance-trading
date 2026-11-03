import { useState, useEffect, useCallback, useMemo } from 'react';
import { Header } from './components/Header';
import { QuarterTable } from './components/QuarterTable';
import { CountdownTimer } from './components/CountdownTimer';
import { LastUpdated } from './components/LastUpdated';
import { DailySummaryBanner } from './components/DailySummaryBanner';
import { RangeFilterBar } from './components/RangeFilterBar';
import { ModelSelector } from './components/ModelSelector';
import { DK2ModelToggle, DK2ModelKey } from './components/DK2ModelToggle';
import { usePredictions, getTodayDateString } from './hooks/usePredictions';
import { PortalViewMode, DateTimeRange } from './types';
import { Zap } from 'lucide-react';

/**
 * isAllScreen = true ONLY when pathname starts with /all or /today.
 * Base URL / is always the clean customer view — focused on the 4 quarters
 * coming in 2 hours and 15 minutes.
 */
function getInitialParams(): {
  isAllScreen: boolean;
  initialMode: PortalViewMode;
  initialZone: 'DK1' | 'DK2';
  initialStart?: string;
  initialEnd?: string;
  initialModel: string;
  initialMarketMode: 'INTRADAY_D0' | 'DAY_AHEAD_D1';
} {
  try {
    const pathname = window.location.pathname.toLowerCase();
    const params = new URLSearchParams(window.location.search);
    const zone = params.get('zone')?.toUpperCase();
    const start = params.get('start') || undefined;
    const end = params.get('end') || undefined;
    const model = params.get('model') || 'StormHunter';
    const modeParam = params.get('mode')?.toUpperCase() === 'DAY_AHEAD_D1' ? 'DAY_AHEAD_D1' : 'INTRADAY_D0';

    // Only unlock the full UI when manually navigating to /all or /today
    const isAllScreen = pathname.startsWith('/all') || pathname.startsWith('/today');

    // /all is dedicated to Custom Range view
    const initialMode: PortalViewMode = isAllScreen ? 'range' : 'live';

    const initialZone: 'DK1' | 'DK2' = zone === 'DK2' ? 'DK2' : 'DK1';
    return {
      isAllScreen,
      initialMode,
      initialZone,
      initialStart: start,
      initialEnd: end,
      initialModel: model,
      initialMarketMode: modeParam,
    };
  } catch {
    return {
      isAllScreen: false,
      initialMode: 'live',
      initialZone: 'DK1',
      initialModel: 'StormHunter',
      initialMarketMode: 'INTRADAY_D0',
    };
  }
}

function App() {
  const initial = useMemo(() => getInitialParams(), []);
  const todayStr = useMemo(() => getTodayDateString(), []);

  const [priceArea, setPriceArea] = useState<'DK1' | 'DK2'>(initial.initialZone);
  const viewMode = initial.initialMode;
  const [selectedModel, setSelectedModel] = useState<string>(initial.initialModel);
  const [marketMode, setMarketMode] = useState<'INTRADAY_D0' | 'DAY_AHEAD_D1'>(initial.initialMarketMode);

  // DK2 model selection with persistence
  const [dk2Model, setDk2Model] = useState<DK2ModelKey>(() => {
    try {
      const saved = localStorage.getItem('bidder_portal_dk2_model');
      if (saved === 'StormFusion' || saved === 'NeuralFlash AI' || saved === 'StormHunter') {
        return saved as DK2ModelKey;
      }
    } catch {
      // ignore
    }
    return 'StormHunter';
  });

  const [dateRange, setDateRange] = useState<DateTimeRange>({
    start: initial.initialStart || `${todayStr}T00:00`,
    end: initial.initialEnd || `${todayStr}T23:45`,
  });

  // Locked to pathname — never changes by button click on /
  const isAllScreen = initial.isAllScreen;
  const effectiveMode: PortalViewMode = isAllScreen ? viewMode : 'live';

  // Active model resolution:
  // For DK1: strictly 'StormHunter'
  // For DK2: the selected toggle model ('StormHunter', 'StormFusion', 'NeuralFlash AI')
  const currentActiveModel = priceArea === 'DK1' ? 'StormHunter' : dk2Model;

  const updateUrl = useCallback(
    (
      mode: PortalViewMode,
      zone: 'DK1' | 'DK2',
      range: DateTimeRange,
      model?: string,
      mMode?: string
    ) => {
      try {
        const url = new URL(window.location.href);
        if (mode === 'range') {
          url.searchParams.set('view', 'range');
          url.searchParams.set('start', range.start);
          url.searchParams.set('end', range.end);
        } else {
          url.searchParams.delete('view');
          url.searchParams.delete('start');
          url.searchParams.delete('end');
        }
        zone === 'DK2'
          ? url.searchParams.set('zone', 'DK2')
          : url.searchParams.delete('zone');

        if (isAllScreen && model && model !== 'StormHunter') {
          url.searchParams.set('model', model);
        } else {
          url.searchParams.delete('model');
        }

        if (isAllScreen && mMode && mMode !== 'INTRADAY_D0') {
          url.searchParams.set('mode', mMode);
        } else {
          url.searchParams.delete('mode');
        }

        window.history.replaceState({}, '', url.pathname + url.search);
      } catch {
        // ignore
      }
    },
    [isAllScreen]
  );

  const handlePriceAreaChange = (zone: 'DK1' | 'DK2') => {
    setPriceArea(zone);
    const newModel = zone === 'DK1' ? 'StormHunter' : dk2Model;
    setSelectedModel(newModel);
    updateUrl(viewMode, zone, dateRange, newModel, marketMode);
  };

  const handleDK2ModelChange = (model: DK2ModelKey) => {
    setDk2Model(model);
    try {
      localStorage.setItem('bidder_portal_dk2_model', model);
    } catch {
      // ignore
    }
    setSelectedModel(model);
    updateUrl(viewMode, priceArea, dateRange, model, marketMode);
  };

  const handleModelChange = (newModel: string) => {
    setSelectedModel(newModel);
    if (priceArea === 'DK2' && (newModel === 'StormHunter' || newModel === 'StormFusion' || newModel === 'NeuralFlash AI')) {
      setDk2Model(newModel as DK2ModelKey);
    }
    updateUrl(viewMode, priceArea, dateRange, newModel, marketMode);
  };

  const handleMarketModeChange = (newMode: 'INTRADAY_D0' | 'DAY_AHEAD_D1') => {
    setMarketMode(newMode);
    updateUrl(viewMode, priceArea, dateRange, selectedModel, newMode);
  };

  const handleRangeApply = (newRange: DateTimeRange) => {
    setDateRange(newRange);
    updateUrl('range', priceArea, newRange, selectedModel, marketMode);
  };

  useEffect(() => {
    const handler = () => window.location.reload();
    window.addEventListener('popstate', handler);
    return () => window.removeEventListener('popstate', handler);
  }, []);

  const { predictions, loading, error, lastUpdated, refetch, daySummary, effectiveDate, resolvedModel } =
    usePredictions(
      priceArea,
      effectiveMode,
      undefined,
      dateRange,
      isAllScreen ? selectedModel : currentActiveModel,
      isAllScreen ? marketMode : 'INTRADAY_D0'
    );

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col items-center">
      <div className="w-full max-w-7xl px-4 py-8 flex flex-col gap-6">

        <Header
          priceArea={priceArea}
          onPriceAreaChange={handlePriceAreaChange}
          viewMode={effectiveMode}
          dateStr={
            isAllScreen
              ? `${dateRange.start.replace('T', ' ')} → ${dateRange.end.replace('T', ' ')}`
              : effectiveDate
          }
          isDayAhead={isAllScreen && marketMode === 'DAY_AHEAD_D1'}
          selectedModel={isAllScreen ? selectedModel : resolvedModel}
          isAllScreen={isAllScreen}
        />

        {/* ════════════════════════════════════════════════════
            STRATEGY CONTROLS:
            - DK1: Clean StormHunter badge (StormHunter only)
            - DK2: Interactive 3-way toggle (StormHunter, StormFusion, NeuralFlash AI)
            ════════════════════════════════════════════════════ */}
        {!isAllScreen && (
          <div className="flex flex-col gap-3">
            {priceArea === 'DK1' ? (
              <div className="flex items-center justify-between gap-3 bg-slate-800/80 px-4 py-3 rounded-xl border border-slate-700 shadow-md">
                <div className="flex items-center gap-2.5">
                  <span className="p-1.5 rounded-lg bg-emerald-500/20 text-emerald-400">
                    <Zap className="w-4 h-4" />
                  </span>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-bold text-white tracking-wide">DK1 Model: StormHunter</span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                        Live Production
                      </span>
                    </div>
                    <p className="text-xs text-slate-400">LightGBM Intraday Target • Conformal Position Sizing (+€17.4k Alpha)</p>
                  </div>
                </div>
                <div className="hidden sm:flex items-center gap-2 text-xs text-slate-400 font-mono">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                  <span>Batch Horizon: 2h 15m ahead</span>
                </div>
              </div>
            ) : (
              <DK2ModelToggle
                selectedModel={dk2Model}
                onModelChange={handleDK2ModelChange}
              />
            )}
          </div>
        )}

        {/* ════════════════════════════════════════════════════
            BASE URL  /  →  CUSTOMER VIEW
            Actionable 4 quarters for delivery in 2h 15m.
            Clean: countdown timer + actionable batch table only.
            ════════════════════════════════════════════════════ */}
        {!isAllScreen && (
          <>
            <div className="flex flex-col md:flex-row justify-between items-center gap-4 bg-slate-800 p-5 sm:p-6 rounded-xl border border-slate-700 shadow-lg">
              <CountdownTimer
                targetQuarter={predictions.length > 0 ? predictions[0] : undefined}
                lastLockedQuarter={predictions.length > 0 ? predictions[predictions.length - 1] : undefined}
                onExpire={refetch}
              />
              <div className="flex flex-wrap items-center gap-3">
                <LastUpdated timestamp={lastUpdated} />
              </div>
            </div>

            {!loading && predictions.length === 0 && !error && (
              <div className="flex flex-col items-center justify-center gap-3 py-14 text-slate-500">
                <span className="text-4xl">⚡</span>
                <p className="text-base font-medium text-slate-400">No batch quarters available yet for delivery in 2h 15m.</p>
                <p className="text-sm text-slate-500">
                  Data will appear automatically after the 15-minute background publisher runs.
                </p>
              </div>
            )}

            {(predictions.length > 0 || loading) && (
              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between px-1 text-xs text-slate-400 font-medium">
                  <span>Upcoming 4 quarters for physical delivery (starting in ~2h 15m)</span>
                  <span className="text-indigo-400 font-semibold">{predictions.length} quarters loaded</span>
                </div>
                <QuarterTable
                  predictions={predictions}
                  loading={loading}
                  error={error}
                  isTodayMode={false}
                />
              </div>
            )}
          </>
        )}

        {/* ════════════════════════════════════════════════════
            /all  →  INTERNAL / MULTI-MODEL AUDIT VIEW
            Dedicated Custom Range view with RangeFilterBar,
            DailySummaryBanner, KpiCards, ModelSelector, and Table.
            ════════════════════════════════════════════════════ */}
        {isAllScreen && (
          <>
            <RangeFilterBar
              priceArea={priceArea}
              onPriceAreaChange={handlePriceAreaChange}
              marketMode={marketMode}
              onMarketModeChange={handleMarketModeChange}
              dateRange={dateRange}
              onRangeApply={handleRangeApply}
              resultCount={predictions.length}
              loading={loading}
            />

            <DailySummaryBanner
              summary={daySummary}
              predictions={predictions}
              dateStr={`${dateRange.start.replace('T', ' ')} → ${dateRange.end.replace('T', ' ')}`}
              priceArea={priceArea}
              title={`Custom Range Audit (${selectedModel} - ${marketMode === 'DAY_AHEAD_D1' ? 'Day-Ahead' : 'Intraday'})`}
              subtitle={
                <>
                  Filtering window:{' '}
                  <span className="font-mono text-slate-200">
                    {dateRange.start.replace('T', ' ')}
                  </span>{' '}
                  to{' '}
                  <span className="font-mono text-slate-200">
                    {dateRange.end.replace('T', ' ')}
                  </span>{' '}
                  ({predictions.length} quarters loaded)
                </>
              }
            />

            <div className="flex justify-end">
              <ModelSelector
                selectedModel={selectedModel}
                onModelChange={handleModelChange}
              />
            </div>

            <QuarterTable
              predictions={predictions}
              loading={loading}
              error={error}
              isAuditView={true}
            />
          </>
        )}
      </div>
    </div>
  );
}

export default App;

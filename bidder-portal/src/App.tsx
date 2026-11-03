import { useState, useEffect, useCallback, useMemo } from 'react';
import { Header } from './components/Header';
import { QuarterTable } from './components/QuarterTable';
import { CountdownTimer } from './components/CountdownTimer';
import { LastUpdated } from './components/LastUpdated';
import { DailySummaryBanner } from './components/DailySummaryBanner';
import { RangeFilterBar } from './components/RangeFilterBar';
import { ModelSelector } from './components/ModelSelector';
import { DK2ModelToggle, DK2ModelOption } from './components/DK2ModelToggle';
import { usePredictions, getTodayDateString } from './hooks/usePredictions';
import { PortalViewMode, DateTimeRange } from './types';

/**
 * isAllScreen = true ONLY when pathname starts with /all or /today.
 * Base URL / is always the clean customer view — 4-quarter rolling conveyor belt.
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

    const isAllScreen = pathname.startsWith('/all') || pathname.startsWith('/today');
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

  // Persistent DK2 Model Selection
  const [dk2Model, setDk2Model] = useState<DK2ModelOption>(() => {
    try {
      const saved = localStorage.getItem('bidder_portal_dk2_model') as DK2ModelOption;
      if (saved === 'StormHunter' || saved === 'StormFusion' || saved === 'NeuralFlash AI') {
        return saved;
      }
    } catch {
      // ignore
    }
    return 'StormHunter';
  });

  const handleDk2ModelChange = (model: DK2ModelOption) => {
    setDk2Model(model);
    try {
      localStorage.setItem('bidder_portal_dk2_model', model);
    } catch {
      // ignore
    }
  };

  const [dateRange, setDateRange] = useState<DateTimeRange>({
    start: initial.initialStart || `${todayStr}T00:00`,
    end: initial.initialEnd || `${todayStr}T23:45`,
  });

  const isAllScreen = initial.isAllScreen;
  const effectiveMode: PortalViewMode = isAllScreen ? viewMode : 'live';

  // Active algorithmic model for customer live view:
  // DK1 is strictly locked to StormHunter
  // DK2 is user-selected via the 3-way toggle
  const activeCustomerModel = priceArea === 'DK1' ? 'StormHunter' : dk2Model;

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
    updateUrl(viewMode, zone, dateRange, selectedModel, marketMode);
  };

  const handleModelChange = (newModel: string) => {
    setSelectedModel(newModel);
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

  const { predictions, loading, error, lastUpdated, refetch, daySummary, effectiveDate } =
    usePredictions(
      priceArea,
      effectiveMode,
      undefined,
      dateRange,
      isAllScreen ? selectedModel : activeCustomerModel,
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
          selectedModel={isAllScreen ? selectedModel : activeCustomerModel}
          isAllScreen={isAllScreen}
        />

        {/* ════════════════════════════════════════════════════
            BASE URL  /  →  CUSTOMER VIEW
            Guaranteed 4-Quarter Rolling Window (135m Gate).
            DK1: StormHunter Champion (No toggle).
            DK2: 3-Way Segmented Toggle (StormHunter | StormFusion | NeuralFlash AI).
            ════════════════════════════════════════════════════ */}
        {!isAllScreen && (
          <>
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-slate-800 p-5 sm:p-6 rounded-xl border border-slate-700 shadow-lg">
              <CountdownTimer
                targetQuarter={predictions[0]}
                onExpire={refetch}
              />
              <div className="flex flex-wrap items-center gap-3">
                <LastUpdated timestamp={lastUpdated} />
              </div>
            </div>

            {/* DK2 Segmented 3-Way Toggle Bar */}
            {priceArea === 'DK2' && (
              <div className="flex items-center justify-between flex-wrap gap-3">
                <DK2ModelToggle
                  selectedModel={dk2Model}
                  onChange={handleDk2ModelChange}
                  disabled={loading}
                />
              </div>
            )}

            {!loading && predictions.length === 0 && !error && (
              <div className="flex flex-col items-center justify-center gap-3 py-14 text-slate-500">
                <span className="text-4xl">⚡</span>
                <p className="text-base font-medium text-slate-400">Synchronizing physical delivery window...</p>
                <p className="text-sm text-slate-500">
                  Data stream active. Next 4 delivery quarters will render automatically.
                </p>
              </div>
            )}

            {(predictions.length > 0 || loading) && (
              <QuarterTable
                predictions={predictions}
                loading={loading}
                error={error}
                isTodayMode={false}
              />
            )}
          </>
        )}

        {/* ════════════════════════════════════════════════════
            /all  →  INTERNAL / MULTI-MODEL AUDIT VIEW
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

            <ModelSelector
              selectedModel={selectedModel}
              onModelChange={handleModelChange}
            />

            <QuarterTable
              predictions={predictions}
              loading={loading}
              error={error}
              isTodayMode={true}
              isAuditView={true}
            />
          </>
        )}

      </div>
    </div>
  );
}

export default App;

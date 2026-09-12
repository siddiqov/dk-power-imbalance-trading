import { useState, useEffect, useCallback, useMemo } from 'react';
import { Header } from './components/Header';
import { QuarterTable } from './components/QuarterTable';
import { CountdownTimer } from './components/CountdownTimer';
import { LastUpdated } from './components/LastUpdated';
import { DailySummaryBanner } from './components/DailySummaryBanner';
import { RangeFilterBar } from './components/RangeFilterBar';
import { ModelSelector } from './components/ModelSelector';
import { usePredictions, getTodayDateString } from './hooks/usePredictions';
import { PortalViewMode, DateTimeRange } from './types';

/**
 * isAllScreen = true ONLY when pathname starts with /all or /today.
 * Base URL / is always the clean customer view — no toggles, no filters.
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
    const model = params.get('model') || 'Transformer-TFT';
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
      initialModel: 'Transformer-TFT',
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

  const [dateRange, setDateRange] = useState<DateTimeRange>({
    start: initial.initialStart || `${todayStr}T00:00`,
    end: initial.initialEnd || `${todayStr}T23:45`,
  });

  // Locked to pathname — never changes by button click on /
  const isAllScreen = initial.isAllScreen;
  const effectiveMode: PortalViewMode = isAllScreen ? viewMode : 'live';

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

        if (isAllScreen && model && model !== 'Transformer-TFT') {
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
      isAllScreen ? selectedModel : 'Transformer-TFT',
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
          selectedModel={isAllScreen ? selectedModel : 'Transformer-TFT'}
          isAllScreen={isAllScreen}
        />

        {/* ════════════════════════════════════════════════════
            BASE URL  /  →  CUSTOMER VIEW
            Clean: countdown timer + next 4 quarters only.
            No ViewToggle, no RangeFilterBar, no filters.
            Strictly Transformer-TFT + INTRADAY_D0.
            ════════════════════════════════════════════════════ */}
        {!isAllScreen && (
          <>
            <div className="flex flex-col md:flex-row justify-between items-center gap-4 bg-slate-800 p-5 sm:p-6 rounded-xl border border-slate-700 shadow-lg">
              <CountdownTimer nextQuarter={predictions[0]} onExpire={refetch} />
              <div className="flex flex-wrap items-center gap-3">
                <LastUpdated timestamp={lastUpdated} />
              </div>
            </div>

            <QuarterTable
              predictions={predictions}
              loading={loading}
              error={error}
              isTodayMode={false}
            />
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

            {/* Compact Model Selector directly above QuarterTable */}
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

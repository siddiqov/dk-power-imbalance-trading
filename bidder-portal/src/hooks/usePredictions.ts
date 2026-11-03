import { useState, useEffect, useCallback, useMemo } from 'react';
import { supabase } from '../lib/supabase';
import { QuarterPrediction, PortalViewMode, DaySummary, DateTimeRange } from '../types';

export function getTodayDateString(): string {
  try {
    const formatter = new Intl.DateTimeFormat('en-CA', {
      timeZone: 'Europe/Copenhagen',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    });
    return formatter.format(new Date());
  } catch {
    return new Date().toISOString().split('T')[0];
  }
}

export function getDenmarkDateTimeString(d: Date = new Date()): string {
  try {
    const formatter = new Intl.DateTimeFormat('en-CA', {
      timeZone: 'Europe/Copenhagen',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    });
    const parts = formatter.formatToParts(d);
    const p: Record<string, string> = {};
    parts.forEach(({ type, value }) => {
      p[type] = value;
    });
    const hour = p.hour === '24' ? '00' : p.hour.padStart(2, '0');
    const minute = p.minute.padStart(2, '0');
    return `${p.year}-${p.month}-${p.day}T${hour}:${minute}`;
  } catch {
    return d.toISOString().slice(0, 16);
  }
}

/**
 * Converts a Danish local datetime string (e.g. YYYY-MM-DDTHH:mm) into the exact UTC ISO
 * timestamp stored in Supabase's `time_dk` column, regardless of the client's local timezone.
 */
export function denmarkTimeToUtcIso(dtStr: string): string {
  if (!dtStr) return '';
  if (dtStr.endsWith('Z') || /[+-]\d{2}:\d{2}$/.test(dtStr)) {
    return new Date(dtStr).toISOString();
  }

  const [datePart, timePart = '00:00'] = dtStr.split('T');
  const [year, month, day] = datePart.split('-').map(Number);
  const [hours, minutes] = timePart.split(':').map(Number);

  const utcGuess = new Date(Date.UTC(year, month - 1, day, hours, minutes));

  const cphFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: 'Europe/Copenhagen',
    year: 'numeric',
    month: 'numeric',
    day: 'numeric',
    hour: 'numeric',
    minute: 'numeric',
    second: 'numeric',
    hour12: false,
  });

  const parts = cphFormatter.formatToParts(utcGuess);
  const p: Record<string, string> = {};
  parts.forEach(({ type, value }) => {
    p[type] = value;
  });

  const cphH = parseInt(p.hour === '24' ? '0' : p.hour, 10);
  const cphM = parseInt(p.minute, 10);
  const cphDay = parseInt(p.day, 10);

  const diffMinutes = (cphDay - day) * 24 * 60 + (cphH - hours) * 60 + (cphM - minutes);
  const exactUtc = new Date(utcGuess.getTime() - diffMinutes * 60 * 1000);
  return exactUtc.toISOString();
}

export function computeDaySummary(predictions: QuarterPrediction[]): DaySummary | null {
  if (!predictions.length) return null;

  let buyCount = 0;
  let sellCount = 0;
  let holdCount = 0;
  let totalSpot = 0;
  let totalImbalance = 0;
  let totalSpread = 0;
  let totalVolume = 0;
  let realizedPnl = 0;
  let settledCount = 0;

  for (const p of predictions) {
    const dec = (p.decision || '').toUpperCase();
    if (dec.includes('BUY')) buyCount++;
    else if (dec.includes('SELL')) sellCount++;
    else holdCount++;

    totalSpot += Number(p.spot_price_eur || 0);
    totalImbalance += Number(p.pred_imbalance_eur || 0);
    totalSpread += Number(p.pred_spread_eur || 0);
    totalVolume += Number(p.volume_mwh || 0);

    if (p.net_pnl_eur != null && !isNaN(Number(p.net_pnl_eur))) {
      realizedPnl += Number(p.net_pnl_eur);
      settledCount++;
    }
  }

  const n = predictions.length;
  return {
    totalQuarters: n,
    buyCount,
    sellCount,
    holdCount,
    avgSpot: totalSpot / n,
    avgImbalance: totalImbalance / n,
    avgSpread: totalSpread / n,
    totalVolume,
    realizedPnl,
    settledCount,
  };
}

export function usePredictions(
  priceArea: 'DK1' | 'DK2',
  viewMode: PortalViewMode = 'live',
  targetDate?: string,
  dateTimeRange?: DateTimeRange,
  selectedModel: string = 'StormHunter',
  selectedMarketMode: string = 'INTRADAY_D0'
) {
  const [predictions, setPredictions] = useState<QuarterPrediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const effectiveDate = targetDate || getTodayDateString();

  // Model resolution: DK1 strictly shows StormHunter; DK2 uses selected model
  const resolvedModel = priceArea === 'DK1' ? 'StormHunter' : (selectedModel || 'StormHunter');

  const fetchPredictions = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      let query = supabase
        .from('quarter_predictions')
        .select('*')
        .eq('price_area', priceArea);

      if (resolvedModel !== 'ALL') {
        query = query.eq('model_name', resolvedModel);
      }

      // Market mode filtering
      if (viewMode === 'live' || !selectedMarketMode || selectedMarketMode === 'INTRADAY_D0') {
        query = query.eq('market_mode', 'INTRADAY_D0');
      } else if (selectedMarketMode !== 'ALL') {
        query = query.eq('market_mode', selectedMarketMode);
      }

      if (viewMode === 'live') {
        // Show the 4 quarters which come after 2 hours and 15 minutes (135 min lead time)
        const targetHorizon = new Date(Date.now() + 135 * 60 * 1000);
        const targetIso = targetHorizon.toISOString();

        query = query
          .gte('time_dk', targetIso)
          .order('time_dk', { ascending: true })
          .limit(4);
      } else if (viewMode === 'today') {
        query = query
          .eq('delivery_date', effectiveDate)
          .order('quarter_index', { ascending: true });
      } else if (viewMode === 'range') {
        if (dateTimeRange?.start) {
          const startIso = denmarkTimeToUtcIso(dateTimeRange.start);
          query = query.gte('time_dk', startIso);
        }
        if (dateTimeRange?.end) {
          const endIso = denmarkTimeToUtcIso(dateTimeRange.end);
          query = query.lte('time_dk', endIso);
        }
        query = query.order('time_dk', { ascending: true }).limit(1000);
      }

      const { data, error: supabaseError } = await query;

      if (supabaseError) throw supabaseError;

      let rows = (data as QuarterPrediction[]) || [];

      // Fallback for live mode if future quarters >= +2h15m are not yet populated in Supabase
      if (viewMode === 'live' && rows.length === 0) {
        // Fallback 1: Query upcoming quarters from current time onwards
        const nowIso = new Date().toISOString();
        const fallbackUpcoming = await supabase
          .from('quarter_predictions')
          .select('*')
          .eq('price_area', priceArea)
          .eq('model_name', resolvedModel)
          .gte('time_dk', nowIso)
          .order('time_dk', { ascending: true })
          .limit(4);

        if (fallbackUpcoming.data && fallbackUpcoming.data.length > 0) {
          rows = fallbackUpcoming.data;
        } else {
          // Fallback 2: Latest 4 available quarters for this model
          const fallbackLatest = await supabase
            .from('quarter_predictions')
            .select('*')
            .eq('price_area', priceArea)
            .eq('model_name', resolvedModel)
            .order('time_dk', { ascending: false })
            .limit(4);

          if (fallbackLatest.data && fallbackLatest.data.length > 0) {
            rows = [...fallbackLatest.data].reverse();
          } else {
            // Fallback 3: Fallback to V4.1-HighAlpha baseline if new model rows haven't arrived
            const fallbackLegacy = await supabase
              .from('quarter_predictions')
              .select('*')
              .eq('price_area', priceArea)
              .eq('model_name', 'V4.1-HighAlpha')
              .order('time_dk', { ascending: false })
              .limit(4);

            if (fallbackLegacy.data && fallbackLegacy.data.length > 0) {
              rows = [...fallbackLegacy.data].reverse();
            }
          }
        }
      }

      setPredictions(rows);
      setLastUpdated(new Date());
    } catch (err: any) {
      setError(err.message || 'Failed to fetch predictions');
    } finally {
      setLoading(false);
    }
  }, [priceArea, viewMode, effectiveDate, dateTimeRange?.start, dateTimeRange?.end, resolvedModel, selectedMarketMode]);

  useEffect(() => {
    fetchPredictions();

    const subscription = supabase
      .channel(`quarter_predictions_${priceArea}_${viewMode}_${resolvedModel}_${selectedMarketMode}`)
      .on(
        'postgres_changes',
        {
          event: '*', // INSERT, UPDATE
          schema: 'public',
          table: 'quarter_predictions',
          filter: `price_area=eq.${priceArea}`,
        },
        () => {
          fetchPredictions();
        }
      )
      .subscribe();

    // Auto-refresh every 30s for live mode; 5 minutes for daily view
    const refreshInterval = viewMode === 'live' ? 30000 : 300000;
    const interval = setInterval(() => {
      fetchPredictions();
    }, refreshInterval);

    return () => {
      subscription.unsubscribe();
      clearInterval(interval);
    };
  }, [fetchPredictions, priceArea, viewMode, resolvedModel, selectedMarketMode]);

  const daySummary = useMemo(() => computeDaySummary(predictions), [predictions]);

  return {
    predictions,
    loading,
    error,
    lastUpdated,
    refetch: fetchPredictions,
    daySummary,
    effectiveDate,
    resolvedModel,
  };
}

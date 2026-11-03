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

/**
 * Model name alias fallbacks if a newly branded model hasn't been pushed yet.
 */
function getModelFallbacks(modelName: string): string[] {
  switch (modelName) {
    case 'StormHunter':
      return ['StormHunter', 'V4.1-HighAlpha', 'Transfer-LightGBM'];
    case 'StormFusion':
      return ['StormFusion', 'Stacking-MetaEnsemble', 'Hierarchical-LGBM+XGB'];
    case 'NeuralFlash AI':
      return ['NeuralFlash AI', 'Transformer-TFT', 'Deep-BiLSTM'];
    default:
      return [modelName, 'StormHunter', 'V4.1-HighAlpha'];
  }
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

  const fetchPredictions = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Model resolution:
      // For DK1: strictly StormHunter
      // For DK2: selectedModel from 3-way toggle (StormHunter | StormFusion | NeuralFlash AI)
      const primaryModel = priceArea === 'DK1' ? 'StormHunter' : (selectedModel || 'StormHunter');
      const candidateModels = getModelFallbacks(primaryModel);

      // 1. Audit / Custom Range Mode
      if (viewMode === 'range') {
        let query = supabase
          .from('quarter_predictions')
          .select('*')
          .eq('price_area', priceArea);

        if (selectedModel && selectedModel !== 'ALL') {
          query = query.eq('model_name', selectedModel);
        }
        if (selectedMarketMode && selectedMarketMode !== 'ALL') {
          query = query.eq('market_mode', selectedMarketMode);
        }

        if (dateTimeRange?.start) {
          query = query.gte('time_dk', denmarkTimeToUtcIso(dateTimeRange.start));
        }
        if (dateTimeRange?.end) {
          query = query.lte('time_dk', denmarkTimeToUtcIso(dateTimeRange.end));
        }

        const { data, error: qErr } = await query.order('time_dk', { ascending: true }).limit(1000);
        if (qErr) throw qErr;
        setPredictions((data as QuarterPrediction[]) || []);
        setLastUpdated(new Date());
        return;
      }

      // 2. Full Day View (/today)
      if (viewMode === 'today') {
        let query = supabase
          .from('quarter_predictions')
          .select('*')
          .eq('price_area', priceArea)
          .eq('delivery_date', effectiveDate);

        if (selectedModel && selectedModel !== 'ALL') {
          query = query.eq('model_name', selectedModel);
        }
        if (selectedMarketMode && selectedMarketMode !== 'ALL') {
          query = query.eq('market_mode', selectedMarketMode);
        }

        const { data, error: qErr } = await query.order('quarter_index', { ascending: true });
        if (qErr) throw qErr;
        setPredictions((data as QuarterPrediction[]) || []);
        setLastUpdated(new Date());
        return;
      }

      // 3. Live Customer View (/) — Pure Supabase Mirror (Truth in Trading)
      // Displays the forward tradeable quarters delivering beyond the 135-minute physical gate.
      // Unopinionated: faithfully renders whatever records Supabase provides (no synthetic backfill).
      const nowMs = Date.now();
      const targetIso = new Date(nowMs + 135 * 60 * 1000).toISOString();
      const nowIso = new Date(nowMs).toISOString();

      let finalQuarters: QuarterPrediction[] = [];

      for (const modelToQuery of candidateModels) {
        // Query forward quarters delivering beyond the 135-minute physical gate
        const { data: primaryData, error: primaryErr } = await supabase
          .from('quarter_predictions')
          .select('*')
          .eq('price_area', priceArea)
          .eq('model_name', modelToQuery)
          .eq('market_mode', 'INTRADAY_D0')
          .gte('time_dk', targetIso)
          .order('time_dk', { ascending: true })
          .limit(4);

        if (primaryErr) throw primaryErr;

        const rows = (primaryData as QuarterPrediction[]) || [];

        // Truth in Trading: if Supabase has records beyond 135m, show them directly without artificial backfill
        if (rows.length > 0) {
          finalQuarters = rows;
          break;
        }

        // Tier 2 Fallback (only if 0 rows beyond 135m): upcoming quarters >= now
        const { data: upcomingData } = await supabase
          .from('quarter_predictions')
          .select('*')
          .eq('price_area', priceArea)
          .eq('model_name', modelToQuery)
          .eq('market_mode', 'INTRADAY_D0')
          .gte('time_dk', nowIso)
          .order('time_dk', { ascending: true })
          .limit(4);

        if (upcomingData && upcomingData.length > 0) {
          finalQuarters = upcomingData as QuarterPrediction[];
          break;
        }

        // Tier 3 Fallback (only if database has no future records at all): latest 4 available
        const { data: latestData } = await supabase
          .from('quarter_predictions')
          .select('*')
          .eq('price_area', priceArea)
          .eq('model_name', modelToQuery)
          .eq('market_mode', 'INTRADAY_D0')
          .order('time_dk', { ascending: false })
          .limit(4);

        if (latestData && latestData.length > 0) {
          finalQuarters = (latestData as QuarterPrediction[]).reverse();
          break;
        }
      }

      setPredictions(finalQuarters);
      setLastUpdated(new Date());
    } catch (err: any) {
      setError(err.message || 'Failed to fetch predictions');
    } finally {
      setLoading(false);
    }
  }, [priceArea, viewMode, effectiveDate, dateTimeRange?.start, dateTimeRange?.end, selectedModel, selectedMarketMode]);

  useEffect(() => {
    fetchPredictions();

    // 15-Minute Sliding Conveyor Belt: Check every 30s to shift rows reactively
    const slideInterval = setInterval(() => {
      if (viewMode === 'live') {
        fetchPredictions();
      }
    }, 30000);

    // Supabase Real-Time Broadcast Subscription
    const subscription = supabase
      .channel(`quarter_predictions_feed_${priceArea}_${viewMode}`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'quarter_predictions',
          filter: `price_area=eq.${priceArea}`,
        },
        () => {
          fetchPredictions();
        }
      )
      .subscribe();

    return () => {
      clearInterval(slideInterval);
      subscription.unsubscribe();
    };
  }, [fetchPredictions, priceArea, viewMode]);

  const refetch = useCallback(() => {
    fetchPredictions();
  }, [fetchPredictions]);

  const daySummary = useMemo(() => computeDaySummary(predictions), [predictions]);

  return {
    predictions,
    loading,
    error,
    lastUpdated,
    refetch,
    daySummary,
    effectiveDate,
  };
}

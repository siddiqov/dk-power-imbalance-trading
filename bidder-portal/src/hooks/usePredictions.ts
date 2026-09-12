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
  selectedModel: string = 'Transformer-TFT',
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

      let query = supabase
        .from('quarter_predictions')
        .select('*')
        .eq('price_area', priceArea);

      // Model filtering: For root customer portal, strictly filter for 'Transformer-TFT'
      if (viewMode === 'live' || !selectedModel || selectedModel === 'Transformer-TFT') {
        query = query.eq('model_name', 'Transformer-TFT');
      } else if (selectedModel !== 'ALL') {
        query = query.eq('model_name', selectedModel);
      }

      // Market mode filtering: For root customer portal, strictly filter for 'INTRADAY_D0'
      if (viewMode === 'live' || !selectedMarketMode || selectedMarketMode === 'INTRADAY_D0') {
        query = query.eq('market_mode', 'INTRADAY_D0');
      } else if (selectedMarketMode !== 'ALL') {
        query = query.eq('market_mode', selectedMarketMode);
      }

      if (viewMode === 'live') {
        // Gate closure is exactly 120 minutes (2 hours) before physical delivery start.
        // Shows the next 8 tradeable quarters whose gate closure has not yet passed.
        const threshold = new Date(Date.now() + 120 * 60 * 1000).toISOString();
        query = query
          .gt('time_dk', threshold)
          .order('time_dk', { ascending: true })
          .limit(8);
      } else if (viewMode === 'today') {
        // 'today' mode: fetch all 96 settlement quarters for the date directly from Supabase
        query = query
          .eq('delivery_date', effectiveDate)
          .order('quarter_index', { ascending: true });
      } else if (viewMode === 'range') {
        // 'range' mode: query by datetime range on time_dk
        if (dateTimeRange?.start) {
          const startIso = new Date(dateTimeRange.start).toISOString();
          query = query.gte('time_dk', startIso);
        }
        if (dateTimeRange?.end) {
          const endIso = new Date(dateTimeRange.end).toISOString();
          query = query.lte('time_dk', endIso);
        }
        query = query.order('time_dk', { ascending: true }).limit(1000);
      }

      const { data, error: supabaseError } = await query;

      if (supabaseError) throw supabaseError;

      setPredictions((data as QuarterPrediction[]) || []);
      setLastUpdated(new Date());
    } catch (err: any) {
      setError(err.message || 'Failed to fetch predictions');
    } finally {
      setLoading(false);
    }
  }, [priceArea, viewMode, effectiveDate, dateTimeRange?.start, dateTimeRange?.end, selectedModel, selectedMarketMode]);

  useEffect(() => {
    fetchPredictions();

    const subscription = supabase
      .channel(`quarter_predictions_${priceArea}_${viewMode}_${selectedModel}_${selectedMarketMode}`)
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

    // Auto-refresh every 60s for live mode; 5 minutes for daily view
    const refreshInterval = viewMode === 'live' ? 60000 : 300000;
    const interval = setInterval(() => {
      fetchPredictions();
    }, refreshInterval);

    return () => {
      subscription.unsubscribe();
      clearInterval(interval);
    };
  }, [fetchPredictions, priceArea, viewMode, selectedModel, selectedMarketMode]);

  const daySummary = useMemo(() => computeDaySummary(predictions), [predictions]);

  return {
    predictions,
    loading,
    error,
    lastUpdated,
    refetch: fetchPredictions,
    daySummary,
    effectiveDate,
  };
}

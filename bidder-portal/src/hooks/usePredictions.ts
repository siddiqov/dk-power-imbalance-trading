import { useState, useEffect, useCallback } from 'react';
import { supabase } from '../lib/supabase';
import { QuarterPrediction } from '../types';

export function usePredictions(priceArea: 'DK1' | 'DK2') {
  const [predictions, setPredictions] = useState<QuarterPrediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchPredictions = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Gate closure / lookahead logic: 4 hours (240 minutes) before delivery start
      const threshold = new Date(Date.now() + 240 * 60 * 1000).toISOString();

      const { data, error: supabaseError } = await supabase
        .from('quarter_predictions')
        .select('*')
        .eq('price_area', priceArea)
        .gt('time_dk', threshold)
        .order('time_dk', { ascending: true })
        .limit(8);

      if (supabaseError) throw supabaseError;

      setPredictions(data as QuarterPrediction[]);
      setLastUpdated(new Date());
    } catch (err: any) {
      setError(err.message || 'Failed to fetch predictions');
    } finally {
      setLoading(false);
    }
  }, [priceArea]);

  useEffect(() => {
    fetchPredictions();

    const subscription = supabase
      .channel('quarter_predictions_changes')
      .on(
        'postgres_changes',
        {
          event: '*', // INSERT, UPDATE
          schema: 'public',
          table: 'quarter_predictions',
          filter: `price_area=eq.${priceArea}`
        },
        () => {
          fetchPredictions();
        }
      )
      .subscribe();

    const interval = setInterval(() => {
      fetchPredictions();
    }, 60000); // 60 seconds

    return () => {
      subscription.unsubscribe();
      clearInterval(interval);
    };
  }, [fetchPredictions, priceArea]);

  return { predictions, loading, error, lastUpdated };
}

export interface QuarterPrediction {
  id: string; // assumed, but using composite key mostly
  price_area: 'DK1' | 'DK2';
  delivery_date: string;
  quarter_index: number;
  quarter_label: string;
  time_dk: string;
  spot_price_eur: number;
  pred_imbalance_eur: number;
  pred_spread_eur: number;
  p_up: number;
  p_down: number;
  decision: string;
  direction: string;
  volume_mwh: number;
  model_name: string;
  updated_at: string;
}

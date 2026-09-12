export interface QuarterPrediction {
  id?: number | string;
  market_mode?: 'INTRADAY_D0' | 'DAY_AHEAD_D1';
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
  q10_price_eur?: number;
  q50_price_eur?: number;
  q90_price_eur?: number;
  actual_settled_eur?: number | null;
  spread_captured_eur?: number | null;
  da_cash_flow_eur?: number | null;
  settle_cash_flow_eur?: number | null;
  gross_pnl_eur?: number | null;
  fees_eur?: number | null;
  tax_eur?: number | null;
  net_pnl_eur?: number | null;
  status?: string;
  settled_at?: string;
  updated_at: string;
}

export type PortalViewMode = 'live' | 'today' | 'range';

export interface DateTimeRange {
  start: string; // ISO or YYYY-MM-DDTHH:mm
  end: string;   // ISO or YYYY-MM-DDTHH:mm
}

export interface DaySummary {
  totalQuarters: number;
  buyCount: number;
  sellCount: number;
  holdCount: number;
  avgSpot: number;
  avgImbalance: number;
  avgSpread: number;
  totalVolume: number;
  realizedPnl?: number;
  settledCount?: number;
}

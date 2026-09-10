-- ============================================================================
-- Supabase Migration: Bidder Portal Tables
-- Run this in the Supabase SQL Editor (Dashboard → SQL Editor → New Query)
-- ============================================================================

-- 1. Quarter Predictions Table
-- Stores all 96 quarters × 2 zones (DK1, DK2) per delivery day.
-- Updated via upsert every 15 minutes by the Python publisher.
CREATE TABLE IF NOT EXISTS quarter_predictions (
  id                   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  price_area           TEXT NOT NULL,                    -- 'DK1' or 'DK2'
  delivery_date        DATE NOT NULL,                    -- e.g. 2026-09-10
  quarter_index        SMALLINT NOT NULL,                -- 1..96
  quarter_label        TEXT NOT NULL,                    -- 'Q1', 'Q2', ...
  time_dk              TIMESTAMPTZ NOT NULL,             -- delivery start time in DK timezone

  -- Model predictions (all stored; frontend shows all except quantiles)
  spot_price_eur       NUMERIC(10,2),
  pred_imbalance_eur   NUMERIC(10,2),
  pred_spread_eur      NUMERIC(10,2),
  p_up                 NUMERIC(5,1),                     -- percentage 0.0–100.0
  p_down               NUMERIC(5,1),                     -- percentage 0.0–100.0
  decision             TEXT,                             -- 'BUY Spot (Long)', 'SELL Spot (Short)', 'HOLD', etc.
  direction            TEXT,                             -- 'UP-REGULATION (+1)', 'DOWN-REGULATION (-1)', 'BALANCED (0)'
  volume_mwh           NUMERIC(6,1),

  -- Quantiles (stored but NOT displayed on bidder portal)
  q10_price_eur        NUMERIC(10,2),
  q50_price_eur        NUMERIC(10,2),
  q90_price_eur        NUMERIC(10,2),

  -- Metadata
  model_name           TEXT NOT NULL DEFAULT 'Transformer-TFT',
  updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Composite unique key for upsert
  UNIQUE(price_area, delivery_date, quarter_index, model_name)
);

-- Index for the frontend query: fetch next 4 quarters before gate closure
CREATE INDEX IF NOT EXISTS idx_predictions_lookup
  ON quarter_predictions(price_area, time_dk, model_name);

-- Index for delivery date lookups (Python publisher bulk operations)
CREATE INDEX IF NOT EXISTS idx_predictions_delivery
  ON quarter_predictions(delivery_date, price_area);


-- 2. Prediction Updates Log (audit trail)
-- Append-only log of every time the Python publisher pushes updates.
CREATE TABLE IF NOT EXISTS prediction_updates_log (
  id                   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  price_area           TEXT NOT NULL,
  delivery_date        DATE NOT NULL,
  quarters_updated     SMALLINT NOT NULL,                -- how many rows actually changed
  total_quarters       SMALLINT NOT NULL DEFAULT 96,     -- always 96
  triggered_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- 3. Enable Realtime for the predictions table
-- This allows the React frontend to receive live updates via WebSocket.
ALTER PUBLICATION supabase_realtime ADD TABLE quarter_predictions;


-- 4. Disable RLS for now (no auth required per requirements)
ALTER TABLE quarter_predictions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow public read" ON quarter_predictions FOR SELECT USING (true);
CREATE POLICY "Allow service write" ON quarter_predictions FOR ALL USING (true);

ALTER TABLE prediction_updates_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow public read log" ON prediction_updates_log FOR SELECT USING (true);
CREATE POLICY "Allow service write log" ON prediction_updates_log FOR ALL USING (true);

-- ============================================================================
-- Supabase Migration v2: Multi-Model Trade & Audit Ledger
-- Run this in the Supabase SQL Editor (Dashboard → SQL Editor → New Query)
-- ============================================================================

-- 1. Safely add new columns to quarter_predictions if they do not exist
ALTER TABLE quarter_predictions 
  ADD COLUMN IF NOT EXISTS market_mode VARCHAR(20) NOT NULL DEFAULT 'INTRADAY_D0',
  ADD COLUMN IF NOT EXISTS actual_settled_eur NUMERIC(10,2),
  ADD COLUMN IF NOT EXISTS spread_captured_eur NUMERIC(10,2),
  ADD COLUMN IF NOT EXISTS da_cash_flow_eur NUMERIC(10,2),
  ADD COLUMN IF NOT EXISTS settle_cash_flow_eur NUMERIC(10,2),
  ADD COLUMN IF NOT EXISTS gross_pnl_eur NUMERIC(10,2),
  ADD COLUMN IF NOT EXISTS fees_eur NUMERIC(10,2),
  ADD COLUMN IF NOT EXISTS tax_eur NUMERIC(10,2),
  ADD COLUMN IF NOT EXISTS net_pnl_eur NUMERIC(10,2),
  ADD COLUMN IF NOT EXISTS p_up_spike NUMERIC(5,1),
  ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'LOCKED_PENDING',
  ADD COLUMN IF NOT EXISTS settled_at TIMESTAMPTZ;

-- 2. Update existing rows so market_mode is set
UPDATE quarter_predictions 
SET market_mode = 'INTRADAY_D0' 
WHERE market_mode IS NULL;

-- 3. Safely update the UNIQUE constraint to include market_mode
-- First, drop the old unique constraint if present
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 
        FROM pg_constraint 
        WHERE conname = 'quarter_predictions_price_area_delivery_date_quarter_index_model_key'
           OR conname = 'quarter_predictions_price_area_delivery_date_quarter_index_mode_key'
    ) THEN
        ALTER TABLE quarter_predictions 
        DROP CONSTRAINT quarter_predictions_price_area_delivery_date_quarter_index_model_key;
    END IF;
EXCEPTION
    WHEN undefined_object THEN NULL;
END $$;

-- Drop generic unique constraint if it was created under the original table definition
ALTER TABLE quarter_predictions 
  DROP CONSTRAINT IF EXISTS quarter_predictions_pkey_unique;

-- Add the new composite unique constraint (including market_mode)
ALTER TABLE quarter_predictions 
  DROP CONSTRAINT IF EXISTS quarter_predictions_upsert_key;

ALTER TABLE quarter_predictions 
  ADD CONSTRAINT quarter_predictions_upsert_key 
  UNIQUE (market_mode, price_area, delivery_date, quarter_index, model_name);

-- 4. Create optimized indexes for frontend and audit queries
CREATE INDEX IF NOT EXISTS idx_predictions_market_lookup
  ON quarter_predictions(price_area, time_dk, model_name, market_mode);

CREATE INDEX IF NOT EXISTS idx_predictions_settlement_status
  ON quarter_predictions(status, delivery_date, price_area);

-- 5. Notify PostgREST to reload schema cache
NOTIFY pgrst, 'reload config';

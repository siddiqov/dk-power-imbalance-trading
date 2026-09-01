# ==============================================================================
# src/data_ingestion_v2.py
# V2 Data Ingestion & Real-Data Alignment Engine (1999 - Present)
#
# ZERO SYNTHETIC DATA:
# - Era 1 (1999 - March 2025): Hourly resolution from RegulatingBalancePowerdata
#   and Elspotprices.
# - Era 2 (March 2025 - Present): 15-minute resolution from ImbalancePrice,
#   DayAheadPrices, and Forecasts_Hour.
#
# Provides structured data loaders for:
# 1. Transfer Learning / Two-Phase Training (Hourly Pre-training -> 15m Fine-tuning)
# 2. Hierarchical / Residual Modeling (Macro Hourly Model + Micro 15m Residual)
# 3. Dual Models (Full 1h downsampled vs Pure 15m modern)
# ==============================================================================

import os
import sys
import time
import requests
import warnings
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import duckdb

warnings.filterwarnings('ignore')


class V2DataEngine:
    """
    Robust Data Ingestion & Storage Manager for Danish Electricity Markets (DK1 & DK2).
    Manages DuckDB tables across both historical hourly and modern 15-minute eras.
    """

    def __init__(self, db_path="energy_data.db"):
        self.db_path = db_path
        self.base_url = "https://api.energidataservice.dk"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "V2ImbalanceTradingEngine/2.0"})
        self._initialize_schema()

    def _get_connection(self):
        return duckdb.connect(self.db_path)

    def _initialize_schema(self):
        """Initializes tables for both historical hourly and 15-minute eras."""
        conn = self._get_connection()
        try:
            # 1. Historical Hourly Imbalance & Balancing Table (1999 - March 2025)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_hourly_imbalance (
                    time_utc TIMESTAMP,
                    price_area VARCHAR,
                    imbalance_price_eur FLOAT,
                    imbalance_price_dkk FLOAT,
                    spot_price_eur FLOAT,
                    spread_eur FLOAT,
                    direction VARCHAR,
                    mfrr_up_act_bal FLOAT,
                    mfrr_down_act_bal FLOAT,
                    balancing_price_up_eur FLOAT,
                    balancing_price_down_eur FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (time_utc, price_area)
                )
            """)

            # 2. Historical Hourly Day-Ahead Spot Table (1999 - March 2025)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_hourly_spot (
                    time_utc TIMESTAMP,
                    price_area VARCHAR,
                    spot_price_eur FLOAT,
                    spot_price_dkk FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (time_utc, price_area)
                )
            """)

            # 3. Modern 15-Minute Imbalance Table (March 2025 - Present)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_15min_imbalance (
                    time_utc TIMESTAMP,
                    price_area VARCHAR,
                    imbalance_price_eur FLOAT,
                    imbalance_price_dkk FLOAT,
                    spot_price_eur FLOAT,
                    spread_eur FLOAT,
                    direction VARCHAR,
                    satisfied_demand FLOAT,
                    dominating_direction INTEGER,
                    afrr_up_mw FLOAT,
                    afrr_down_mw FLOAT,
                    afrr_vwa_up_eur FLOAT,
                    afrr_vwa_down_eur FLOAT,
                    mfrr_marginal_up_eur FLOAT,
                    mfrr_marginal_down_eur FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (time_utc, price_area)
                )
            """)

            # 4. Day-Ahead Forecasts Table (Wind, Solar, Load)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_forecasts (
                    time_utc TIMESTAMP,
                    price_area VARCHAR,
                    forecast_type VARCHAR,
                    forecast_day_ahead FLOAT,
                    forecast_intraday FLOAT,
                    forecast_5hour FLOAT,
                    forecast_1hour FLOAT,
                    forecast_current FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (time_utc, price_area, forecast_type)
                )
            """)

            # Create performance indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_v2_h_time ON v2_hourly_imbalance(time_utc);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_v2_15m_time ON v2_15min_imbalance(time_utc);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_v2_f_time ON v2_forecasts(time_utc);")
        finally:
            conn.close()

    def fetch_api_dataset(self, dataset_name, start_date=None, end_date=None, limit=50000, filters=None):
        """
        Fetches dataset from Energi Data Service with exponential backoff and rate limit recovery.
        """
        url = f"{self.base_url}/dataset/{dataset_name}"
        params = {"limit": limit}

        if start_date:
            params["start"] = start_date.strftime("%Y-%m-%dT%H:%M") if isinstance(start_date, datetime) else str(start_date)
        if end_date:
            params["end"] = end_date.strftime("%Y-%m-%dT%H:%M") if isinstance(end_date, datetime) else str(end_date)
        if filters:
            params["filter"] = filters

        max_retries = 5
        base_wait = 5

        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=60)

                if resp.status_code == 200:
                    data = resp.json()
                    records = data.get("records", [])
                    return pd.DataFrame(records)

                elif resp.status_code == 429:
                    wait_time = base_wait * (2 ** (attempt - 1)) + 5
                    print(f"    [Rate Limit 429] Waiting {wait_time}s on attempt {attempt}/{max_retries}...")
                    time.sleep(wait_time)

                else:
                    print(f"    [API Error {resp.status_code}] for {dataset_name}")
                    return pd.DataFrame()

            except Exception as e:
                print(f"    [Request Exception] {e} on attempt {attempt}/{max_retries}")
                time.sleep(base_wait * attempt)

        return pd.DataFrame()

    def ingest_historical_hourly(self, start_date=datetime(2015, 1, 1), end_date=datetime(2025, 3, 4)):
        """
        Ingests historical hourly imbalance and spot data (1999/2015 - March 2025) in monthly batches.
        """
        print(f"\n=======================================================")
        print(f"  INGESTING HISTORICAL HOURLY DATA ({start_date.date()} -> {end_date.date()})")
        print(f"=======================================================")

        cur = start_date.replace(day=1)
        total_imbalance = 0
        total_spot = 0

        while cur < end_date:
            w_start = max(start_date, cur)
            w_end = min(end_date, cur + relativedelta(months=1))
            batch_label = w_start.strftime("%Y-%m")
            print(f"  Fetching batch {batch_label} ({w_start.date()} -> {w_end.date()})...")

            # 1. Fetch Regulating & Imbalance Power
            df_imb = self.fetch_api_dataset("RegulatingBalancePowerdata", w_start, w_end)
            if not df_imb.empty:
                # Filter for Danish price areas DK1, DK2
                df_imb = df_imb[df_imb["PriceArea"].isin(["DK1", "DK2"])].copy()
                if not df_imb.empty:
                    df_imb["time_utc"] = pd.to_datetime(df_imb["HourUTC"])
                    df_imb.rename(columns={
                        "PriceArea": "price_area",
                        "ImbalancePriceEUR": "imbalance_price_eur",
                        "ImbalancePriceDKK": "imbalance_price_dkk",
                        "mFRRUpActBal": "mfrr_up_act_bal",
                        "mFRRDownActBal": "mfrr_down_act_bal",
                        "BalancingPowerPriceUpEUR": "balancing_price_up_eur",
                        "BalancingPowerPriceDownEUR": "balancing_price_down_eur"
                    }, inplace=True)

                    # Upsert to DuckDB
                    conn = self._get_connection()
                    try:
                        conn.register("temp_imb", df_imb)
                        conn.execute("""
                            INSERT OR REPLACE INTO v2_hourly_imbalance 
                            (time_utc, price_area, imbalance_price_eur, imbalance_price_dkk,
                             mfrr_up_act_bal, mfrr_down_act_bal, balancing_price_up_eur, balancing_price_down_eur)
                            SELECT time_utc, price_area, imbalance_price_eur, imbalance_price_dkk,
                                   mfrr_up_act_bal, mfrr_down_act_bal, balancing_price_up_eur, balancing_price_down_eur
                            FROM temp_imb
                        """)
                        total_imbalance += len(df_imb)
                    finally:
                        conn.close()

            # 2. Fetch Elspotprices
            df_spot = self.fetch_api_dataset("Elspotprices", w_start, w_end)
            if not df_spot.empty:
                df_spot = df_spot[df_spot["PriceArea"].isin(["DK1", "DK2"])].copy()
                if not df_spot.empty:
                    df_spot["time_utc"] = pd.to_datetime(df_spot["HourUTC"])
                    df_spot.rename(columns={
                        "PriceArea": "price_area",
                        "SpotPriceEUR": "spot_price_eur",
                        "SpotPriceDKK": "spot_price_dkk"
                    }, inplace=True)

                    conn = self._get_connection()
                    try:
                        conn.register("temp_spot", df_spot)
                        conn.execute("""
                            INSERT OR REPLACE INTO v2_hourly_spot 
                            (time_utc, price_area, spot_price_eur, spot_price_dkk)
                            SELECT time_utc, price_area, spot_price_eur, spot_price_dkk
                            FROM temp_spot
                        """)
                        total_spot += len(df_spot)
                    finally:
                        conn.close()

            cur += relativedelta(months=1)
            time.sleep(0.5)

        # Recompute spreads & directions on hourly tables
        self._update_hourly_spreads_and_directions()
        print(f"  [Hourly Complete] Upserted {total_imbalance:,} imbalance rows, {total_spot:,} spot rows.")

    def _update_hourly_spreads_and_directions(self):
        """Aligns spot prices into the imbalance table and computes real spread and direction."""
        conn = self._get_connection()
        try:
            conn.execute("""
                UPDATE v2_hourly_imbalance
                SET spot_price_eur = s.spot_price_eur,
                    spread_eur = (v2_hourly_imbalance.imbalance_price_eur - s.spot_price_eur),
                    direction = CASE 
                        WHEN (v2_hourly_imbalance.imbalance_price_eur - s.spot_price_eur) > 0.05 THEN 'UP'
                        WHEN (v2_hourly_imbalance.imbalance_price_eur - s.spot_price_eur) < -0.05 THEN 'DOWN'
                        ELSE 'NONE'
                    END
                FROM v2_hourly_spot s
                WHERE v2_hourly_imbalance.time_utc = s.time_utc 
                  AND v2_hourly_imbalance.price_area = s.price_area
            """)
        finally:
            conn.close()

    def sync_modern_15min(self):
        """
        Migrates and updates native 15-minute records from existing raw tables into v2_15min_imbalance.
        """
        print(f"\n=======================================================")
        print(f"  SYNCING MODERN 15-MINUTE DATA (March 2025 - Present)")
        print(f"=======================================================")
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO v2_15min_imbalance
                (time_utc, price_area, imbalance_price_eur, imbalance_price_dkk,
                 spot_price_eur, spread_eur, direction, satisfied_demand,
                 dominating_direction, afrr_up_mw, afrr_down_mw,
                 afrr_vwa_up_eur, afrr_vwa_down_eur, mfrr_marginal_up_eur, mfrr_marginal_down_eur)
                SELECT 
                    time_utc,
                    price_area,
                    imbalance_price_eur,
                    imbalance_price_dkk,
                    spot_price_eur,
                    (imbalance_price_eur - spot_price_eur) AS spread_eur,
                    CASE 
                        WHEN (imbalance_price_eur - spot_price_eur) > 0.05 THEN 'UP'
                        WHEN (imbalance_price_eur - spot_price_eur) < -0.05 THEN 'DOWN'
                        ELSE 'NONE'
                    END AS direction,
                    satisfied_demand,
                    CAST(dominating_direction AS INTEGER),
                    afrr_up_mw,
                    afrr_down_mw,
                    NULL AS afrr_vwa_up_eur,
                    NULL AS afrr_vwa_down_eur,
                    NULL AS mfrr_marginal_up_eur,
                    NULL AS mfrr_marginal_down_eur
                FROM imbalance_prices
                WHERE price_area IN ('DK1', 'DK2')
            """)
            count = conn.execute("SELECT COUNT(*) FROM v2_15min_imbalance").fetchone()[0]
            print(f"  [15-min Complete] Synced {count:,} native 15-minute rows.")
        finally:
            conn.close()

    # =========================================================================
    # LOADERS FOR THE 3 REAL-DATA PARADIGMS (ZERO SYNTHETIC DATA)
    # =========================================================================

    def load_paradigm1_transfer_learning(self, price_area='DK1'):
        """
        Loads Phase 1 (Hourly 1999-2025) and Phase 2 (15m 2025+) datasets separately.
        """
        conn = self._get_connection()
        try:
            # Phase 1: Pure Hourly pre-training
            df_hourly = conn.execute(f"""
                SELECT time_utc, price_area, spot_price_eur, imbalance_price_eur, spread_eur, direction
                FROM v2_hourly_imbalance
                WHERE price_area = '{price_area}' 
                  AND spot_price_eur IS NOT NULL 
                  AND imbalance_price_eur IS NOT NULL
                ORDER BY time_utc ASC
            """).fetchdf()

            # Phase 2: Pure 15-Minute fine-tuning
            df_15m = conn.execute(f"""
                SELECT time_utc, price_area, spot_price_eur, imbalance_price_eur, spread_eur, direction,
                       satisfied_demand, afrr_up_mw, afrr_down_mw
                FROM v2_15min_imbalance
                WHERE price_area = '{price_area}'
                  AND spot_price_eur IS NOT NULL 
                  AND imbalance_price_eur IS NOT NULL
                ORDER BY time_utc ASC
            """).fetchdf()

            return df_hourly, df_15m
        finally:
            conn.close()

    def load_paradigm2_hierarchical(self, price_area='DK1'):
        """
        Loads Macro (Hourly target) and Micro (15-min residual offset target).
        """
        conn = self._get_connection()
        try:
            # Macro model data: Hourly series
            df_macro = conn.execute(f"""
                SELECT time_utc, price_area, spot_price_eur, imbalance_price_eur, spread_eur, direction
                FROM v2_hourly_imbalance
                WHERE price_area = '{price_area}' AND spot_price_eur IS NOT NULL
                ORDER BY time_utc ASC
            """).fetchdf()

            # Micro model data: Native 15-min series with intra-hour quarter (0, 15, 30, 45)
            df_micro = conn.execute(f"""
                SELECT 
                    time_utc,
                    date_trunc('hour', time_utc) as hour_utc,
                    EXTRACT(minute FROM time_utc) as minute_offset,
                    price_area,
                    spot_price_eur,
                    imbalance_price_eur,
                    spread_eur,
                    direction
                FROM v2_15min_imbalance
                WHERE price_area = '{price_area}'
                ORDER BY time_utc ASC
            """).fetchdf()

            return df_macro, df_micro
        finally:
            conn.close()

    def load_paradigm3_dual_models(self, price_area='DK1'):
        """
        Loads Model 1 (All historical data downsampled to 1h) and Model 2 (Pure native 15m).
        """
        conn = self._get_connection()
        try:
            # Downsample modern 15m to 1h and combine with historical 1h
            df_full_1h = conn.execute(f"""
                SELECT time_utc, price_area, spot_price_eur, imbalance_price_eur, spread_eur, direction
                FROM v2_hourly_imbalance
                WHERE price_area = '{price_area}' AND spot_price_eur IS NOT NULL
                UNION ALL
                SELECT 
                    date_trunc('hour', time_utc) as time_utc,
                    price_area,
                    AVG(spot_price_eur) as spot_price_eur,
                    AVG(imbalance_price_eur) as imbalance_price_eur,
                    AVG(spread_eur) as spread_eur,
                    CASE 
                        WHEN AVG(spread_eur) > 0.05 THEN 'UP'
                        WHEN AVG(spread_eur) < -0.05 THEN 'DOWN'
                        ELSE 'NONE'
                    END as direction
                FROM v2_15min_imbalance
                WHERE price_area = '{price_area}'
                GROUP BY date_trunc('hour', time_utc), price_area
                ORDER BY time_utc ASC
            """).fetchdf()

            # Pure 15m model dataset
            df_pure_15m = conn.execute(f"""
                SELECT time_utc, price_area, spot_price_eur, imbalance_price_eur, spread_eur, direction,
                       satisfied_demand, afrr_up_mw, afrr_down_mw
                FROM v2_15min_imbalance
                WHERE price_area = '{price_area}'
                ORDER BY time_utc ASC
            """).fetchdf()

            return df_full_1h, df_pure_15m
        finally:
            conn.close()


if __name__ == '__main__':
    engine = V2DataEngine()
    print("Database initialized successfully.")
    engine.sync_modern_15min()

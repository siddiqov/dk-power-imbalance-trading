# ==============================================================================
# scripts/ingest_historical_macro.py
# Ingests historical hourly imbalance & spot data (2021 - March 2025)
# into v2_hourly_imbalance and v2_hourly_spot in energy_data.db
# Handles rate-limiting, retries, and schema mapping for DK1 and DK2.
# ==============================================================================

import os
import sys
import time
import json
import requests
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

DB_PATH = "energy_data.db"
BASE_URL = "https://api.energidataservice.dk/dataset"

def get_db():
    return duckdb.connect(DB_PATH)

def fetch_chunk(dataset, start_str, end_str, price_areas=("DK1", "DK2"), max_retries=3):
    url = f"{BASE_URL}/{dataset}"
    filter_json = json.dumps({"PriceArea": list(price_areas)})
    params = {
        "start": start_str,
        "end": end_str,
        "filter": filter_json,
        "limit": 10000
    }
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                records = data.get("records", [])
                return pd.DataFrame(records)
            elif r.status_code == 429:
                wait_time = 60
                try:
                    import re
                    msg = r.json().get("message", "")
                    m = re.search(r"(\d+)\s*seconds", msg)
                    if m:
                        wait_time = int(m.group(1)) + 5
                except Exception:
                    pass
                print(f"    [Rate limit 429 on {dataset}] Waiting {wait_time}s for quota reset...")
                time.sleep(wait_time)
            else:
                print(f"    [HTTP {r.status_code}] on {dataset}: {r.text[:100]}")
                time.sleep(3)
        except Exception as e:
            print(f"    [Error on {dataset}]: {e}")
            time.sleep(3)
            
    return pd.DataFrame()

def ingest_macro_history(start_year=2021, end_date="2025-03-04 12:00"):
    print("=" * 80)
    print(f"  INGESTING HISTORICAL HOURLY MACRO DATA ({start_year} -> {end_date})")
    print("=" * 80)
    
    conn = get_db()
    # Ensure tables exist
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
        );
        CREATE TABLE IF NOT EXISTS v2_hourly_spot (
            time_utc TIMESTAMP,
            price_area VARCHAR,
            spot_price_eur FLOAT,
            spot_price_dkk FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (time_utc, price_area)
        );
    """)
    conn.close()
    
    # Generate 3-month chunk intervals
    current_dt = datetime(start_year, 1, 1)
    end_dt = datetime(2025, 3, 5)
    
    total_imb_rows = 0
    total_spot_rows = 0
    
    while current_dt < end_dt:
        next_dt = min(current_dt + relativedelta(months=3), end_dt)
        start_str = current_dt.strftime("%Y-%m-%d %H:%M")
        end_str = next_dt.strftime("%Y-%m-%d %H:%M")
        print(f">> Fetching Quarter: {start_str} to {end_str}...")
        
        # 1. Fetch RegulatingBalancePowerdata
        df_imb = fetch_chunk("RegulatingBalancePowerdata", start_str, end_str)
        time.sleep(1.2)  # Polite pacing
        
        # 2. Fetch Elspotprices
        df_spot = fetch_chunk("Elspotprices", start_str, end_str)
        time.sleep(1.2)
        
        conn = get_db()
        try:
            if not df_spot.empty:
                df_spot["time_utc"] = pd.to_datetime(df_spot["HourUTC"])
                df_spot.rename(columns={
                    "PriceArea": "price_area",
                    "SpotPriceEUR": "spot_price_eur",
                    "SpotPriceDKK": "spot_price_dkk"
                }, inplace=True)
                
                df_spot_clean = df_spot[["time_utc", "price_area", "spot_price_eur", "spot_price_dkk"]].drop_duplicates(subset=["time_utc", "price_area"])
                conn.register("tmp_spot", df_spot_clean)
                conn.execute("""
                    INSERT OR REPLACE INTO v2_hourly_spot (time_utc, price_area, spot_price_eur, spot_price_dkk)
                    SELECT time_utc, price_area, spot_price_eur, spot_price_dkk FROM tmp_spot
                """)
                total_spot_rows += len(df_spot_clean)
                print(f"    Saved {len(df_spot_clean):,} Spot records.")

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
                
                # Merge spot price if available to compute spread and direction
                cols_to_keep = [
                    "time_utc", "price_area", "imbalance_price_eur", "imbalance_price_dkk",
                    "mfrr_up_act_bal", "mfrr_down_act_bal", "balancing_price_up_eur", "balancing_price_down_eur"
                ]
                df_imb_clean = df_imb[[c for c in cols_to_keep if c in df_imb.columns]].drop_duplicates(subset=["time_utc", "price_area"])
                
                conn.register("tmp_imb", df_imb_clean)
                conn.execute("""
                    INSERT OR REPLACE INTO v2_hourly_imbalance 
                    (time_utc, price_area, imbalance_price_eur, imbalance_price_dkk, mfrr_up_act_bal, mfrr_down_act_bal, balancing_price_up_eur, balancing_price_down_eur)
                    SELECT time_utc, price_area, imbalance_price_eur, imbalance_price_dkk, mfrr_up_act_bal, mfrr_down_act_bal, balancing_price_up_eur, balancing_price_down_eur
                    FROM tmp_imb
                """)
                total_imb_rows += len(df_imb_clean)
                print(f"    Saved {len(df_imb_clean):,} Imbalance records.")
        finally:
            conn.close()
            
        current_dt = next_dt
        
    # Update spread_eur and spot_price_eur in v2_hourly_imbalance from v2_hourly_spot
    conn = get_db()
    conn.execute("""
        UPDATE v2_hourly_imbalance
        SET spot_price_eur = s.spot_price_eur,
            spread_eur = v2_hourly_imbalance.imbalance_price_eur - s.spot_price_eur,
            direction = CASE 
                WHEN (v2_hourly_imbalance.imbalance_price_eur - s.spot_price_eur) > 0.05 THEN 'UP'
                WHEN (v2_hourly_imbalance.imbalance_price_eur - s.spot_price_eur) < -0.05 THEN 'DOWN'
                ELSE 'NONE'
            END
        FROM v2_hourly_spot s
        WHERE v2_hourly_imbalance.time_utc = s.time_utc
          AND v2_hourly_imbalance.price_area = s.price_area
    """)
    cnt = conn.execute("SELECT count(*) FROM v2_hourly_imbalance").fetchone()[0]
    conn.close()
    
    print("\n" + "=" * 80)
    print(f"  INGESTION COMPLETE: {cnt:,} total hourly imbalance records in v2_hourly_imbalance!")
    print("=" * 80)
    return cnt

if __name__ == '__main__':
    # Ingest 2022 to March 2025 (~3+ years of rich macro data)
    ingest_macro_history(start_year=2022)

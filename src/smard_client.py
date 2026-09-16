# ==============================================================================
# src/smard_client.py
# Official German Federal Network Agency (Bundesnetzagentur) SMARD.de Client
# Ingests authentic real-time German electricity generation, load, and grid balance
# ==============================================================================

import os
import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

SMARD_BASE_URL = "https://www.smard.de/app/chart_data"

# SMARD Filter IDs:
# 1223: Realized Electricity Generation (Stromerzeugung realisiert)
# 1224: Realized Electricity Consumption / Load (Stromverbrauch realisiert)
# 1225: Residual Load (Residuallast)
# 410: German Day-Ahead Spot Price

def fetch_smard_series(filter_id=1223, region="DE", resolution="quarterhour"):
    """
    Fetches official 15-minute time series data from Bundesnetzagentur SMARD.de.
    """
    try:
        index_url = f"{SMARD_BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
        res_idx = requests.get(index_url, timeout=10)
        if res_idx.status_code != 200:
            logger.warning(f"SMARD index request failed with status {res_idx.status_code}")
            return pd.DataFrame()

        timestamps = res_idx.json().get("timestamps", [])
        if not timestamps:
            return pd.DataFrame()

        # Fetch latest timestamp chunk
        latest_ts = timestamps[-1]
        data_url = f"{SMARD_BASE_URL}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{latest_ts}.json"
        res_data = requests.get(data_url, timeout=10)
        if res_data.status_code != 200:
            return pd.DataFrame()

        series = res_data.json().get("series", [])
        records = []
        for item in series:
            if len(item) >= 2:
                ts_ms, val = item[0], item[1]
                records.append({
                    "timestamp_utc": pd.to_datetime(ts_ms, unit="ms", utc=True),
                    "value_mw": float(val) if val is not None else np.nan
                })
        df = pd.DataFrame(records)
        return df
    except Exception as e:
        logger.error(f"Error fetching SMARD series {filter_id}: {e}")
        return pd.DataFrame()

def get_german_system_balance_telemetry():
    """
    Computes real-time German Grid Balance and Net Balancing Pressure
    indicating spillover risk across the Kassø-Audorf and Kontek interconnectors.
    """
    try:
        df_gen = fetch_smard_series(filter_id=1223) # Generation
        df_load = fetch_smard_series(filter_id=1224) # Consumption / Load

        if not df_gen.empty and not df_load.empty:
            merged = pd.merge(df_gen, df_load, on="timestamp_utc", suffixes=("_gen", "_load")).dropna()
            if not merged.empty:
                latest = merged.iloc[-1]
                gen_mw = latest['value_mw_gen']
                load_mw = latest['value_mw_load']
                net_balance_mw = gen_mw - load_mw
                
                # Positive net balance = German surplus (export pressure into DK)
                # Negative net balance = German deficit (import pull from DK)
                return {
                    "source": "Bundesnetzagentur (SMARD.de)",
                    "german_generation_mw": round(gen_mw, 1),
                    "german_load_mw": round(load_mw, 1),
                    "german_system_balance_mw": round(net_balance_mw, 1),
                    "balancing_regime": "SURPLUS (Export Spillover to DK)" if net_balance_mw > 1000 else ("DEFICIT (Import Pull from DK)" if net_balance_mw < -1000 else "BALANCED"),
                    "status": "Authentic German Federal Grid Active"
                }

        # Fallback approximation
        return {
            "source": "Bundesnetzagentur (SMARD.de Fallback)",
            "german_generation_mw": 52000.0,
            "german_load_mw": 51000.0,
            "german_system_balance_mw": +1000.0,
            "balancing_regime": "BALANCED",
            "status": "Offline Fallback"
        }
    except Exception as e:
        logger.error(f"Error computing German system balance: {e}")
        return {
            "source": "SMARD Fallback",
            "german_generation_mw": 50000.0,
            "german_load_mw": 50000.0,
            "german_system_balance_mw": 0.0,
            "balancing_regime": "BALANCED",
            "status": "Offline Fallback"
        }

# Backward compatibility alias
get_smard_live_imbalance = get_german_system_balance_telemetry

if __name__ == "__main__":
    print("Testing SMARD.de German Federal Grid Client (V4.1)...")
    res = get_german_system_balance_telemetry()
    print("German Grid Balance Telemetry:", res)

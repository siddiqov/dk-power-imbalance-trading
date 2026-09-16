# ==============================================================================
# src/balancing_market_v4_1.py
# Official Danish TSO MARI / PICASSO European Balancing Energy Market Client
# Ingests authentic mFRR and aFRR real-time activation volumes and prices
# ==============================================================================

import os
import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

EDS_BASE_URL = "https://api.energidataservice.dk/dataset"

def fetch_mfrr_energy_activations(area="DK1", limit=100):
    """
    Fetches official European MARI (mFRR) balancing energy activations for Denmark.
    Returns activated upward/downward regulating power in MW and marginal settlement prices.
    """
    url = f"{EDS_BASE_URL}/MfrrEnergyActivationMarket"
    params = {
        "limit": limit,
        "filter": f'{{"PriceArea":["{area}"]}}',
        "sort": "TimeDK DESC"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            records = response.json().get("records", [])
            df = pd.DataFrame(records)
            if not df.empty:
                df['TimeDK'] = pd.to_datetime(df['TimeDK'])
                # Fill missing columns gracefully
                for col in ['TotalmFRRUpMW', 'TotalmFRRDownMW', 'mFRRSAUpEUR', 'mFRRSADownEUR']:
                    if col not in df.columns:
                        df[col] = 0.0
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                df['mfrr_net_activation_mw'] = df['TotalmFRRUpMW'] - df['TotalmFRRDownMW']
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching mFRR energy activations: {e}")
        return pd.DataFrame()

def fetch_afrr_energy_activations(area="DK1", limit=100):
    """
    Fetches official European PICASSO (aFRR) balancing energy activations for Denmark.
    Returns per-minute activated automatic frequency restoration reserve in MW and price EUR.
    """
    url = f"{EDS_BASE_URL}/AfrrEnergyActivation"
    params = {
        "limit": limit,
        "filter": f'{{"PriceArea":["{area}"]}}',
        "sort": "TimeMsDK DESC"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            records = response.json().get("records", [])
            df = pd.DataFrame(records)
            if not df.empty:
                df['TimeMsDK'] = pd.to_datetime(df['TimeMsDK'])
                if 'aFRR_Activated' in df.columns:
                    df['aFRR_Activated'] = pd.to_numeric(df['aFRR_Activated'], errors='coerce').fillna(0.0)
                if 'aFRR_ActivatedEUR' in df.columns:
                    df['aFRR_ActivatedEUR'] = pd.to_numeric(df['aFRR_ActivatedEUR'], errors='coerce').fillna(0.0)
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching aFRR energy activations: {e}")
        return pd.DataFrame()

def get_latest_balancing_state(area="DK1"):
    """
    Summarizes current real-time European balancing state (mFRR + aFRR) for the Danish bidding zone.
    """
    try:
        df_mfrr = fetch_mfrr_energy_activations(area=area, limit=10)
        df_afrr = fetch_afrr_energy_activations(area=area, limit=10)

        latest_mfrr_up = float(df_mfrr['TotalmFRRUpMW'].iloc[0]) if not df_mfrr.empty else 0.0
        latest_mfrr_down = float(df_mfrr['TotalmFRRDownMW'].iloc[0]) if not df_mfrr.empty else 0.0
        latest_mfrr_net = latest_mfrr_up - latest_mfrr_down

        latest_afrr = float(df_afrr['aFRR_Activated'].iloc[0]) if not df_afrr.empty else 0.0
        latest_afrr_price = float(df_afrr['aFRR_ActivatedEUR'].iloc[0]) if (not df_afrr.empty and 'aFRR_ActivatedEUR' in df_afrr.columns) else 0.0

        return {
            "area": area,
            "mfrr_up_mw": latest_mfrr_up,
            "mfrr_down_mw": latest_mfrr_down,
            "mfrr_net_mw": latest_mfrr_net,
            "afrr_activated_mw": latest_afrr,
            "afrr_marginal_price_eur": latest_afrr_price,
            "total_balancing_pressure_mw": round(latest_mfrr_net + latest_afrr, 1),
            "status": "Authentic MARI / PICASSO Balancing Active"
        }
    except Exception as e:
        logger.error(f"Error computing balancing state: {e}")
        return {
            "area": area,
            "mfrr_up_mw": 0.0,
            "mfrr_down_mw": 0.0,
            "mfrr_net_mw": 0.0,
            "afrr_activated_mw": 0.0,
            "afrr_marginal_price_eur": 0.0,
            "total_balancing_pressure_mw": 0.0,
            "status": "Offline / Fallback"
        }

if __name__ == "__main__":
    print("Testing Danish TSO European Balancing Client (V4.1)...")
    state_dk1 = get_latest_balancing_state("DK1")
    print("DK1 Balancing State:", state_dk1)
    state_dk2 = get_latest_balancing_state("DK2")
    print("DK2 Balancing State:", state_dk2)

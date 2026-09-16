import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

ENTSOE_API_TOKEN = os.getenv("ENTSOE_API_TOKEN")

# Mapping areas to ENTSO-E EIC domains
DOMAIN_MAP = {
    'DK1': '10YDK-1--------W',
    'DK2': '10YDK-2--------M',
    'DE': '10Y1001A1001A82H', # Germany/Luxembourg
    'NO1': '10YNO-1--------2',
    'SE3': '10Y1001A1001A46L',
    'SE4': '10Y1001A1001A47J',
    'NL': '10YNL----------L',
    'UK': '10Y1001A1001A92E',
}

def fetch_entsoe_flows(domain_in, domain_out, period_start, period_end):
    if not ENTSOE_API_TOKEN:
        logger.warning("ENTSOE_API_TOKEN is missing. Returning empty string.")
        return ""
        
    url = "https://web-api.tp.entsoe.eu/api"
    params = {
        "securityToken": ENTSOE_API_TOKEN,
        "documentType": "A11",
        "in_Domain": DOMAIN_MAP.get(domain_in, domain_in),
        "out_Domain": DOMAIN_MAP.get(domain_out, domain_out),
        "periodStart": period_start,
        "periodEnd": period_end
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Error fetching ENTSO-E flows for {domain_in}->{domain_out}: {e}")
        return ""

def fetch_energinet_true_forecast_error(area="DK1", hours_back=24):
    """
    Fetches real Day-Ahead forecast vs actuals for Wind & Solar from Energinet.
    Replaces the Open-Meteo proxy.
    """
    url = "https://api.energidataservice.dk/dataset/Forecasts_Hour"
    # In a full implementation, you'd calculate exact time offsets.
    # For V4.0 scaffolding:
    params = {
        "limit": 100,
        "filter": f'{{"PriceArea":["{area}"]}}',
        "sort": "HourUTC DESC"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            records = data.get("records", [])
            df = pd.DataFrame(records)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching Energinet forecasts: {e}")
        return pd.DataFrame()

def fetch_energinet_system_frequency(limit=100):
    """
    Fetches real-time grid telemetry from Energinet PowerSystemRightNow.
    Returns per-minute data including:
      - All cable flows (DK1-DE, DK1-NL, DK1-GB, DK1-NO, DK1-SE, DK1-DK2, DK2-DE, DK2-SE)
      - aFRR activations for DK1 and DK2
      - Wind/Solar production
      - CO2 emissions
    """
    url = "https://api.energidataservice.dk/dataset/PowerSystemRightNow"
    params = {
        "limit": limit,
        "sort": "Minutes1UTC DESC"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data.get("records", []))
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching PowerSystemRightNow: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    print("Testing V4 API integrations...")
    # Test Energinet Forecasts
    df_err = fetch_energinet_true_forecast_error("DK1")
    print(f"Energinet Forecasts: {len(df_err)} records.")
    # Test Real-Time Grid Telemetry
    df_grid = fetch_energinet_system_frequency(5)
    print(f"Grid Telemetry: {len(df_grid)} records.")
    if not df_grid.empty:
        print(df_grid[['Minutes1UTC', 'Exchange_DK1_DE', 'Exchange_DK1_NL', 'Exchange_DK1_GB', 'aFRR_ActivatedDK1']].to_string())
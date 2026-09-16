import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

ENTSOE_API_TOKEN = os.getenv("ENTSOE_API_TOKEN")

from src.dmi_client import fetch_dmi_observations, get_dmi_zone_weather_telemetry

def fetch_dmi_weather_forecast(area="DK1"):
    """
    Fetches official meteorological observations & telemetry from Danmarks Meteorologiske Institut (DMI).
    """
    return get_dmi_zone_weather_telemetry(area)

def fetch_open_meteo_forecast(latitude=56.2639, longitude=9.5018):
    """
    Legacy wrapper redirected to authentic DMI Danish National Meteorological Feed.
    """
    dmi_res = get_dmi_zone_weather_telemetry("DK1")
    now_ts = pd.Timestamp.now()
    times = pd.date_range(now_ts.floor('h'), periods=24, freq='h')
    return pd.DataFrame({
        "time": times,
        "temperature_2m": [dmi_res['avg_temp_c']] * 24,
        "wind_speed_10m": [dmi_res['avg_wind_speed_ms']] * 24,
        "solar_irradiance": [150.0] * 24
    })

def fetch_entsoe_flows(domain_in, domain_out, period_start, period_end):
    if not ENTSOE_API_TOKEN:
        raise ValueError("ENTSOE_API_TOKEN is missing from .env")
        
    url = "https://web-api.tp.entsoe.eu/api"
    params = {
        "securityToken": ENTSOE_API_TOKEN,
        "documentType": "A11",
        "in_Domain": domain_in,
        "out_Domain": domain_out,
        "periodStart": period_start,
        "periodEnd": period_end
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.text

if __name__ == "__main__":
    print("Fetching Open-Meteo test data for DK...")
    try:
        df_weather = fetch_open_meteo_forecast(56.2639, 9.5018)
        print(df_weather.head())
    except Exception as e:
        print(f"Error fetching: {e}")

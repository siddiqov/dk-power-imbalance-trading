import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

ENTSOE_API_TOKEN = os.getenv("ENTSOE_API_TOKEN")

def fetch_open_meteo_forecast(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "wind_speed_10m", "direct_normal_irradiance"],
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature_2m": data["hourly"]["temperature_2m"],
        "wind_speed_10m": data["hourly"]["wind_speed_10m"],
        "solar_irradiance": data["hourly"]["direct_normal_irradiance"]
    })
    return df

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

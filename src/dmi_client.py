# ==============================================================================
# src/dmi_client.py
# Official DMI (Danmarks Meteorologiske Institut) Open Data API Client
# The Authoritative National Meteorological Authority for the Kingdom of Denmark
# ==============================================================================

import os
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

DMI_BASE_URL = "https://opendataapi.dmi.dk/v2/metObs"

# Representative key meteorological stations across Danish Bidding Zones
# DK1: West Denmark (Jutland & Funen + North Sea wind corridor)
# DK2: East Denmark (Zealand, Bornholm & Baltic wind corridor)
DK1_STATIONS = ['06056', '06030', '06080', '06041', '06051', '06058'] # Blaavand, Thyboroen, Esbjerg, Skagen, etc.
DK2_STATIONS = ['06180', '06184', '06190', '06170', '06188', '06193'] # Drogden, Copenhagen, Bornholm, Roenne, etc.

def fetch_dmi_observations(parameter_id="wind_speed", limit=100, station_id=None, period=None):
    """
    Fetches official meteorological observations directly from DMI Open Data API.

    Without `period="latest"` and a `station_id`, DMI returns an unfiltered slice of its
    whole archive - any station, any date (observed data going back decades) - so a plain
    limited query is not actually "live". Pass `period="latest"` with a specific `station_id`
    to get that station's most recent reading.

    Supported Parameters:
      - 'wind_speed' : Wind speed in m/s
      - 'wind_dir'   : Wind direction in degrees
      - 'temp_dry'   : Air temperature in Celsius
      - 'radia_glob' : Global solar irradiance in W/m2
    """
    url = f"{DMI_BASE_URL}/collections/observation/items"
    params = {
        "parameterId": parameter_id,
        "limit": limit
    }
    if station_id:
        params["stationId"] = station_id
    if period:
        params["period"] = period

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            features = data.get("features", [])
            records = []
            for f in features:
                props = f.get("properties", {})
                geom = f.get("geometry", {})
                coords = geom.get("coordinates", [None, None])
                records.append({
                    "observed_utc": props.get("observed"),
                    "parameter": props.get("parameterId"),
                    "value": props.get("value"),
                    "station_id": props.get("stationId"),
                    "lon": coords[0] if len(coords) > 0 else None,
                    "lat": coords[1] if len(coords) > 1 else None
                })
            df = pd.DataFrame(records)
            if not df.empty:
                df['observed_utc'] = pd.to_datetime(df['observed_utc'])
            return df
        else:
            logger.warning(f"DMI API returned status {response.status_code}: {response.text[:200]}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching DMI observations for {parameter_id}: {e}")
        return pd.DataFrame()

def _latest_regional_reading(parameter_id, target_stations, max_stations_tried=4):
    """Most recent value (`period=latest`) from the first responsive station in the list.

    An unfiltered query returns DMI's whole archive (any station, any date back to the
    1950s) - `period=latest` + a specific `station_id` is what actually makes this current.
    """
    for station_id in target_stations[:max_stations_tried]:
        df = fetch_dmi_observations(parameter_id=parameter_id, limit=1, station_id=station_id,
                                     period="latest")
        if not df.empty:
            return float(df.iloc[0]["value"]), station_id, df.iloc[0].get("observed_utc")
    return None, None, None


def get_dmi_zone_weather_telemetry(area="DK1"):
    """
    Returns the most recent DMI observation (Wind Speed, Temp) from a representative
    station for the specified Danish bidding zone (DK1 or DK2).
    """
    try:
        target_stations = DK1_STATIONS if area == "DK1" else DK2_STATIONS
        wind, wind_station, wind_obs = _latest_regional_reading("wind_speed", target_stations)
        temp, temp_station, temp_obs = _latest_regional_reading("temp_dry", target_stations)
        avg_wind = wind if wind is not None else 7.5    # Sensible Danish coastal default (m/s)
        avg_temp = temp if temp is not None else 12.0

        return {
            "source": "Danish Meteorological Institute (DMI)",
            "area": area,
            "avg_wind_speed_ms": round(float(avg_wind), 2),
            "avg_temp_c": round(float(avg_temp), 1),
            "station_id": wind_station or temp_station,
            "observed_utc": str(wind_obs or temp_obs) if (wind_obs or temp_obs) else None,
            "status": "Authentic DMI Live Feed Active" if (wind is not None or temp is not None)
                      else "Offline Fallback (no station responded)"
        }
    except Exception as e:
        logger.error(f"Error computing DMI zone weather: {e}")
        return {
            "source": "DMI (Fallback)",
            "area": area,
            "avg_wind_speed_ms": 7.5,
            "avg_temp_c": 12.0,
            "status": "Offline Cache"
        }

if __name__ == "__main__":
    print("Testing DMI Official Open Data Client...")
    res_dk1 = get_dmi_zone_weather_telemetry("DK1")
    print("DK1 DMI Weather Telemetry:", res_dk1)
    res_dk2 = get_dmi_zone_weather_telemetry("DK2")
    print("DK2 DMI Weather Telemetry:", res_dk2)

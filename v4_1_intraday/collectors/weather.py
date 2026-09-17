"""Weather forecasts from Open-Meteo's previous-runs API (free, no key; DMI/ECMWF blend "best_match").

Only `*_previous_day1` variables are stored: the value for hour H comes from a model run
issued at least one day before H. This is point-in-time safe for intraday decisions and is
exactly the same product in training and in live use.

Zone value = mean over a few representative points (onshore/offshore wind and load centres).
Hourly values are repeated on the four quarters of the hour.
"""
from __future__ import annotations

import logging

import pandas as pd

from .common import ensure_tables, http_get, windows

log = logging.getLogger("nurex41id.weather")
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARS = ["temperature_2m", "wind_speed_100m", "cloud_cover", "shortwave_radiation"]
POINTS = {
    "DK1": [(55.52, 7.90), (56.46, 9.40), (55.40, 10.40), (57.05, 9.92)],   # Horns Rev, mid-Jutland, Funen, Aalborg
    "DK2": [(55.68, 12.57), (55.02, 12.93), (55.10, 14.70), (55.45, 11.80)],  # Copenhagen, Kriegers Flak, Bornholm, Zealand
}
RUN = "d1"


def parse(payload, area: str) -> pd.DataFrame:
    locs = payload if isinstance(payload, list) else [payload]
    frames = []
    for loc in locs:
        h = loc.get("hourly") or {}
        if not h.get("time"):
            continue
        d = pd.DataFrame({"time_utc": pd.to_datetime(h["time"])})
        for v in VARS:
            col = f"{v}_previous_day1"
            if col in h:
                d[v] = pd.to_numeric(pd.Series(h[col]), errors="coerce")
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["area", "time_utc", "var", "run", "value"])
    m = pd.concat(frames).groupby("time_utc").mean(numeric_only=True)
    q = m.reindex(pd.date_range(m.index.min(), m.index.max() + pd.Timedelta(minutes=45), freq="15min")).ffill(limit=3)
    long = q.reset_index(names="time_utc").melt(id_vars="time_utc", var_name="var", value_name="value").dropna()
    long["area"] = area
    long["run"] = RUN
    return long[["area", "time_utc", "var", "run", "value"]]


def collect(store, start, end, window_days: int = 60, getter=None) -> int:
    ensure_tables(store)
    get = getter or (lambda params: http_get(URL, params=params).json())
    total = 0
    for area, pts in POINTS.items():
        for a, b in windows(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + pd.Timedelta(days=2),
                            window_days):
            params = {
                "latitude": ",".join(str(p[0]) for p in pts),
                "longitude": ",".join(str(p[1]) for p in pts),
                "hourly": ",".join(f"{v}_previous_day1" for v in VARS),
                "start_date": a.strftime("%Y-%m-%d"),
                "end_date": (b - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "timezone": "GMT",
                "wind_speed_unit": "ms",
            }
            try:
                df = parse(get(params), area)
            except Exception as e:
                log.warning("weather %s %s..%s: %s", area, a.date(), b.date(), e)
                continue
            if len(df):
                total += store.upsert("weather_fc", df)
            log.info("weather %s %s -> %s: %d rows", area, a.date(), b.date(), len(df))
    return total

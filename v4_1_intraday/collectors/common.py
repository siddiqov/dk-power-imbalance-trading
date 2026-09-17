"""Shared helpers: extra DuckDB tables, secrets, HTTP with retries."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

from .. import settings as S

log = logging.getLogger("nurex41id.collect")

EXTRA_SCHEMA = {
    # Nord Pool REMIT UMM (outages). One row per message version x asset x time period.
    "umm_events": """
        message_id VARCHAR, version INTEGER, publication_utc TIMESTAMP, message_type VARCHAR,
        unavailability_type VARCHAR, event_status VARCHAR, asset_kind VARCHAR, asset_name VARCHAR,
        area VARCHAR, area_eic VARCHAR, fuel VARCHAR, installed_mw DOUBLE, available_mw DOUBLE,
        unavailable_mw DOUBLE, event_start TIMESTAMP, event_stop TIMESTAMP, ingested_at TIMESTAMP,
        PRIMARY KEY (message_id, version, asset_kind, asset_name, event_start)""",
    # Weather forecasts (Open-Meteo previous-runs API): value for time_utc from a run issued >= 1 day before.
    "weather_fc": """
        area VARCHAR, time_utc TIMESTAMP, var VARCHAR, run VARCHAR, value DOUBLE, ingested_at TIMESTAMP,
        PRIMARY KEY (area, time_utc, var, run)""",
    # System frequency, 15-minute statistics of the 3-minute Fingrid series (Nordic synchronous area).
    "frequency": """
        area VARCHAR, time_utc TIMESTAMP, f_mean DOUBLE, f_std DOUBLE, f_min DOUBLE, f_max DOUBLE,
        n INTEGER, ingested_at TIMESTAMP,
        PRIMARY KEY (area, time_utc)""",
}


def ensure_tables(store) -> None:
    for name, cols in EXTRA_SCHEMA.items():
        store.con.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols})")


def secret(name: str, default: str | None = None) -> str | None:
    """Read from the environment, loading Nurex_V4_2/.env and Basic_Approach/.env first."""
    try:
        from dotenv import load_dotenv
        load_dotenv(S.V42_ROOT / ".env")
        load_dotenv(S.BASE / ".env")
    except ImportError:
        pass
    v = os.getenv(name)
    return v if v else default


def http_get(url: str, params=None, headers=None, timeout: int = 60, retries: int = 5,
             session: requests.Session | None = None) -> requests.Response:
    s = session or requests
    delay = 2.0
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = s.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                log.warning("%s -> %s, retry %d in %.0fs", url, last, attempt, delay)
            else:
                raise RuntimeError(f"{url} HTTP {r.status_code}: {r.text[:300]}")
        except requests.RequestException as e:
            last = str(e)
            log.warning("%s request error %s, retry %d in %.0fs", url, e, attempt, delay)
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise RuntimeError(f"{url} failed after {retries} attempts: {last}")


def utc_naive(x) -> pd.Series | pd.Timestamp:
    """Any timestamp(s) -> tz-naive UTC."""
    if isinstance(x, (pd.Series, pd.Index)):
        t = pd.to_datetime(x, utc=True)
        return t.dt.tz_localize(None) if isinstance(t, pd.Series) else t.tz_localize(None)
    t = pd.Timestamp(x)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.tz_localize(None)


def windows(start, end, days: int):
    cur = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def data_dir() -> Path:
    cfg = S.load()
    p = cfg.path("intraday_data_dir") if "intraday_data_dir" in cfg.raw else S.V42_ROOT / "data" / "nordpool_id"
    p.mkdir(parents=True, exist_ok=True)
    return p

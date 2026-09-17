"""Nordic system frequency from Fingrid Open Data (dataset 177, 3-minute values).

Needs a free API key (https://data.fingrid.fi -> register) in Nurex_V4_2/.env as FINGRID_API_KEY.
DK2 is in the Nordic synchronous area, so this is a direct DK2 signal. DK1 belongs to the
Continental European area (not covered here); DK1 relies on aFRR activation instead.
"""
from __future__ import annotations

import logging

import pandas as pd

from .common import ensure_tables, http_get, secret, utc_naive, windows

log = logging.getLogger("nurex41id.freq")
URL = "https://data.fingrid.fi/api/datasets/177/data"


def to_quarters(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["area", "time_utc", "f_mean", "f_std", "f_min", "f_max", "n"])
    t = utc_naive(raw["startTime"])
    v = pd.to_numeric(raw["value"], errors="coerce")
    d = pd.DataFrame({"time_utc": t.dt.floor("15min"), "v": v}).dropna()
    g = d.groupby("time_utc")["v"]
    out = pd.DataFrame({"f_mean": g.mean(), "f_std": g.std(), "f_min": g.min(), "f_max": g.max(),
                        "n": g.size()}).reset_index()
    out["area"] = "NORDIC"
    return out


def collect(store, start, end, window_days: int = 7, getter=None) -> int:
    ensure_tables(store)
    key = secret("FINGRID_API_KEY")
    if getter is None and not key:
        raise RuntimeError("FINGRID_API_KEY not set (free key from data.fingrid.fi) - frequency skipped")
    get = getter or (lambda params: http_get(URL, params=params, headers={"x-api-key": key}).json())
    total = 0
    for a, b in windows(pd.Timestamp(start), pd.Timestamp(end), window_days):
        rows, page = [], 1
        while True:
            data = get({"startTime": a.strftime("%Y-%m-%dT%H:%M:%SZ"), "endTime": b.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "format": "json", "pageSize": 20000, "page": page, "sortOrder": "asc"})
            items = data.get("data", [])
            rows.extend(items)
            pg = data.get("pagination") or {}
            if not items or not pg.get("nextPage"):
                break
            page = pg["nextPage"]
        df = to_quarters(pd.DataFrame(rows))
        if len(df):
            total += store.upsert("frequency", df)
        log.info("frequency %s -> %s: %d quarters", a.date(), b.date(), len(df))
    return total

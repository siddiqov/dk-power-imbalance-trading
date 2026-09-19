"""ENTSO-E Transparency Platform collector for V4.1 (needs ENTSOE_API_KEY in Nurex_V4_2/.env).

Writes to the V4.2 table `entsoe_series` (series, time_utc, value), 15-minute UTC grid:

  day-ahead information (known on D-1, see availability.entsoe_dayahead_publish_local)
    loadfc:<Z>            day-ahead total load forecast, MW               Z in DK1, DK2, DE_LU
    sched:<A>><N>         net day-ahead scheduled exchange A->N, MW (positive = export from A)
    ntc:<A>><N>           day-ahead net transfer capacity A->N, MW (where still published)
    wsfc:<Z>:<type>       day-ahead wind/solar forecast (Solar, Wind Onshore, Wind Offshore), MW

  realised information (known after availability.entsoe_actual_lag_minutes)
    load:<Z>              actual total load, MW
    phys:<A>><N>          net physical flow A->N, MW (positive = export from A)

Missing borders/series are logged and skipped - nothing is invented.
"""
from __future__ import annotations

import logging

import pandas as pd

from .common import secret, windows

log = logging.getLogger("nurex41id.entsoe")

ZONES = {"DK_1": "DK1", "DK_2": "DK2", "DE_LU": "DE_LU", "NL": "NL", "NO_2": "NO_2", "SE_3": "SE_3",
         "SE_4": "SE_4", "GB": "GB"}
BORDERS = {
    "DK_1": ["DE_LU", "NL", "NO_2", "SE_3", "DK_2", "GB"],
    "DK_2": ["DE_LU", "SE_4", "DK_1"],
}
LOAD_ZONES = ["DK_1", "DK_2", "DE_LU"]
WS_ZONES = ["DE_LU", "DK_1", "DK_2"]


def _keys() -> list[str]:
    from .. import settings as S
    keys = []
    for name in ("ENTSOE_API_KEY", "ENTSOE_TOKEN"):
        v = secret(name)
        if v and v not in keys:
            keys.append(v)
    # the root Basic_Approach/.env may hold a different token under the same name
    try:
        from dotenv import dotenv_values
        for f in (S.BASE / ".env", S.V42_ROOT / ".env"):
            for name in ("ENTSOE_API_KEY", "ENTSOE_TOKEN"):
                v = (dotenv_values(f) or {}).get(name)
                if v and v not in keys:
                    keys.append(v)
    except Exception:
        pass
    return keys


def client():
    """First ENTSO-E key that is accepted (tested with one day of DK1 load).
    Fails fast with a clear message instead of retrying every window with a refused key."""
    from entsoe import EntsoePandasClient
    keys = _keys()
    if not keys:
        raise RuntimeError("ENTSOE_API_KEY is not set in Nurex_V4_2/.env")
    errors = []
    end = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=2)
    for i, k in enumerate(keys, 1):
        c = EntsoePandasClient(api_key=k)
        try:
            c.query_load("DK_1", start=end - pd.Timedelta(days=1), end=end)
            log.info("ENTSO-E key #%d accepted", i)
            return c
        except Exception as e:
            msg = str(e)
            if "401" in msg or "Unauthorized" in msg:
                errors.append(f"key #{i} (...{k[-4:]}): 401 Unauthorized")
            else:                      # other errors (no data, network) -> key may be fine
                log.warning("ENTSO-E key #%d test query failed (%s) - using it anyway", i, msg[:120])
                return c
    raise RuntimeError("ENTSO-E refused every key: " + "; ".join(errors) +
                       ". Activate API access: log in at transparency.entsoe.eu, email transparency@entsoe.eu "
                       "with subject 'Restful API access' and your account e-mail, then generate a token under "
                       "My Account Settings -> Web API Security Token and put it in Nurex_V4_2/.env as ENTSOE_API_KEY.")


def to_15min(s: pd.Series) -> pd.Series:
    """tz-aware series of any resolution -> tz-naive UTC 15-minute series (hourly values repeated)."""
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(s).astype(float).copy()
    idx = pd.DatetimeIndex(s.index)
    idx = idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx
    s.index = idx
    s = s[~s.index.duplicated(keep="last")].sort_index()
    step = s.index.to_series().diff().median() if len(s) > 1 else pd.Timedelta(minutes=15)
    if pd.notna(step) and step > pd.Timedelta(minutes=15):
        full = pd.date_range(s.index.min(), s.index.max() + step - pd.Timedelta(minutes=15), freq="15min")
        s = s.reindex(full).ffill(limit=int(step / pd.Timedelta(minutes=15)) - 1)
    return s


def _frame(name: str, s: pd.Series) -> pd.DataFrame:
    s = to_15min(s).dropna()
    return pd.DataFrame({"series": name, "time_utc": s.index, "value": s.values})


def _first_col(x) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def _net(c, fn, a, n, start, end, **kw) -> pd.Series:
    out = to_15min(fn(a, n, start=start, end=end, **kw))
    back = to_15min(fn(n, a, start=start, end=end, **kw))
    return out.sub(back, fill_value=0.0)


def fetch_window(c, start: pd.Timestamp, end: pd.Timestamp, parts=None) -> pd.DataFrame:
    parts = set(parts or ["load", "loadfc", "sched", "phys", "ntc", "wsfc"])
    frames = []

    def attempt(label, f):
        try:
            r = f()
            if r is not None and len(r):
                frames.append(r)
        except Exception as e:  # NoMatchingDataError etc.
            if "401" in str(e) or "Unauthorized" in str(e):
                raise RuntimeError(f"ENTSO-E key refused (401) during {label}") from e
            log.warning("ENTSO-E %s %s..%s: %s", label, start.date(), end.date(), str(e)[:160])

    for z in LOAD_ZONES:
        if "load" in parts:
            attempt(f"load {z}", lambda z=z: _frame(f"load:{ZONES[z]}", _first_col(c.query_load(z, start=start, end=end))))
        if "loadfc" in parts:
            attempt(f"loadfc {z}", lambda z=z: _frame(f"loadfc:{ZONES[z]}",
                                                       _first_col(c.query_load_forecast(z, start=start, end=end))))
    for a, ns in BORDERS.items():
        for n in ns:
            tag = f"{ZONES[a]}>{ZONES[n]}"
            if "sched" in parts:
                attempt(f"sched {tag}", lambda a=a, n=n, tag=tag: _frame(
                    f"sched:{tag}", _net(c, c.query_scheduled_exchanges, a, n, start, end, dayahead=True)))
            if "phys" in parts:
                attempt(f"phys {tag}", lambda a=a, n=n, tag=tag: _frame(
                    f"phys:{tag}", _net(c, c.query_crossborder_flows, a, n, start, end)))
            if "ntc" in parts:
                attempt(f"ntc {tag}", lambda a=a, n=n, tag=tag: _frame(
                    f"ntc:{tag}", c.query_net_transfer_capacity_dayahead(a, n, start=start, end=end)))
    if "wsfc" in parts:
        for z in WS_ZONES:
            def ws(z=z):
                df = c.query_wind_and_solar_forecast(z, start=start, end=end)
                df = df if isinstance(df, pd.DataFrame) else df.to_frame()
                return pd.concat([_frame(f"wsfc:{ZONES[z]}:{col}", df[col]) for col in df.columns],
                                 ignore_index=True)
            attempt(f"wsfc {z}", ws)
    if not frames:
        return pd.DataFrame(columns=["series", "time_utc", "value"])
    return pd.concat(frames, ignore_index=True)


def collect(store, start, end, window_days: int = 14, parts=None, c=None) -> int:
    from .common import ensure_tables
    ensure_tables(store)
    c = c or client()
    total = 0
    for a, b in windows(pd.Timestamp(start), pd.Timestamp(end), window_days):
        df = fetch_window(c, pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"), parts)
        if len(df):
            df = df.drop_duplicates(["series", "time_utc"], keep="last")
            total += store.upsert("entsoe_series", df)
        log.info("ENTSO-E %s -> %s: %d rows", a.date(), b.date(), len(df))
    return total

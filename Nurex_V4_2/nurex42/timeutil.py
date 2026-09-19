"""Time helpers: UTC/local conversion, client batch schedule and zones.

All timestamps inside the system are tz-naive UTC (pandas datetime64[ns]).
Local (Europe/Copenhagen) time is only used to define days, batches and
publication times.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

QUARTER = pd.Timedelta(minutes=15)

# Zones of the rolling plan
ZONE_SETTLED = "SETTLED"        # delivered and imbalance price published
ZONE_LOCKED = "LOCKED"          # batch submitted; immutable
ZONE_NEXT = "NEXT_BATCH"        # the next batch to be submitted
ZONE_DYNAMIC = "DYNAMIC"        # provisional, re-optimised on new data


def utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def to_utc_naive(ts, tz: str = "UTC") -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(tz)
    return t.tz_convert("UTC").tz_localize(None)


def to_local(ts_utc, tz: str) -> pd.Timestamp:
    return pd.Timestamp(ts_utc).tz_localize("UTC").tz_convert(tz)


def local_time_on_day(day_local: pd.Timestamp | str, hhmm: str, tz: str) -> pd.Timestamp:
    """UTC-naive timestamp of HH:MM local time on a given local date."""
    d = pd.Timestamp(day_local).date()
    h, m = (int(x) for x in hhmm.split(":"))
    loc = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=h, minute=m).tz_localize(
        tz, ambiguous=True, nonexistent="shift_forward")
    return loc.tz_convert("UTC").tz_localize(None)


def local_day_quarters(day_local, tz: str) -> pd.DatetimeIndex:
    """All delivery quarters (UTC-naive starts) of a local day (92/96/100 on DST days)."""
    d = pd.Timestamp(day_local).normalize()
    start = d.tz_localize(tz)
    end = (d + pd.Timedelta(days=1)).tz_localize(tz)
    idx = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="15min", inclusive="left")
    return idx.tz_localize(None)


def batch_start(quarters_utc, tz: str, quarters_per_batch: int = 8) -> pd.Series:
    """Start (UTC-naive) of the client batch containing each quarter.

    Batches are fixed blocks of local wall-clock time: 00:00-01:45, 02:00-03:45, ...
    """
    q = pd.to_datetime(pd.Series(quarters_utc)).reset_index(drop=True)
    loc = q.dt.tz_localize("UTC").dt.tz_convert(tz)
    block_h = quarters_per_batch // 4
    minutes_into_block = (loc.dt.hour % block_h) * 60 + loc.dt.minute
    return q - pd.to_timedelta(minutes_into_block, unit="min")


def batch_deadline(batch_start_utc, lead_minutes: int):
    return pd.to_datetime(batch_start_utc) - pd.Timedelta(minutes=lead_minutes)


def decision_time(batch_start_utc, lead_minutes: int, final_run_minutes: int):
    """When the final forecast for a batch is computed (information cut-off)."""
    return batch_deadline(batch_start_utc, lead_minutes) - pd.Timedelta(minutes=final_run_minutes)


def provisional_as_of(delivery_day_local, cfg) -> pd.Timestamp:
    """Cut-off of the D+1 provisional plan: run_time_local on the day before delivery."""
    tz = cfg["local_tz"]
    prev = pd.Timestamp(delivery_day_local).normalize() - pd.Timedelta(days=1)
    return local_time_on_day(prev, cfg["provisional"]["run_time_local"], tz)


def assign_zone(quarter_utc: pd.Timestamp, bstart_utc: pd.Timestamp, now_utc: pd.Timestamp,
                is_locked: bool, is_settled: bool, cfg) -> str:
    if is_settled:
        return ZONE_SETTLED
    if is_locked:
        return ZONE_LOCKED
    lead = cfg["batch"]["lead_minutes"]
    deadline = bstart_utc - pd.Timedelta(minutes=lead)
    # the next batch is the earliest one whose deadline has not passed
    nxt = next_batch_start(now_utc, cfg)
    if bstart_utc == nxt and now_utc < deadline:
        return ZONE_NEXT
    return ZONE_DYNAMIC


def next_batch_start(now_utc: pd.Timestamp, cfg) -> pd.Timestamp:
    """Earliest batch start whose submission deadline is still in the future."""
    tz = cfg["local_tz"]
    lead = pd.Timedelta(minutes=cfg["batch"]["lead_minutes"])
    q = pd.Timestamp(now_utc).floor("15min")
    for _ in range(4 * 24 * 3):
        bs = batch_start([q], tz, cfg["batch"]["quarters_per_batch"]).iloc[0]
        if bs >= q and bs - lead > now_utc:
            return bs
        q += QUARTER
    raise RuntimeError("could not find next batch")


def daily_batch_table(day_local, cfg) -> pd.DataFrame:
    """Schedule of the client's batches for one local delivery day."""
    tz = cfg["local_tz"]
    qs = local_day_quarters(day_local, tz)
    bs = batch_start(qs, tz, cfg["batch"]["quarters_per_batch"])
    df = pd.DataFrame({"quarter_utc": qs, "batch_start_utc": bs.values})
    out = df.groupby("batch_start_utc").agg(first_q=("quarter_utc", "min"), last_q=("quarter_utc", "max"),
                                            n_quarters=("quarter_utc", "size")).reset_index()
    out["deadline_utc"] = batch_deadline(out["batch_start_utc"], cfg["batch"]["lead_minutes"])
    for c in ["batch_start_utc", "last_q", "deadline_utc"]:
        out[c + "_local"] = out[c].dt.tz_localize("UTC").dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M")
    return out

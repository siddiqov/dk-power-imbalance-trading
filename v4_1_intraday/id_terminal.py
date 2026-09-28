"""Live Nord Pool intraday (XBID) market table for the dashboard, read from the recorder files
(Nurex_V4_2/data/nordpool_id, written by `train_v4_1.py record-intraday`). 2026-09-28.

Only recorded market data is shown - nothing is estimated. A product without recorded data gets
empty (NaN) cells, never a made-up price.

latest_market(area, day_local) -> one row per product (QH 15-min and PH 60-min contracts) that
delivers on that local day:
  product, duration_min, dlvry_start_utc, dlvry_start_local,
  bid, bid_mw, ask, ask_mw, depth_bid_mw, depth_ask_mw   (last recorded order-book state)
  dam, vwap, last, high, low, turnover_mw                  (last recorded contract statistics)
  book_time_utc, stats_time_utc                            (when those values were received)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .collectors import nordpool_id as npid


def _latest(df: pd.DataFrame, aid: int) -> pd.DataFrame:
    if df is None or df.empty or "contract_id" not in df.columns:
        return pd.DataFrame()
    d = df[(df["area_id"] == aid)].dropna(subset=["contract_id"])
    if "deleted" in d.columns:
        d = d[d["deleted"] != True]  # noqa: E712
    return d.sort_values("recv_utc").groupby("contract_id").tail(1).set_index("contract_id")


def latest_market(area: str, day_local, tz: str = "Europe/Copenhagen", price_div: float = 100.0,
                  qty_div: float = 1000.0, root=None) -> pd.DataFrame:
    day = pd.Timestamp(day_local).normalize()
    lo, hi = day - pd.Timedelta(days=1), day + pd.Timedelta(days=1)
    t = {k: npid.load(k, start=lo, end=hi, root=root) for k in ("areas", "contracts", "book", "stats")}
    areas, con = t["areas"], t["contracts"]
    if areas.empty or con.empty:
        return pd.DataFrame()
    ids = areas.dropna(subset=["dk"])
    ids = ids[ids["dk"] == area]["area_id"].dropna().astype(int).unique()
    if len(ids) == 0:
        return pd.DataFrame()
    aid = int(ids[0])

    def _has(s):
        try:
            lst = json.loads(s)
            return len(lst) == 0 or aid in lst
        except Exception:
            return False

    con = con.dropna(subset=["contract_id", "dlvry_start", "dlvry_end"]).drop_duplicates("contract_id", keep="last")
    con = con[con["area_ids"].astype(str).apply(_has)].copy()
    con["duration_min"] = ((con["dlvry_end"] - con["dlvry_start"]).dt.total_seconds() / 60).round().astype(int)
    con = con[con["duration_min"].isin([15, 60])]
    loc = con["dlvry_start"].dt.tz_localize("UTC").dt.tz_convert(tz)
    con = con[(loc.dt.tz_localize(None).dt.normalize() == day).values]
    if con.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"product": con["name"].values, "duration_min": con["duration_min"].values,
                        "dlvry_start_utc": con["dlvry_start"].values,
                        "dlvry_start_local": con["dlvry_start"].dt.tz_localize("UTC").dt.tz_convert(tz)
                        .dt.tz_localize(None).values}, index=con["contract_id"].values)
    b = _latest(t["book"], aid)
    if len(b):
        b = b.reindex(out.index)
        out["bid"] = pd.to_numeric(b["best_bid"], errors="coerce") / price_div
        out["ask"] = pd.to_numeric(b["best_ask"], errors="coerce") / price_div
        out["bid_mw"] = pd.to_numeric(b["bid_qty_best"], errors="coerce") / qty_div
        out["ask_mw"] = pd.to_numeric(b["ask_qty_best"], errors="coerce") / qty_div
        out["depth_bid_mw"] = pd.to_numeric(b["bid_qty_depth"], errors="coerce") / qty_div
        out["depth_ask_mw"] = pd.to_numeric(b["ask_qty_depth"], errors="coerce") / qty_div
        out["book_time_utc"] = b["recv_utc"]
    s = _latest(t["stats"], aid)
    if len(s):
        s = s.reindex(out.index)
        for src, dst in (("da_price", "dam"), ("vwap", "vwap"), ("last_price", "last"), ("high", "high"), ("low", "low")):
            out[dst] = pd.to_numeric(s[src], errors="coerce") / price_div
        out["turnover_mw"] = pd.to_numeric(s["turnover"], errors="coerce") / qty_div
        out["stats_time_utc"] = s["recv_utc"]
    for c in ("bid", "ask", "bid_mw", "ask_mw", "depth_bid_mw", "depth_ask_mw", "dam", "vwap", "last", "high",
              "low", "turnover_mw", "book_time_utc", "stats_time_utc"):
        if c not in out.columns:
            out[c] = np.nan
    # an empty side of the book is a real state: no price, 0 MW (not a made-up price)
    return out.sort_values(["duration_min", "dlvry_start_utc"]).reset_index(drop=True)

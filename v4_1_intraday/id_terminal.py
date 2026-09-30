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
Each product has an XBID (NX_) and a Nord Pool local (NI_) contract; they are merged into one row.
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
    out["contract_id"] = out.index.astype(str)
    out = _one_row_per_product(out)
    # an empty side of the book is a real state: no price, 0 MW (not a made-up price)
    return out.sort_values(["duration_min", "dlvry_start_utc"]).reset_index(drop=True)


def _one_row_per_product(out: pd.DataFrame) -> pd.DataFrame:
    """Nord Pool lists each product twice for a Danish area: the cross-border XBID contract (NX_...)
    and the Nord Pool local contract (NI_...), each with its own order book and trades. 2026-09-29:
    merge them into one row per product from the recorded values only -
      bid = highest bid of the two books, ask = lowest ask (with that side's MW),
      depth / turnover = sum, vwap = turnover-weighted, high = max, low = min,
      dam / last = from the contract with the larger turnover (normally XBID)."""
    rows = []
    for (prod, dur), g in out.groupby(["product", "duration_min"], sort=False):
        r = g.iloc[0].to_dict()
        gb = g.dropna(subset=["bid"])
        if len(gb):
            k = gb["bid"].idxmax()
            r["bid"], r["bid_mw"] = gb.at[k, "bid"], gb.at[k, "bid_mw"]
        else:
            r["bid"], r["bid_mw"] = np.nan, np.nan
        ga = g.dropna(subset=["ask"])
        if len(ga):
            k = ga["ask"].idxmin()
            r["ask"], r["ask_mw"] = ga.at[k, "ask"], ga.at[k, "ask_mw"]
        else:
            r["ask"], r["ask_mw"] = np.nan, np.nan
        for c in ("depth_bid_mw", "depth_ask_mw"):
            r[c] = g[c].sum(min_count=1)
        to = pd.to_numeric(g["turnover_mw"], errors="coerce")
        r["turnover_mw"] = to.sum(min_count=1)
        w = g.dropna(subset=["vwap"])
        tw = pd.to_numeric(w["turnover_mw"], errors="coerce").fillna(0)
        if len(w) and tw.sum() > 0:
            r["vwap"] = float((w["vwap"] * tw).sum() / tw.sum())
        else:
            r["vwap"] = w["vwap"].iloc[0] if len(w) else np.nan
        r["high"] = g["high"].max()
        r["low"] = g["low"].min()
        main = g.loc[to.fillna(-1).idxmax()] if to.notna().any() else g.iloc[0]
        r["dam"] = main["dam"] if pd.notna(main["dam"]) else g["dam"].dropna().iloc[0] if g["dam"].notna().any() else np.nan
        r["last"] = main["last"]
        for c in ("book_time_utc", "stats_time_utc"):
            r[c] = pd.to_datetime(g[c]).max()
        r["contract_id"] = ",".join(g["contract_id"])
        rows.append(r)
    return pd.DataFrame(rows)

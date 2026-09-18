"""Glue between dashboard_v4_1.py (Streamlit) and the V4.1 intraday engine.

The dashboard only displays; every V4.1 number comes from here:
  - decisions are made per quarter as of gate closure (or as of now for future quarters)
  - settlement uses the published imbalance price, negative prices included
  - PnL = +/- spread * MWh - MWh * cost (cost from config_v41.yaml)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import settings as S
from . import pipeline_id as P
from .features_id import IntradayFeatureBuilder
from nurex42 import timeutil as tu
from nurex42.storage import Store

_CACHE: dict = {}


def config():
    if "cfg" not in _CACHE:
        _CACHE["cfg"] = S.load()
    return _CACHE["cfg"]


def cost_per_mwh() -> float:
    return config().cost_per_mwh


def load_bundle(area: str):
    path = P.model_path(config(), area)
    if not path.exists():
        return None
    return P.IntradayBundle.load(path)


def model_info(area: str) -> dict | None:
    cfg = config()
    p = P.model_path(cfg, area).with_suffix(".json")
    info = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    rep_dir = cfg.path("reports_dir")
    sums = sorted(rep_dir.glob("replay_summary_*.json")) if rep_dir.exists() else []
    if info is not None and sums:
        s = json.loads(sums[-1].read_text(encoding="utf-8")).get(area)
        if s:
            info["replay"] = {k: s["trading"].get(k) for k in
                              ("trades", "mwh_traded", "net_eur", "net_eur_per_mwh", "net_eur_at_stress_costs",
                               "max_drawdown_eur", "sharpe_daily_ann")}
            info["replay_file"] = sums[-1].name
    return info


def _builder(max_age_s: int = 300):
    """Feature builder cached for a few minutes. If the store is locked by a running update,
    the previous builder is reused (DuckDB allows one writer at a time)."""
    import time
    now = pd.Timestamp.now()
    fb = _CACHE.get("fb")
    if fb is not None and (now - _CACHE["fb_at"]).total_seconds() <= max_age_s:
        return fb
    last = None
    for _ in range(3):
        try:
            st = Store(config().db_path, read_only=True)
            try:
                fb = IntradayFeatureBuilder(st, config())
            finally:
                st.close()
            _CACHE["fb"], _CACHE["fb_at"] = fb, now
            return fb
        except Exception as e:  # database busy
            last = e
            time.sleep(2)
    if _CACHE.get("fb") is not None:
        return _CACHE["fb"]
    raise RuntimeError(f"V4.2 store busy: {last}")


def day_decisions(area: str, date_str: str, now=None) -> pd.DataFrame:
    """One row per local delivery quarter of `date_str`, keyed by local 'YYYY-MM-DD HH:MM'."""
    cfg = config()
    b = load_bundle(area)
    if b is None:
        return pd.DataFrame()
    qs = tu.local_day_quarters(date_str, cfg["local_tz"])
    out = P.predict_quarters(None, cfg, b, qs, now=now, fb=_builder())
    out["time_dk_str"] = (out["quarter_utc"].dt.tz_localize("UTC").dt.tz_convert(cfg["local_tz"])
                          .dt.strftime("%Y-%m-%d %H:%M"))
    c = cfg.cost_per_mwh
    s = np.where(out["action"] == "BUY", 1.0, np.where(out["action"] == "SELL", -1.0, 0.0))
    settled = out["spread_actual"].notna()
    out["pnl_eur"] = np.where(settled, s * out["mwh"] * out["spread_actual"] - np.where(s != 0, out["mwh"] * c, 0.0),
                              np.nan)
    out["settled"] = settled
    out["source"] = "recomputed"
    # locked paper-trading decisions override the recomputed ones
    try:
        from . import journal as J
        j = J.read(cfg, area=area, start=qs.min(), end=qs.max() + pd.Timedelta(minutes=15))
    except Exception:
        j = pd.DataFrame()
    if len(j):
        j = j.set_index("quarter_utc")
        idx = out["quarter_utc"].isin(j.index)
        k = out.loc[idx, "quarter_utc"]
        for col in ("action", "mwh", "edge", "exp_spread", "p_up", "p_flat", "p_down", "q10", "q50", "q90", "reason"):
            if col in j.columns:
                vals = k.map(j[col])
                keep = vals.notna()
                out.loc[vals[keep].index, col] = vals[keep]
        out.loc[idx, "source"] = k.map(j["status"]).values
        out.loc[idx, "decision_final"] = True
        jp = k.map(j["pnl_eur"])
        out.loc[jp[jp.notna()].index, "pnl_eur"] = jp[jp.notna()]
        out["mwh"] = out["mwh"].astype(float)
    return out


def sources_status() -> pd.DataFrame:
    """Coverage of every data source (for the sidebar)."""
    import io
    import contextlib
    import types
    from train_v4_1 import cmd_sources   # reuse the CLI table
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df = cmd_sources(types.SimpleNamespace(), config())
    df = df.rename(columns={"n": "rows"})
    df["last"] = pd.to_datetime(df["last"]).dt.strftime("%Y-%m-%d %H:%M")
    return df[["source", "rows", "last"]]


# Border label shown in the ledger, per zone, in display order.
FLOW_BORDERS = {
    "DK1": [("DE_LU", "DE"), ("NO_2", "NO2"), ("SE_3", "SE3"), ("NL", "NL"), ("GB", "GB"), ("DK_2", "DK2")],
    "DK2": [("DE_LU", "DE"), ("SE_4", "SE4"), ("DK_1", "DK1")],
}


def day_flows(area: str, date_str: str) -> pd.DataFrame:
    """Per-quarter cross-border exchange for one local day, from ENTSO-E rows in the store.

    Columns per border: 'sched_<label>' (day-ahead scheduled exchange, MW, positive = export
    from `area`) and 'dev_<label>' (realised physical flow minus schedule, MW, where published).
    Authentic values only: a quarter with no published row stays NaN - nothing is filled in.
    Returns an empty frame when the store is busy, so the ledger still renders.
    """
    cfg = config()
    qs = tu.local_day_quarters(date_str, cfg["local_tz"])
    if len(qs) == 0:
        return pd.DataFrame()
    borders = FLOW_BORDERS.get(area, [])
    if not borders:
        return pd.DataFrame()
    names = [f"{p}:{area}>{b}" for b, _ in borders for p in ("sched", "phys")]
    try:
        st = Store(cfg.db_path, read_only=True)
        try:
            raw = st.df(
                "SELECT series, time_utc, value FROM entsoe_series "
                "WHERE time_utc >= ? AND time_utc <= ? AND series IN ("
                + ",".join("?" * len(names)) + ")",
                [qs.min(), qs.max()] + names)
        finally:
            st.close()
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    wide = raw.pivot_table(index="time_utc", columns="series", values="value", aggfunc="last")
    wide = wide.reindex(pd.DatetimeIndex(qs))
    out = pd.DataFrame(index=wide.index)
    for b, label in borders:
        sc, ph = f"sched:{area}>{b}", f"phys:{area}>{b}"
        if sc in wide.columns:
            out[f"sched_{label}"] = wide[sc].values
            if ph in wide.columns:
                out[f"dev_{label}"] = wide[ph].values - wide[sc].values
    if out.empty:
        return pd.DataFrame()
    out["time_dk_str"] = (out.index.tz_localize("UTC").tz_convert(cfg["local_tz"])
                          .strftime("%Y-%m-%d %H:%M"))
    return out.reset_index(drop=True)

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
import time as _time
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


# What-if threshold levels for the dashboard. 'validated' uses the thresholds chosen on the
# validation window during training; the others loosen them and are NOT validated - they exist
# to show what the same forecasts would have done at a lower bar for taking a trade.
THRESHOLD_LEVELS = {
    "validated": None,
    "balanced": (0.5, 0.03),    # (margin multiplier, pmin reduction)
    "aggressive": (0.25, 0.06),
}
PMIN_FLOOR = 0.35


def decision_params(area: str, level: str = "validated") -> dict | None:
    """Trading thresholds for `level`, or None to use the model's own validated ones.

    A side the training switched off stays off: turning it back on would trade on a signal that
    showed no validated edge at all, which is a different thing from lowering a working bar.
    """
    factors = THRESHOLD_LEVELS.get(level)
    if factors is None:
        return None
    b = load_bundle(area)
    if b is None or not b.decision_params:
        return None
    base = b.decision_params
    mult, drop = factors
    out = {k: v for k, v in base.items() if k not in ("buy", "sell")}
    for side in ("buy", "sell"):
        s = base.get(side)
        out[side] = None if not s else {
            "margin": round(float(s["margin"]) * mult, 2),
            "pmin": max(PMIN_FLOOR, round(float(s["pmin"]) - drop, 3)),
        }
    out["no_trade"] = not (out.get("buy") or out.get("sell"))
    return out


def day_decisions(area: str, date_str: str, now=None, params: dict | None = None) -> pd.DataFrame:
    """One row per local delivery quarter of `date_str`, keyed by local 'YYYY-MM-DD HH:MM'.

    `params` overrides the model's validated thresholds (a what-if view). The paper-trading
    journal is then left alone: locked decisions were taken at the real thresholds, so pasting
    them over a what-if run would mix the two.
    """
    cfg = config()
    b = load_bundle(area)
    if b is None:
        return pd.DataFrame()
    qs = tu.local_day_quarters(date_str, cfg["local_tz"])
    out = P.predict_quarters(None, cfg, b, qs, now=now, fb=_builder())
    if params is not None:
        from nurex42 import decision as dec
        d = dec.decide(out, out["quarter_utc"], b.model.stress, cfg, params)
        for col in ("action", "mwh", "edge", "reason"):
            out[col] = d[col].values

    out["time_dk_str"] = (out["quarter_utc"].dt.tz_localize("UTC").dt.tz_convert(cfg["local_tz"])
                          .dt.strftime("%Y-%m-%d %H:%M"))
    c = cfg.cost_per_mwh
    s = np.where(out["action"] == "BUY", 1.0, np.where(out["action"] == "SELL", -1.0, 0.0))
    settled = out["spread_actual"].notna()
    out["pnl_eur"] = np.where(settled, s * out["mwh"] * out["spread_actual"] - np.where(s != 0, out["mwh"] * c, 0.0),
                              np.nan)
    out["settled"] = settled
    out["source"] = "recomputed" if params is None else "what-if"
    if params is not None:
        return out
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

    The V4.2 store allows a single writer, so a collector run locks out readers. Like _builder(),
    this retries and then falls back to the last good result for the same day rather than
    dropping the flow columns out of the ledger. The reason is reported in df.attrs['status']:
    'ok', 'stale' (cached, store was busy), 'busy' (no data and no cache) or 'unpublished'.
    """
    cfg = config()
    qs = tu.local_day_quarters(date_str, cfg["local_tz"])
    if len(qs) == 0:
        return _flows_status(pd.DataFrame(), "unpublished")
    borders = FLOW_BORDERS.get(area, [])
    if not borders:
        return _flows_status(pd.DataFrame(), "unpublished")

    # Accept the legacy series spelling too: zone codes were written without the underscore
    # (DK1>DK2) before the rename, so older days would otherwise show a blank column.
    variants = {}
    for b, label in borders:
        variants[b] = [b] + [v for v in (b.replace("_", ""),) if v != b]
    names = [f"{p}:{area}>{v}" for b, _ in borders for v in variants[b] for p in ("sched", "phys")]

    raw, last = None, None
    for attempt in range(3):
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
            break
        except Exception as e:  # database busy: one writer at a time
            last = e
            raw = None
            if attempt < 2:
                _time.sleep(2)

    if raw is None:  # never got a read in
        cached = _CACHE.get(("flows", area, date_str))
        if cached is not None:
            return _flows_status(cached.copy(), "stale", last)
        return _flows_status(pd.DataFrame(), "busy", last)

    if raw.empty:
        return _flows_status(pd.DataFrame(), "unpublished")

    wide = raw.pivot_table(index="time_utc", columns="series", values="value", aggfunc="last")
    wide = wide.reindex(pd.DatetimeIndex(qs))

    def _series(prefix, border):
        """First variant of this border actually present, else None."""
        for v in variants[border]:
            col = f"{prefix}:{area}>{v}"
            if col in wide.columns and wide[col].notna().any():
                return wide[col]
        return None

    out = pd.DataFrame(index=wide.index)
    for b, label in borders:
        sc = _series("sched", b)
        if sc is None:
            continue
        out[f"sched_{label}"] = sc.values
        ph = _series("phys", b)
        if ph is not None:
            out[f"dev_{label}"] = ph.values - sc.values
    if out.empty:
        return _flows_status(pd.DataFrame(), "unpublished")

    out["time_dk_str"] = (out.index.tz_localize("UTC").tz_convert(cfg["local_tz"])
                          .strftime("%Y-%m-%d %H:%M"))
    out = out.reset_index(drop=True)
    _CACHE[("flows", area, date_str)] = out.copy()
    return _flows_status(out, "ok")


def _flows_status(df: pd.DataFrame, status: str, err=None) -> pd.DataFrame:
    """Tag a flow frame so the dashboard can say why columns are missing."""
    df.attrs["status"] = status
    df.attrs["error"] = None if err is None else str(err)
    return df

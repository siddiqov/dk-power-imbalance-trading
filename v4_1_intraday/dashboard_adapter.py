"""Glue between dashboard_v4_1.py (Streamlit) and the V4.1 intraday engine.

The dashboard only displays; every V4.1 number comes from here:
  - decisions are made per quarter as of gate closure (or as of now for future quarters)
  - settlement uses the published imbalance price, negative prices included
  - PnL = +/- spread * MWh - MWh * cost (cost from config_v41.yaml)
"""
from __future__ import annotations

import copy
import json
import os
import shutil
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


def _cfg_with_risk_override(risk_overrides: dict | None = None):
    """config(), or a copy with `risk` fields overridden for a what-if sizing view.

    Used for the "bigger trade sizes" checkbox: a bigger `batch_risk_fraction` sizes new
    decisions larger, without touching the shared cached config (other tabs, and the
    scheduled paper-trading cycle, which runs as its own process, are unaffected) and
    without touching already-locked/settled quarters, which keep the size they were
    actually decided at.
    """
    cfg = config()
    if not risk_overrides:
        return cfg
    raw = copy.deepcopy(cfg.raw)
    raw["risk"].update(risk_overrides)
    return type(cfg)(raw)


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


# --- Read-only access that survives the scheduled cycle ----------------------------------
# The paper-trading cycle (train_v4_1.py cycle, every 15 min) writes nurex42.duckdb for a few
# minutes per run. DuckDB lets no other process open the file while a writer holds it, so the
# dashboard used to go blank ("HOLD") during every write. Whenever the dashboard does get a
# read-only connection (so no writer is active and the file is consistent) it refreshes a copy,
# nurex42.snapshot.duckdb, at most every _SNAPSHOT_MAX_AGE_S seconds. When the live file is
# busy, reads fall back to that copy. The snapshot is only ever read by the dashboard; the
# cycle and the journal keep using the live store.
_SNAPSHOT_MAX_AGE_S = 600


def _snapshot_path(cfg=None) -> Path:
    p = Path((cfg or config()).db_path)
    return p.with_name(p.stem + ".snapshot" + p.suffix)


def _refresh_snapshot(cfg) -> None:
    """Copy the live store to the snapshot. Call only while holding a read-only connection."""
    snap = _snapshot_path(cfg)
    try:
        if snap.exists() and (_time.time() - snap.stat().st_mtime) < _SNAPSHOT_MAX_AGE_S:
            return
        src = Path(cfg.db_path)
        tmp = snap.with_name(snap.name + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, snap)
        wal, snap_wal = Path(str(src) + ".wal"), Path(str(snap) + ".wal")
        if wal.exists():
            shutil.copyfile(wal, snap_wal)
        elif snap_wal.exists():
            snap_wal.unlink()
    except Exception:
        pass  # best effort: never fail a read that already succeeded


def _open_ro(cfg, attempts: int = 3):
    """Read-only Store on the live file if it is free, else on the latest snapshot.
    Returns (store, source) with source "live" or "snapshot"."""
    last = None
    for i in range(attempts):
        try:
            st = Store(cfg.db_path, read_only=True)
            _refresh_snapshot(cfg)
            return st, "live"
        except Exception as e:  # database busy: one writer at a time
            last = e
            if i < attempts - 1:
                _time.sleep(2)
    snap = _snapshot_path(cfg)
    if snap.exists():
        try:
            return Store(snap, read_only=True), "snapshot"
        except Exception as e:
            last = e
    raise RuntimeError(f"V4.2 store busy: {last}")


def _builder(max_age_s: int = 300):
    """Feature builder cached for a few minutes. Reads the live store, or the snapshot while a
    running update holds it; if neither can be opened the previous builder is reused."""
    now = pd.Timestamp.now()
    fb = _CACHE.get("fb")
    if fb is not None and (now - _CACHE["fb_at"]).total_seconds() <= max_age_s:
        return fb
    try:
        st, src = _open_ro(config())
    except RuntimeError:
        if _CACHE.get("fb") is not None:
            return _CACHE["fb"]
        raise
    try:
        fb = IntradayFeatureBuilder(st, config())
    finally:
        st.close()
    # a snapshot-based builder is kept for one minute only, so the live data comes back quickly
    at = now if src == "live" else now - pd.Timedelta(seconds=max(0, max_age_s - 60))
    _CACHE["fb"], _CACHE["fb_at"], _CACHE["fb_src"] = fb, at, src
    return fb


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


def day_decisions(area: str, date_str: str, now=None, params: dict | None = None,
                  risk_overrides: dict | None = None) -> pd.DataFrame:
    """One row per local delivery quarter of `date_str`, keyed by local 'YYYY-MM-DD HH:MM'.

    `params` overrides the model's validated thresholds (a what-if view). `risk_overrides`
    overrides `risk` config fields (e.g. a bigger `batch_risk_fraction` for bigger trade
    sizes) for this call only. The paper-trading journal is then left alone: locked decisions
    were taken at the real thresholds and real sizing, so pasting them over a what-if run
    would mix the two.
    """
    cfg = _cfg_with_risk_override(risk_overrides)
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
                          .dt.strftime("%Y-%m-%d %H:%M%z"))
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


def _predictions(area: str, date_str: str, now=None, max_age_s: int = 900, cfg=None):
    """Point-in-time forecasts for one local day, cached briefly.

    Forecasts do not depend on the threshold level, so every level and the risk overlay reuse
    one pass. Cached only when `now` is not pinned AND `cfg` is the default config - a
    what-if `cfg` (bigger trade sizes) always computes fresh so it can never be served
    stale sizes from a previous, differently-configured call.
    """
    override = cfg is not None
    cfg = cfg or config()
    key = ("pred", area, date_str)
    if now is None and not override:
        hit = _CACHE.get(key)
        if hit is not None and (pd.Timestamp.now() - hit[0]).total_seconds() <= max_age_s:
            return hit[1].copy()
    b = load_bundle(area)
    if b is None:
        return pd.DataFrame()
    qs = tu.local_day_quarters(date_str, cfg["local_tz"])
    out = P.predict_quarters(None, cfg, b, qs, now=now, fb=_builder())
    out["time_dk_str"] = (out["quarter_utc"].dt.tz_localize("UTC").dt.tz_convert(cfg["local_tz"])
                          .dt.strftime("%Y-%m-%d %H:%M%z"))
    if now is None and not override:
        _CACHE[key] = (pd.Timestamp.now(), out.copy())
    return out


def _score(d: pd.DataFrame, area: str, level: str, cfg=None):
    """Apply one threshold level's decisions to a copy of the forecasts."""
    from nurex42 import decision as dec
    cfg = cfg or config()
    b = load_bundle(area)
    d = d.copy()
    params = decision_params(area, level)
    if params is not None:                      # 'validated' keeps the model's own decisions
        dd = dec.decide(d, d["quarter_utc"], b.model.stress, cfg, params)
        for col in ("action", "mwh", "edge", "reason"):
            d[col] = dd[col].values
    d["level"] = level
    return d


def _pnl(d: pd.DataFrame, cost: float) -> pd.DataFrame:
    s = np.where(d["action"] == "BUY", 1.0, np.where(d["action"] == "SELL", -1.0, 0.0))
    settled = d["spread_actual"].notna()
    d["pnl_eur"] = np.where(settled, s * d["mwh"] * d["spread_actual"]
                            - np.where(s != 0, d["mwh"] * cost, 0.0), np.nan)
    d["settled"] = settled
    return d


def day_decisions_risked(area: str, date_str: str, levels=("validated",), now=None,
                        risk_overrides: dict | None = None) -> dict:
    """Decisions for one local day with the SAME risk overlay the walk-forward replay applies.

    Until this existed, `apply_risk_overlays` ran only inside the replay: the backtest stopped
    trading after a bad day while the live view kept going, so the two were not the same system.

    The overlay is path dependent - it needs the equity curve in decision order - so the run
    starts `drawdown_window_days` before the target day and returns only the target day's rows.
    Each level gets its own equity path, because a level that loses faster hits the stop sooner.

    A quarter whose imbalance price is due but missing from the store contributes 0 PnL to that
    equity path (its costs still count). Run backfill_imbalance to keep those gaps rare.
    """
    cfg = _cfg_with_risk_override(risk_overrides)
    if load_bundle(area) is None:
        return {}
    win = int(float(cfg["risk"].get("drawdown_window_days", 7)))
    lag = pd.Timedelta(minutes=float(cfg["availability"]["imbalance_lag_minutes"]))
    target = pd.Timestamp(date_str).normalize()
    days = [(target - pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in range(win, -1, -1)]

    base = [_predictions(area, d, now=now, cfg=cfg) for d in days]
    base = [b for b in base if not b.empty]
    if not base:
        return {}
    base = pd.concat(base, ignore_index=True)

    cost = cfg.cost_per_mwh
    out = {}
    for lvl in levels:
        d = _score(base, area, lvl, cfg=cfg)
        d["as_of_trade"] = d["as_of_utc"]
        d["label_known_at"] = d["quarter_utc"] + pd.Timedelta(minutes=15) + lag
        # A missing label must not poison the equity path with NaN; it contributes no PnL.
        d["spread"] = d["spread_actual"].astype(float).fillna(0.0)
        r = P.apply_risk_overlays(d, cfg)
        r = r[r["time_dk_str"].str.startswith(date_str)].copy()
        r = _pnl(r, cost)
        r["source"] = "risk-managed"
        r["level"] = lvl
        out[lvl] = r.reset_index(drop=True)
    return out


def day_decisions_multi(area: str, date_str: str, levels=("validated", "balanced", "aggressive"),
                       now=None) -> dict:
    """The same forecasts for one local day, scored under several threshold levels.

    The model runs once: every level shares identical predictions and differs only in the bar a
    quarter must clear to be traded, so this is one prediction pass plus N cheap decision passes.

    Returns {level: DataFrame}. The journal is never applied - locked decisions were taken at the
    validated thresholds, so applying them to one level only would make the comparison unfair.
    Every level here is therefore 'recomputed', which is the like-for-like basis a tournament needs.
    """
    from nurex42 import decision as dec

    cfg = config()
    b = load_bundle(area)
    if b is None:
        return {}
    qs = tu.local_day_quarters(date_str, cfg["local_tz"])
    base = P.predict_quarters(None, cfg, b, qs, now=now, fb=_builder())
    base["time_dk_str"] = (base["quarter_utc"].dt.tz_localize("UTC").dt.tz_convert(cfg["local_tz"])
                           .dt.strftime("%Y-%m-%d %H:%M%z"))
    c = cfg.cost_per_mwh

    out = {}
    for lvl in levels:
        d = base.copy()
        params = decision_params(area, lvl)
        if params is not None:                      # 'validated' keeps the model's own decisions
            dd = dec.decide(d, d["quarter_utc"], b.model.stress, cfg, params)
            for col in ("action", "mwh", "edge", "reason"):
                d[col] = dd[col].values
        s = np.where(d["action"] == "BUY", 1.0, np.where(d["action"] == "SELL", -1.0, 0.0))
        settled = d["spread_actual"].notna()
        d["pnl_eur"] = np.where(settled,
                                s * d["mwh"] * d["spread_actual"] - np.where(s != 0, d["mwh"] * c, 0.0),
                                np.nan)
        d["settled"] = settled
        d["level"] = lvl
        d["source"] = "recomputed"
        out[lvl] = d
    return out


def trained_until(area: str):
    """When the model's training data ends - days after this are out of sample."""
    b = load_bundle(area)
    return None if b is None else pd.Timestamp(b.trained_until)


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
    df["last"] = pd.to_datetime(df["last"]).dt.strftime("%Y-%m-%d %H:%M%z")
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
    try:
        st, _src = _open_ro(cfg)  # live store, or the snapshot while the cycle is writing
        try:
            raw = st.df(
                "SELECT series, time_utc, value FROM entsoe_series "
                "WHERE time_utc >= ? AND time_utc <= ? AND series IN ("
                + ",".join("?" * len(names)) + ")",
                [qs.min(), qs.max()] + names)
        finally:
            st.close()
    except Exception as e:  # neither the live store nor a snapshot could be read
        last = e
        raw = None

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
                          .strftime("%Y-%m-%d %H:%M%z"))
    out = out.reset_index(drop=True)
    _CACHE[("flows", area, date_str)] = out.copy()
    return _flows_status(out, "ok")


def _flows_status(df: pd.DataFrame, status: str, err=None) -> pd.DataFrame:
    """Tag a flow frame so the dashboard can say why columns are missing."""
    df.attrs["status"] = status
    df.attrs["error"] = None if err is None else str(err)
    return df


def day_dayahead(areas, date_str: str) -> pd.DataFrame:
    """Day-ahead prices (EUR/MWh) per local delivery quarter of `date_str`, from the V4.2 store.

    One column per area (e.g. 'DE', 'DK1'), plus 'time_dk_str' keyed like day_flows. Authentic
    values only: a quarter with no published price stays NaN - nothing is copied or filled in.
    The reason for an empty frame is in df.attrs['status'] ('ok', 'stale', 'busy', 'unpublished').
    """
    cfg = config()
    areas = [areas] if isinstance(areas, str) else list(areas)
    qs = tu.local_day_quarters(date_str, cfg["local_tz"])
    ck = ("da", tuple(areas), date_str)
    if len(qs) == 0 or not areas:
        return _flows_status(pd.DataFrame(), "unpublished")
    try:
        st, _src = _open_ro(cfg)
        try:
            raw = st.df("SELECT area, time_utc, price_eur FROM dayahead WHERE time_utc >= ? AND "
                        "time_utc <= ? AND area IN (" + ",".join("?" * len(areas)) + ")",
                        [qs.min(), qs.max()] + areas)
        finally:
            st.close()
    except Exception as e:
        cached = _CACHE.get(ck)
        if cached is not None:
            return _flows_status(cached.copy(), "stale", e)
        return _flows_status(pd.DataFrame(), "busy", e)
    if raw.empty:
        return _flows_status(pd.DataFrame(), "unpublished")
    wide = raw.pivot_table(index="time_utc", columns="area", values="price_eur", aggfunc="last")
    wide = wide.reindex(pd.DatetimeIndex(qs))
    out = pd.DataFrame({a: (wide[a].values if a in wide.columns else np.nan) for a in areas},
                       index=wide.index)
    out["time_dk_str"] = (out.index.tz_localize("UTC").tz_convert(cfg["local_tz"])
                          .strftime("%Y-%m-%d %H:%M%z"))
    out = out.reset_index(drop=True)
    _CACHE[ck] = out.copy()
    return _flows_status(out, "ok")


def day_scaffold(area: str, date_str: str) -> pd.DataFrame:
    """96-quarter ledger frame for one local day built only from authentic stored data.

    Columns: quarter, time_dk ('YYYY-MM-DD HH:MM' local), time_utc, spot_price_eur (DA auction),
    actual_settled_imbalance_eur (Energinet imbalance price, NaN until published), status.
    Used when the legacy tournament table cannot be built. Empty if the day has no DA prices.
    """
    cfg = config()
    qs = tu.local_day_quarters(date_str, cfg["local_tz"])
    if len(qs) == 0:
        return pd.DataFrame()
    st, _src = _open_ro(cfg)
    try:
        da = st.df("SELECT time_utc, price_eur FROM dayahead WHERE area = ? AND time_utc >= ? AND time_utc <= ?",
                   [area, qs.min(), qs.max()])
        im = st.df("SELECT time_utc, imbalance_eur FROM imbalance WHERE area = ? AND time_utc >= ? AND time_utc <= ?",
                   [area, qs.min(), qs.max()])
    finally:
        st.close()
    if da.empty:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(qs)
    spot = da.drop_duplicates("time_utc").set_index("time_utc")["price_eur"].reindex(idx)
    imb = (im.drop_duplicates("time_utc").set_index("time_utc")["imbalance_eur"].reindex(idx)
           if not im.empty else pd.Series(np.nan, index=idx))
    local = idx.tz_localize("UTC").tz_convert(cfg["local_tz"])
    out = pd.DataFrame({
        "time_dk": local.strftime("%Y-%m-%d %H:%M"),
        "time_utc": idx.strftime("%Y-%m-%d %H:%M"),
        "spot_price_eur": spot.values,
        "actual_settled_imbalance_eur": imb.values,
    })
    out = out[out["spot_price_eur"].notna()].drop_duplicates("time_dk").reset_index(drop=True)
    out["status"] = np.where(out["actual_settled_imbalance_eur"].notna(), "Settled", "Pending")
    out.insert(0, "quarter", [f"Q{i + 1}" for i in range(len(out))])
    return out

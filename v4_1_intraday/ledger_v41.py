"""One-day V4.1 ledger for the V4.1-only dashboard (dashboard_v41_live.py).

Built only from the V4.1 engine and the V4.2 store - no V2/V3/V4.0 code:
  * quarters, DK day-ahead spot, published imbalance price  -> dashboard_adapter.day_scaffold
  * DE day-ahead spot                                       -> dashboard_adapter.day_dayahead
  * scheduled cross-border exchange per border              -> dashboard_adapter.day_flows
  * locked decisions (the honest record)                    -> journal.read
  * recomputed decisions / live forecasts                   -> dashboard_adapter.day_decisions_risked

view="locked" (default): each quarter shows its journal decision (LOCKED or MISSED). A quarter
    not locked yet shows the current forecast marked FORECAST, with no PnL. A past quarter
    with no journal row (before the journal started) shows NO RECORD.
view="recomputed": every quarter is re-decided by today's model (in-sample for past days).
Sizes are MWh per 15-min quarter (MW = MWh x 4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import dashboard_adapter as A
from . import journal as J

DEC_COLS = ["action", "mwh", "edge", "exp_spread", "p_up", "p_flat", "p_down", "q10", "q50", "q90", "reason"]


def _key(df: pd.DataFrame) -> pd.Series:
    """Local wall-clock minute 'YYYY-MM-DD HH:MM' (adapter strings carry a +hhmm offset)."""
    return df["time_dk_str"].astype(str).str.slice(0, 16)


def _by_key(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(_k=_key(df).values).drop_duplicates("_k").set_index("_k")


def day_ledger(area: str, date_str: str, view: str = "locked") -> tuple[pd.DataFrame, dict]:
    cfg = A.config()
    tz = cfg["local_tz"]
    info = {"flows": "n/a", "de": "n/a", "recomputed": "n/a"}

    base = A.day_scaffold(area, date_str)
    if base.empty:
        return base, info
    base["quarter_utc"] = pd.to_datetime(base["time_utc"])
    k = base["time_dk"]

    # DE spot
    try:
        de = A.day_dayahead("DE", date_str)
        info["de"] = de.attrs.get("status", "ok")
        base["de_spot_eur"] = k.map(_by_key(de)["DE"]).astype(float) if not de.empty else np.nan
    except Exception as e:
        info["de"] = f"error: {e}"
        base["de_spot_eur"] = np.nan

    # flows (scheduled exchange, + = export from area)
    borders = [lbl for _, lbl in A.FLOW_BORDERS.get(area, [])]
    try:
        fl = A.day_flows(area, date_str)
        info["flows"] = fl.attrs.get("status", "ok")
        m = _by_key(fl) if not fl.empty else pd.DataFrame()
    except Exception as e:
        info["flows"] = f"error: {e}"
        m = pd.DataFrame()
    have = []
    for lbl in borders:
        c = f"sched_{lbl}"
        base[f"flow_{lbl}"] = k.map(m[c]).astype(float) if c in m.columns else np.nan
        if c in m.columns:
            have.append(f"flow_{lbl}")
    base["net_flow_mw"] = base[have].sum(axis=1, min_count=1) if have else np.nan

    # recomputed / live forecast (validated thresholds + the replay's risk overlay)
    try:
        rc = A.day_decisions_risked(area, date_str, levels=("validated",)).get("validated", pd.DataFrame())
        info["recomputed"] = "ok" if not rc.empty else "empty"
    except Exception as e:
        rc = pd.DataFrame()
        info["recomputed"] = f"error: {e}"
    rcm = _by_key(rc) if not rc.empty else pd.DataFrame()

    # journal (locked record)
    qs = base["quarter_utc"]
    j = J.read(cfg, area=area, start=qs.min(), end=qs.max() + pd.Timedelta(minutes=15))
    jm = j.drop_duplicates("quarter_utc").set_index("quarter_utc") if not j.empty else pd.DataFrame()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    lead = cfg.lead

    rows = []
    for i, r in base.iterrows():
        q = r["quarter_utc"]
        rec = {}
        if view == "locked" and len(jm) and q in jm.index:
            jr = jm.loc[q]
            rec = {c: jr.get(c) for c in DEC_COLS}
            rec["source"] = str(jr.get("status"))           # LOCKED / MISSED
            rec["spread_actual"] = jr.get("spread_actual")
            rec["pnl_eur"] = jr.get("pnl_eur")
        else:
            kk = r["time_dk"]
            if len(rcm) and kk in rcm.index:
                rr = rcm.loc[kk]
                rec = {c: rr.get(c) for c in DEC_COLS}
            else:
                rec = {c: np.nan for c in DEC_COLS}
                rec["action"] = "HOLD"
            gate_passed = (q - lead) <= now
            if view == "locked":
                rec["source"] = "NO RECORD" if gate_passed else "FORECAST"
                rec["pnl_eur"] = np.nan        # only locked decisions earn PnL in this view
                rec["spread_actual"] = np.nan
            else:
                rec["source"] = "RECOMPUTED" if gate_passed else "FORECAST"
                rec["spread_actual"] = rcm.loc[kk].get("spread_actual") if len(rcm) and kk in rcm.index else np.nan
                rec["pnl_eur"] = rcm.loc[kk].get("pnl_eur") if len(rcm) and kk in rcm.index else np.nan
        rows.append(rec)
    dec = pd.DataFrame(rows, index=base.index)
    out = pd.concat([base, dec], axis=1)
    out["mwh"] = pd.to_numeric(out["mwh"], errors="coerce").fillna(0.0)
    out["mw"] = out["mwh"] * 4.0
    # settled actual spread from the store when the decision row lacks it
    store_spread = out["actual_settled_imbalance_eur"] - out["spot_price_eur"]
    out["spread_actual"] = pd.to_numeric(out["spread_actual"], errors="coerce")
    out.loc[out["spread_actual"].isna() & out["source"].isin(["LOCKED", "MISSED", "RECOMPUTED"]),
            "spread_actual"] = store_spread
    out["pred_imb_eur"] = out["spot_price_eur"] + pd.to_numeric(out["exp_spread"], errors="coerce")
    return out, info


def journal_daily(area: str | None = None, days: int = 30) -> pd.DataFrame:
    """Settled journal PnL per local day and area (locked decisions only)."""
    cfg = A.config()
    start = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=days)
    j = J.read(cfg, area=area, start=start)
    if j.empty:
        return pd.DataFrame()
    j = j[j["pnl_eur"].notna()].copy()
    if j.empty:
        return pd.DataFrame()
    j["day"] = j["quarter_utc"].dt.tz_localize("UTC").dt.tz_convert(cfg["local_tz"]).dt.date
    j["traded"] = j["action"] != "HOLD"
    g = j.groupby(["day", "area"]).agg(net_eur=("pnl_eur", "sum"), trades=("traded", "sum"),
                                       mwh=("mwh", lambda s: float(s[j.loc[s.index, "traded"]].sum())))
    return g.reset_index()


def status(area: str) -> dict:
    """Freshness and risk state for the status bar."""
    cfg = A.config()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    out = {"now_utc": now}
    try:
        st, src = A._open_ro(cfg)
        try:
            r = st.df("SELECT max(time_utc) AS t FROM imbalance WHERE area = ? AND imbalance_eur IS NOT NULL", [area])
        finally:
            st.close()
        last = pd.Timestamp(r["t"].iloc[0]) if len(r) and pd.notna(r["t"].iloc[0]) else None
        out["store_source"] = src
    except Exception as e:
        last, out["store_source"] = None, f"unavailable ({e})"
    out["last_imbalance_utc"] = last
    # age at the next decision time, measured like the stale-data guard (quarter start -> decision)
    out["imb_age_min"] = (now - last).total_seconds() / 60.0 if last is not None else None
    out["guard_min"] = float(cfg["intraday"].get("stale_data_guard_minutes") or 90)
    try:
        con = J.connect(cfg)
        try:
            mult, why = J._live_risk_state(con, area, cfg, now)
            lr = con.execute("SELECT max(locked_at_utc) FROM decisions WHERE area=?", (area,)).fetchone()[0]
        finally:
            con.close()
        out["risk_mult"], out["risk_reason"] = mult, why
        out["last_lock_utc"] = pd.Timestamp(lr) if lr else None
    except Exception as e:
        out["risk_mult"], out["risk_reason"], out["last_lock_utc"] = None, f"unavailable ({e})", None
    return out

"""Locked decision journal for V4.1 intraday paper trading (SQLite, safe for concurrent readers).

Why: the dashboard can recompute any day with the current model, but a retrained model
changes history. The journal stores each quarter's decision ONCE, before its gate closure,
with the information and model available at that moment. Performance is judged on this
journal only.

  lock(store, cfg)    decide every quarter whose gate closes within `lock_ahead_minutes`
                      (as of now, i.e. before the gate); quarters whose gate passed without a
                      lock are written as MISSED/HOLD (never decided with later information)
  settle(store, cfg)  add the published imbalance spread and PnL to locked quarters
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import itertools

import numpy as np
import pandas as pd

from . import pipeline_id as P
from .features_id import IntradayFeatureBuilder

log = logging.getLogger("nurex41id.journal")
Q = pd.Timedelta(minutes=15)

COLS = ["area", "quarter_utc", "gate_utc", "locked_at_utc", "as_of_utc", "status", "model_version",
        "model_trained_until", "action", "mwh", "mw", "edge", "exp_spread", "p_up", "p_flat", "p_down",
        "q10", "q50", "q90", "reason", "hour_all_same_side", "spread_actual", "pnl_eur", "cost_eur_mwh",
        "settled_at_utc"]

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS decisions (
    area TEXT NOT NULL, quarter_utc TEXT NOT NULL, gate_utc TEXT, locked_at_utc TEXT, as_of_utc TEXT,
    status TEXT, model_version TEXT, model_trained_until TEXT, action TEXT, mwh REAL, mw REAL, edge REAL,
    exp_spread REAL, p_up REAL, p_flat REAL, p_down REAL, q10 REAL, q50 REAL, q90 REAL, reason TEXT,
    hour_all_same_side INTEGER, spread_actual REAL, pnl_eur REAL, cost_eur_mwh REAL, settled_at_utc TEXT,
    PRIMARY KEY (area, quarter_utc));
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def path(cfg) -> Path:
    return cfg.path("journal_path") if "journal_path" in cfg.raw else cfg.path("models_dir").parent / "data" / "v41_journal.sqlite"


def connect(cfg) -> sqlite3.Connection:
    p = path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=30)
    con.executescript(SCHEMA)
    return con


def _ts(x) -> str | None:
    return None if x is None or pd.isna(x) else pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S")


def journal_start(con) -> pd.Timestamp | None:
    r = con.execute("SELECT value FROM meta WHERE key='journal_start_utc'").fetchone()
    return pd.Timestamp(r[0]) if r else None


def read(cfg, area: str | None = None, start=None, end=None) -> pd.DataFrame:
    p = path(cfg)
    if not p.exists():
        return pd.DataFrame(columns=COLS)
    con = sqlite3.connect(str(p), timeout=30)
    try:
        sql, args = "SELECT * FROM decisions WHERE 1=1", []
        if area:
            sql += " AND area=?"
            args.append(area)
        if start is not None:
            sql += " AND quarter_utc>=?"
            args.append(_ts(start))
        if end is not None:
            sql += " AND quarter_utc<?"
            args.append(_ts(end))
        df = pd.read_sql_query(sql + " ORDER BY area, quarter_utc", con, params=args)
    finally:
        con.close()
    for c in ("quarter_utc", "gate_utc", "locked_at_utc", "as_of_utc", "settled_at_utc"):
        df[c] = pd.to_datetime(df[c])
    return df


def _hour_agreement(d: pd.DataFrame, tz: str) -> pd.Series:
    """1 if all four quarters of the local hour carry the same non-HOLD action (hourly-product candidate)."""
    h = d["quarter_utc"].dt.tz_localize("UTC").dt.tz_convert(tz).dt.floor("h")
    g = d.assign(h=h.values).groupby("h")["action"]
    same = g.transform(lambda s: int(len(s) == 4 and s.nunique() == 1 and s.iloc[0] != "HOLD"))
    return same.astype(int)



def _live_risk_state(con, area: str, cfg, now: pd.Timestamp) -> tuple:
    """Return (size_multiplier, reason) from settled journal PnL history.

    Reads settled rows in the drawdown window and computes:
      - daily loss stop (sum of today's settled PnL vs daily_loss_stop_fraction * collateral)
      - drawdown guard (cumulative peak-to-trough over drawdown_window_days)

    Returns (1.0, None) when no history yet (safe default = full size).
    """
    r = cfg["risk"]
    coll = cfg.collateral_eur
    daily_stop_thresh = -float(r["daily_loss_stop_fraction"]) * coll
    half_thresh = float(r["drawdown_half_fraction"]) * coll
    halt_thresh = float(r["drawdown_stop_fraction"]) * coll
    win_days = int(float(r.get("drawdown_window_days", 7)))
    tz = cfg["local_tz"]

    horizon_ts = _ts(now - pd.Timedelta(days=win_days))
    try:
        df = pd.read_sql_query(
            "SELECT quarter_utc, pnl_eur FROM decisions "
            "WHERE area=? AND pnl_eur IS NOT NULL AND quarter_utc>=? ORDER BY quarter_utc",
            con, params=[area, horizon_ts])
    except Exception:
        return 1.0, None
    if df.empty:
        return 1.0, None

    pnls = df["pnl_eur"].astype(float).tolist()
    cum = list(itertools.accumulate(pnls, initial=0.0))
    peak = max(cum)
    equity = cum[-1]
    drawdown = max(0.0, peak - equity)

    today_local = pd.Timestamp(now).tz_localize("UTC").tz_convert(tz).date()
    today_pnl = df.loc[
        pd.to_datetime(df["quarter_utc"]).dt.tz_localize("UTC").dt.tz_convert(tz).dt.date == today_local,
        "pnl_eur"
    ].astype(float).sum()

    if today_pnl <= daily_stop_thresh:
        return 0.0, "daily loss stop"
    if drawdown >= halt_thresh:
        return 0.0, "drawdown halt"
    if drawdown >= half_thresh:
        return 0.5, "drawdown guard"
    return 1.0, None


def _apply_live_risk(rows: list, con, area: str, cfg, now: pd.Timestamp) -> list:
    """Scale or zero-out live lock rows according to the current risk state."""
    mult, reason = _live_risk_state(con, area, cfg, now)
    if mult == 1.0:
        return rows
    r = cfg["risk"]
    step = float(r["mwh_step"])
    mn = float(r["min_mwh_per_quarter"])
    for row in rows:
        if row.get("action") in ("BUY", "SELL"):
            new_mwh = np.floor(float(row["mwh"]) * mult / step) * step if mult > 0 else 0.0
            row["mwh"] = new_mwh if new_mwh + 1e-9 >= mn else 0.0
            row["mw"] = row["mwh"] * 4.0
            if row["mwh"] <= 0:
                row["action"] = "HOLD"
                row["reason"] = reason
    log.info("[%s] live risk overlay: mult=%.1f reason=%s applied to %d rows",
             area, mult, reason, len(rows))
    return rows

def _apply_hourly_cap(rows: list, con, area: str, cfg, tz: str) -> list:
    """Enforce max_mwh_per_hour: cap the aggregate MWh across all four quarters of a delivery
    hour.  Reads already-locked (non-HOLD) rows from the journal, then scales down new rows
    proportionally when the combined total would exceed the cap.

    FIX (audit #6): without this, up to 4 open quarters in one hour accumulate unbounded
    hourly exposure; the per-quarter cap alone does not limit the combined risk.
    """
    from collections import defaultdict
    r = cfg["risk"]
    cap = float(r.get("max_mwh_per_hour", float(r["max_mwh_per_quarter"]) * 2.0))
    step = float(r["mwh_step"])
    mn = float(r["min_mwh_per_quarter"])

    # Group new LOCKED trade rows by delivery hour
    hour_rows: dict = defaultdict(list)
    for row in rows:
        if row.get("status") == "LOCKED" and row.get("action") in ("BUY", "SELL"):
            q = pd.Timestamp(row["quarter_utc"]).tz_localize("UTC").tz_convert(tz).floor("h")
            hour_rows[q].append(row)

    for hour, hrows in hour_rows.items():
        h_start = hour.tz_convert("UTC").tz_localize(None)
        h_end = h_start + pd.Timedelta(hours=1)
        try:
            db = pd.read_sql_query(
                "SELECT COALESCE(SUM(mwh), 0.0) AS ex FROM decisions "
                "WHERE area=? AND quarter_utc>=? AND quarter_utc<? AND action != 'HOLD'",
                con, params=[area, _ts(h_start), _ts(h_end)])
            existing_mwh = float(db["ex"].iloc[0])
        except Exception:
            existing_mwh = 0.0

        new_mwh = sum(float(row["mwh"]) for row in hrows)
        if existing_mwh + new_mwh <= cap + 1e-9:
            continue  # within cap, no action needed

        allowed = max(0.0, cap - existing_mwh)
        log.warning("[%s] hourly cap %.1f MWh/h applied: existing=%.1f new=%.1f allowed=%.1f",
                    area, cap, existing_mwh, new_mwh, allowed)
        if allowed < mn:
            for row in hrows:
                row["action"] = "HOLD"; row["mwh"] = 0.0; row["mw"] = 0.0
                row["reason"] = "hourly exposure cap"
        else:
            scale = allowed / new_mwh
            for row in hrows:
                snapped = np.floor(float(row["mwh"]) * scale / step) * step
                row["mwh"] = snapped if snapped + 1e-9 >= mn else 0.0
                row["mw"] = row["mwh"] * 4.0
                if row["mwh"] <= 0:
                    row["action"] = "HOLD"; row["reason"] = "hourly exposure cap"
    return rows


def lock(store, cfg, now=None, fb=None, bundles=None) -> dict:
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    ahead = pd.Timedelta(minutes=int(cfg["intraday"].get("lock_ahead_minutes", 20)))
    lead = cfg.lead
    con = connect(cfg)
    out = {}
    try:
        start = journal_start(con)
        if start is None:
            start = now
            con.execute("INSERT INTO meta VALUES ('journal_start_utc', ?)", (_ts(now),))
            con.commit()
        fb = fb or IntradayFeatureBuilder(store, cfg)
        for area in cfg["areas"]:
            b = (bundles or {}).get(area)
            if b is None:
                mp = P.model_path(cfg, area)
                if not mp.exists():
                    log.warning("[%s] no model - run `train_v4_1.py train`", area)
                    continue
                b = P.IntradayBundle.load(mp)
            done = {r[0] for r in con.execute(
                "SELECT quarter_utc FROM decisions WHERE area=? AND quarter_utc>=?", (area, _ts(start)))}
            # candidate quarters: gate in [start, now + ahead]
            q0 = (start + lead).ceil("15min")
            q1 = (now + ahead + lead).floor("15min")
            qs = pd.date_range(q0, q1, freq="15min")
            qs = [q for q in qs if _ts(q) not in done]
            future = [q for q in qs if q - lead > now]
            missed = [q for q in qs if q - lead <= now]
            rows = []
            for q in missed:
                rows.append({"area": area, "quarter_utc": _ts(q), "gate_utc": _ts(q - lead), "locked_at_utc": _ts(now),
                             "as_of_utc": None, "status": "MISSED", "model_version": b.version,
                             "model_trained_until": _ts(b.trained_until), "action": "HOLD", "mwh": 0.0, "mw": 0.0,
                             "reason": "gate passed before the system ran", "cost_eur_mwh": cfg.cost_per_mwh})
            if future:
                # decide the whole local hour(s) so the hourly agreement flag can be computed
                tz = cfg["local_tz"]
                hours = {pd.Timestamp(q).tz_localize("UTC").tz_convert(tz).floor("h") for q in future}
                allq = sorted({h.tz_convert("UTC").tz_localize(None) + k * Q for h in hours for k in range(4)}
                              | set(future))
                pr = P.predict_quarters(store, cfg, b, pd.Series(allq), now=now, fb=fb)
                pr["hour_all_same_side"] = _hour_agreement(pr, tz).values
                pr = pr[pr["quarter_utc"].isin(future)]
                for _, r in pr.iterrows():
                    rows.append({"area": area, "quarter_utc": _ts(r["quarter_utc"]), "gate_utc": _ts(r["gate_utc"]),
                                 "locked_at_utc": _ts(now), "as_of_utc": _ts(r["as_of_utc"]), "status": "LOCKED",
                                 "model_version": b.version, "model_trained_until": _ts(b.trained_until),
                                 "action": r["action"], "mwh": float(r["mwh"]), "mw": float(r["mwh"]) * 4.0,
                                 "edge": float(r["edge"]), "exp_spread": float(r["exp_spread"]),
                                 "p_up": float(r["p_up"]), "p_flat": float(r["p_flat"]), "p_down": float(r["p_down"]),
                                 "q10": float(r["q10"]), "q50": float(r["q50"]), "q90": float(r["q90"]),
                                 "reason": r["reason"], "hour_all_same_side": int(r["hour_all_same_side"]),
                                 "cost_eur_mwh": cfg.cost_per_mwh})
            # Apply live risk overlay (daily stop + drawdown guard) to sized rows
            rows = _apply_live_risk(rows, con, area, cfg, now)
            rows = _apply_hourly_cap(rows, con, area, cfg, cfg["local_tz"])
            for row in rows:
                keys = [k for k in COLS if k in row]
                con.execute(f"INSERT OR IGNORE INTO decisions ({','.join(keys)}) VALUES ({','.join('?' * len(keys))})",
                            [row[k] for k in keys])
            con.commit()
            out[area] = {"locked": len(future), "missed": len(missed),
                         "trades": int(sum(1 for r in rows if r["action"] != "HOLD"))}
            log.info("[%s] journal: %s", area, out[area])
    finally:
        con.close()
    return out


def settle(store, cfg, now=None) -> int:
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    con = connect(cfg)
    n = 0
    try:
        open_rows = pd.read_sql_query(
            "SELECT area, quarter_utc, action, mwh, cost_eur_mwh FROM decisions WHERE spread_actual IS NULL", con)
        if open_rows.empty:
            return 0
        lo = pd.to_datetime(open_rows["quarter_utc"]).min()
        imb = store.df("SELECT area, time_utc, imbalance_eur - spot_eur AS spread FROM imbalance "
                       "WHERE imbalance_eur IS NOT NULL AND spot_eur IS NOT NULL AND time_utc >= ?", [lo])
        imb["key"] = (imb["area"].astype(str) + "|"
                      + pd.to_datetime(imb["time_utc"]).dt.strftime("%Y-%m-%d %H:%M:%S").astype(str))
        sp = dict(zip(imb["key"], imb["spread"]))
        for _, r in open_rows.iterrows():
            q = pd.Timestamp(r["quarter_utc"])
            if q + Q > now:
                continue
            s = sp.get(f"{r['area']}|{r['quarter_utc']}")
            if s is None or pd.isna(s):
                continue
            sign = 1.0 if r["action"] == "BUY" else (-1.0 if r["action"] == "SELL" else 0.0)
            mwh = float(r["mwh"] or 0.0)
            cost = float(r["cost_eur_mwh"] if r["cost_eur_mwh"] is not None else cfg.cost_per_mwh)
            pnl = sign * mwh * float(s) - (mwh * cost if sign != 0 else 0.0)
            con.execute("UPDATE decisions SET spread_actual=?, pnl_eur=?, settled_at_utc=? WHERE area=? AND quarter_utc=?",
                        (float(s), float(pnl), _ts(now), r["area"], r["quarter_utc"]))
            n += 1
        con.commit()
    finally:
        con.close()
    return n


def summary(cfg, days: int | None = None) -> pd.DataFrame:
    j = read(cfg)
    if j.empty:
        return pd.DataFrame()
    if days:
        j = j[j["quarter_utc"] >= j["quarter_utc"].max() - pd.Timedelta(days=days)]
    rows = []
    for area, ga in j.groupby("area"):
        g = ga[ga["spread_actual"].notna()]
        t = g[g["action"] != "HOLD"]
        mwh = float(t["mwh"].sum())
        rows.append({"area": area, "locked": int((ga["status"] == "LOCKED").sum()),
                     "open": int(ga["spread_actual"].isna().sum()), "settled_quarters": len(g),
                     "trades": len(t), "mwh": round(mwh, 1), "net_eur": round(float(t["pnl_eur"].sum()), 2),
                     "eur_per_mwh": round(float(t["pnl_eur"].sum()) / mwh, 2) if mwh else np.nan,
                     "win_rate_pct": round(100 * float((t["pnl_eur"] > 0).mean()), 1) if len(t) else np.nan,
                     "missed_gates": int((ga["status"] == "MISSED").sum()),
                     "first": ga["quarter_utc"].min(), "last": ga["quarter_utc"].max()})
    return pd.DataFrame(rows)

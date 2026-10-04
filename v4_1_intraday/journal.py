"""Locked decision journal for V4.1 intraday paper trading (SQLite, safe for concurrent readers).

Why: the dashboard can recompute any day with the current model, but a retrained model
changes history. The journal stores each quarter's decision ONCE, before its gate closure,
with the information and model available at that moment. Performance is judged on this
journal only.

  lock(store, cfg)    decide every quarter whose deadline is within `lock_ahead_minutes`
                      (as of min(now, decision time), i.e. before the deadline); quarters whose
                      deadline passed without a lock are written as MISSED/HOLD (never decided
                      with later information).
                      gate mode : deadline = the quarter's own gate closure (delivery - 60 min)
                      batch mode: deadline = batch deadline (first quarter of the hour - 2h15);
                                  the 4 quarters of a batch are locked together and a CSV + JSON
                                  batch file is written for the BRP (batch_schedule.batch_files_dir)
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
        "settled_at_utc", "batch_start_utc", "deadline_utc"]

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS decisions (
    area TEXT NOT NULL, quarter_utc TEXT NOT NULL, gate_utc TEXT, locked_at_utc TEXT, as_of_utc TEXT,
    status TEXT, model_version TEXT, model_trained_until TEXT, action TEXT, mwh REAL, mw REAL, edge REAL,
    exp_spread REAL, p_up REAL, p_flat REAL, p_down REAL, q10 REAL, q50 REAL, q90 REAL, reason TEXT,
    hour_all_same_side INTEGER, spread_actual REAL, pnl_eur REAL, cost_eur_mwh REAL, settled_at_utc TEXT,
    batch_start_utc TEXT, deadline_utc TEXT,
    PRIMARY KEY (area, quarter_utc));
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def path(cfg, shadow: bool = False) -> Path:
    """Main journal = the decisions sent to the client (each zone's recommended model).
    shadow=True = the comparison journal: LightGBM in zones whose recommended model is not
    LightGBM (2026-10-04). It is locked and settled the same way but never reaches the client."""
    p = cfg.path("journal_path") if "journal_path" in cfg.raw else cfg.path("models_dir").parent / "data" / "v41_journal.sqlite"
    return p.with_name(p.stem + "_shadow_lgbm" + p.suffix) if shadow else p


def shadow_areas(cfg) -> list[str]:
    """Zones that also run LightGBM in the shadow journal."""
    return [a for a in cfg["areas"] if P.recommended_family(cfg, a) != "lgbm"]


def connect(cfg, shadow: bool = False) -> sqlite3.Connection:
    p = path(cfg, shadow)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=30)
    con.executescript(SCHEMA)
    # journals created before 2026-09-27 lack the batch columns -> add them (existing rows stay NULL)
    have = {r[1] for r in con.execute("PRAGMA table_info(decisions)")}
    for c in ("batch_start_utc", "deadline_utc"):
        if c not in have:
            con.execute(f"ALTER TABLE decisions ADD COLUMN {c} TEXT")
    con.commit()
    return con


def _ts(x) -> str | None:
    return None if x is None or pd.isna(x) else pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S")


def journal_start(con) -> pd.Timestamp | None:
    r = con.execute("SELECT value FROM meta WHERE key='journal_start_utc'").fetchone()
    return pd.Timestamp(r[0]) if r else None


def read(cfg, area: str | None = None, start=None, end=None, shadow: bool = False) -> pd.DataFrame:
    p = path(cfg, shadow)
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
    for c in ("quarter_utc", "gate_utc", "locked_at_utc", "as_of_utc", "settled_at_utc",
              "batch_start_utc", "deadline_utc"):
        if c in df.columns:
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


def quarter_number(quarters_utc, tz: str) -> pd.Series:
    """Client quarter number within the local delivery day: Q1 = 00:00 local. Counts real
    15-minute steps, so the clock-change days have 92 / 100 quarters."""
    q = pd.to_datetime(pd.Series(quarters_utc)).reset_index(drop=True)
    loc = q.dt.tz_localize("UTC").dt.tz_convert(tz)
    day0 = loc.dt.normalize()          # local midnight (tz-aware) -> real elapsed time
    return ((loc - day0).dt.total_seconds() // 900 + 1).astype(int)


def write_batch_files(cfg, area: str, rows: list) -> list[Path]:
    """One CSV + JSON per locked batch: results/v4_1_batch/batches/<area>/<local date>/batch_<HHMM>.*
    Only newly LOCKED rows are passed in, so an existing file is never rewritten."""
    import json
    if not rows:
        return []
    tz = cfg["local_tz"]
    d = pd.DataFrame(rows)
    d["q"] = pd.to_datetime(d["quarter_utc"])
    d["bs"] = pd.to_datetime(d["batch_start_utc"])
    from .settings import BASE
    root = Path(cfg.bs.get("batch_files_dir") or "results/v4_1_batch/batches")
    root = root if root.is_absolute() else BASE / root
    written = []
    for bs, g in d.groupby("bs"):
        g = g.sort_values("q")
        bl = pd.Timestamp(bs).tz_localize("UTC").tz_convert(tz)
        qloc = g["q"].dt.tz_localize("UTC").dt.tz_convert(tz)
        out = pd.DataFrame({
            "area": area,
            "delivery_date_local": qloc.dt.strftime("%Y-%m-%d").values,
            "quarter_no": quarter_number(g["q"], tz).values,
            "quarter_start_local": qloc.dt.strftime("%Y-%m-%d %H:%M%z").values,
            "quarter_start_utc": g["q"].dt.strftime("%Y-%m-%d %H:%M").values,
            "action": g["action"].values, "mwh": g["mwh"].astype(float).round(3).values,
            "exp_spread_eur_mwh": pd.to_numeric(g.get("exp_spread"), errors="coerce").round(2).values,
            "reason": g["reason"].values})
        folder = root / area / bl.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / f"batch_{bl.strftime('%H%M')}"
        if stem.with_suffix(".csv").exists():       # repeated hour on the October clock change
            stem = stem.with_name(stem.name + f"_{bl.strftime('%z').lstrip('+')}")
        out.to_csv(stem.with_suffix(".csv"), index=False)
        meta = {"area": area, "batch_start_utc": _ts(bs), "batch_start_local": bl.strftime("%Y-%m-%d %H:%M%z"),
                "deadline_utc": g["deadline_utc"].iloc[0], "locked_at_utc": g["locked_at_utc"].iloc[0],
                "as_of_utc": g["as_of_utc"].iloc[0], "model_version": g["model_version"].iloc[0],
                "status": "LOCKED", "quarters": out.to_dict(orient="records")}
        stem.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        written.append(stem.with_suffix(".csv"))
        log.info("[%s] batch file %s (%d quarters)", area, stem.with_suffix(".csv").name, len(out))
    return written


def _fallback_buy_ok(con, area: str, cfg, now) -> tuple[bool, float]:
    """Kill switch of the fallback BUY rule: off when its settled PnL over the last
    kill_window_days is below -kill_loss_eur."""
    f = (cfg["decision"].get("fallback_buy") or {})
    if not f.get("enabled"):
        return False, 0.0
    since = _ts(now - pd.Timedelta(days=int(f.get("kill_window_days", 30))))
    r = con.execute("SELECT COALESCE(SUM(pnl_eur), 0) FROM decisions WHERE area=? AND reason='fallback BUY rule' "
                    "AND pnl_eur IS NOT NULL AND quarter_utc>=?", (area, since)).fetchone()
    net = float(r[0] or 0.0)
    ok = net > -float(f.get("kill_loss_eur", 300.0))
    if not ok:
        log.warning("[%s] fallback BUY rule OFF: %.0f EUR over the last %s days", area, net,
                    f.get("kill_window_days", 30))
    return ok, net


def lock(store, cfg, now=None, fb=None, bundles=None, shadow: bool = False) -> dict:
    """shadow=False: lock each zone's recommended model into the main journal (+ client batch
    files). shadow=True: lock LightGBM for the zones in shadow_areas() into the shadow journal
    (no batch files)."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    ahead = pd.Timedelta(minutes=int(cfg["intraday"].get("lock_ahead_minutes", 20)))
    # longest distance between a quarter and its deadline (batch: last quarter of the hour)
    span = pd.Timedelta(hours=8)
    areas = shadow_areas(cfg) if shadow else list(cfg["areas"])
    if not areas:
        return {}
    con = connect(cfg, shadow)
    out = {}
    try:
        start = journal_start(con)
        if start is None:
            start = now
            con.execute("INSERT INTO meta VALUES ('journal_start_utc', ?)", (_ts(now),))
            con.execute("INSERT OR REPLACE INTO meta VALUES ('mode', ?)", (cfg.mode,))
            con.commit()
        fb = fb or IntradayFeatureBuilder(store, cfg)
        for area in areas:
            b = (bundles or {}).get(area)
            if b is None:
                mp = P.model_path(cfg, area, "lgbm" if shadow else None)
                if not mp.exists() and not shadow and P.recommended_family(cfg, area) != "lgbm":
                    # 2026-10-04 safety net: the recommended model is not trained yet on this
                    # machine (e.g. just after a code update) -> keep trading with LightGBM
                    # instead of sending nothing, until `train` has produced the new model.
                    alt = P.model_path(cfg, area, "lgbm")
                    if alt.exists():
                        log.warning("[%s] %s model missing (%s) - locking with V4.1 LightGBM until it is trained",
                                    area, P.recommended_family(cfg, area), mp.name)
                        mp = alt
                if not mp.exists():
                    log.warning("[%s] no %s-mode model - run `train_v4_1.py%s train`", area, cfg.mode,
                                " --mode batch" if cfg.batch_mode else "")
                    continue
                b = P.IntradayBundle.load(mp)
            done = {r[0] for r in con.execute(
                "SELECT quarter_utc FROM decisions WHERE area=? AND quarter_utc>=?", (area, _ts(start)))}
            # candidate quarters: deadline in [journal start, now + ahead]
            grid = pd.Series(pd.date_range(start.floor("15min"), (now + ahead + span).ceil("15min"), freq="15min"))
            dl = cfg.deadlines(grid)
            sel = ((dl >= start) & (dl <= now + ahead)).values
            cand = pd.DataFrame({"q": grid[sel].values, "dl": dl[sel].values,
                                 "bs": cfg.batch_starts(grid)[sel].values})
            cand = cand[~cand["q"].map(_ts).isin(done)]
            fut_m = (cand["dl"] > now).values
            future = list(cand.loc[fut_m, "q"])
            info = cand.set_index("q")
            rows = []
            for q, r0 in cand[~fut_m].set_index("q").iterrows():
                rows.append({"area": area, "quarter_utc": _ts(q), "gate_utc": _ts(r0["dl"]), "locked_at_utc": _ts(now),
                             "as_of_utc": None, "status": "MISSED", "model_version": b.version,
                             "model_trained_until": _ts(b.trained_until), "action": "HOLD", "mwh": 0.0, "mw": 0.0,
                             "reason": ("batch deadline" if cfg.batch_mode else "gate") + " passed before the system ran",
                             "cost_eur_mwh": cfg.cost_per_mwh,
                             "batch_start_utc": _ts(r0["bs"]), "deadline_utc": _ts(r0["dl"])})
            missed = [r["quarter_utc"] for r in rows]
            if future:
                # decide the whole local hour(s) so the hourly agreement flag can be computed
                tz = cfg["local_tz"]
                hours = {pd.Timestamp(q).tz_localize("UTC").tz_convert(tz).floor("h") for q in future}
                allq = sorted({h.tz_convert("UTC").tz_localize(None) + k * Q for h in hours for k in range(4)}
                              | set(future))
                fb_ok, fb_net = _fallback_buy_ok(con, area, cfg, now)
                pr = P.predict_quarters(store, cfg, b, pd.Series(allq), now=now, fb=fb, fallback_buy_ok=fb_ok)
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
                                 "cost_eur_mwh": cfg.cost_per_mwh,
                                 "batch_start_utc": _ts(info.loc[r["quarter_utc"], "bs"]),
                                 "deadline_utc": _ts(info.loc[r["quarter_utc"], "dl"])})
            # Apply live risk overlay (daily stop + drawdown guard) to sized rows
            rows = _apply_live_risk(rows, con, area, cfg, now)
            rows = _apply_hourly_cap(rows, con, area, cfg, cfg["local_tz"])
            for row in rows:
                keys = [k for k in COLS if k in row]
                con.execute(f"INSERT OR IGNORE INTO decisions ({','.join(keys)}) VALUES ({','.join('?' * len(keys))})",
                            [row[k] for k in keys])
            con.commit()
            if cfg.batch_mode and future and not shadow:
                try:
                    write_batch_files(cfg, area, [r for r in rows if r.get("status") == "LOCKED"])
                except Exception as e:          # a file problem must never undo the lock
                    log.error("[%s] batch file not written: %s", area, e)
            out[area] = {"locked": len(future), "missed": len(missed),
                         "trades": int(sum(1 for r in rows if r["action"] != "HOLD"))}
            log.info("[%s] %sjournal: %s", area, "shadow LightGBM " if shadow else "", out[area])
    finally:
        con.close()
    return out


def settle(store, cfg, now=None, shadow: bool = False) -> int:
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    if shadow and not path(cfg, True).exists():
        return 0
    con = connect(cfg, shadow)
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


def summary(cfg, days: int | None = None, shadow: bool = False) -> pd.DataFrame:
    j = read(cfg, shadow=shadow)
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

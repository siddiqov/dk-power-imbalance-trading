"""Nurex V4.1 - DK model improvement research (plan "Nurex V4.1 - DK Model Improvement Plan").

RUN ON THE DENMARK PC ONLY, via scripts_v41\\run_dk_research_v41.bat.
Research only: reads a private copy of the store and the live journals (read-only).
It never writes models, never changes config_v41.yaml, never touches the live journal/tasks.

Stages (each result is saved as soon as it is ready; a re-run skips finished runs):
  0  week review     6-12 Oct live journals per zone / model        -> week_review.md
  S  smoke test      one short replay + scoring (catches errors in minutes, not hours)
  2  candidates      walk-forward replays from --start with different training settings
  1  safety switch   every replay re-scored with no-skill / losing-streak pauses
  R  report          REPORT.md + summary.csv + best.json (ranked against the plan's 5 criteria)

Plan criteria (section 2), judged on the replay period:
  1 forecast beats predicting zero (MAE of exp_spread and of q50 < MAE of 0)
  2 net P&L positive in both halves of the period
  3 calibrated: mean P(up) / P(down) within 5 points of the actual up / down shares
  4 uses both sides: at least one BUY and one SELL
  5 max drawdown no worse than the current setup (base, no switch) of the same zone + model
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from v4_1_intraday import settings as S                      # noqa: E402
from v4_1_intraday import pipeline_id as P                   # noqa: E402
from v4_1_intraday import journal as J                       # noqa: E402
from v4_1_intraday.features_id import IntradayFeatureBuilder  # noqa: E402
from nurex42 import decision as dec                          # noqa: E402
from nurex42.storage import Store                            # noqa: E402

log = logging.getLogger("research_dk")

WEEK = ("2026-10-06", "2026-10-12")          # test week, local (Europe/Copenhagen) delivery days

# candidate training settings (config overrides, dotted keys). "base" = live config unchanged.
CANDIDATES = {
    "base":   {},
    "H90":    {"model.plain_recency_half_life_days": 90},
    "H60":    {"model.plain_recency_half_life_days": 60},
    "W180":   {"intraday.train_window_days": 180},
    "BAL":    {"logreg.class_weight_grid": ["balanced"]},
    "H90BAL": {"model.plain_recency_half_life_days": 90, "logreg.class_weight_grid": ["balanced"]},
}
# run order = priority (most useful first, so a partial night still answers the main question)
RUNS = [
    ("DK1", "logreg", "base"), ("DK1", "lgbm", "base"), ("DK2", "lgbm", "base"),
    ("DK1", "logreg", "H90"), ("DK1", "lgbm", "H90"), ("DK2", "lgbm", "H90"),
    ("DK1", "logreg", "W180"), ("DK1", "lgbm", "W180"), ("DK2", "lgbm", "W180"),
    ("DK1", "logreg", "BAL"), ("DK1", "logreg", "H90BAL"),
    ("DK1", "logreg", "H60"), ("DK1", "lgbm", "H60"), ("DK2", "lgbm", "H60"),
    ("DK2", "logreg", "base"), ("DK2", "logreg", "H90BAL"),
]
# safety-switch variants applied to every replay: (name, kind, window_days, margin)
GATES = [
    ("none", None, 0, 0.0),
    ("skill7", "skill", 7, 0.0),       # pause while the last 7 days' MAE is not below predicting zero
    ("skill14", "skill", 14, 0.0),
    ("skill7m5", "skill", 7, 0.05),    # ... must beat zero by at least 5 %
    ("pnl7", "pnl", 7, 0.0),           # pause while the model's own (unpaused) trades lost money over 7 days
    ("pnl14", "pnl", 14, 0.0),
]
MIN_SETTLED = 96 * 3                   # a switch needs >= 3 days of settled quarters before it can pause
DEADBAND = 0.5


# ----------------------------------------------------------------------------- helpers
def with_overrides(cfg, over: dict):
    c = copy.deepcopy(cfg)
    for k, v in over.items():
        node = c.raw
        *path, last = k.split(".")
        for part in path:
            node = node.setdefault(part, {})
        node[last] = copy.deepcopy(v)
    return c


def local_day(ts: pd.Series, tz: str) -> pd.Series:
    return pd.to_datetime(ts).dt.tz_localize("UTC").dt.tz_convert(tz).dt.strftime("%Y-%m-%d")


def max_dd(net: pd.Series) -> float:
    eq = net.cumsum()
    return float((eq - eq.cummax()).min()) if len(eq) else 0.0


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else (f"{v:,.3f}" if abs(v) < 1 else f"{v:,.0f}"))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- stage 0
def week_review(cfg, out: Path) -> None:
    tz = cfg["local_tz"]
    rows = []
    for shadow in (False, True):
        p = J.path(cfg, shadow)
        if not p.exists():
            log.info("journal not found: %s", p)
            continue
        con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True, timeout=30)
        d = pd.read_sql("SELECT * FROM decisions", con)
        con.close()
        if d.empty:
            continue
        d["quarter_utc"] = pd.to_datetime(d["quarter_utc"])
        d["day"] = local_day(d["quarter_utc"], tz)
        d = d[(d["day"] >= WEEK[0]) & (d["day"] <= WEEK[1])]
        for area, g in d.groupby("area"):
            fam = "lgbm" if shadow else P.recommended_family(cfg, area)
            s = g[g["spread_actual"].notna() & (g["status"] != "MISSED")]
            sp = s["spread_actual"].astype(float)
            act = s["action"].fillna(dec.HOLD)
            trade = act.isin([dec.BUY, dec.SELL])
            lean = np.sign(s["exp_spread"].astype(float))
            cost = s["cost_eur_mwh"].astype(float).fillna(cfg.cost_per_mwh)
            hold = ~trade
            up, dn = sp > DEADBAND, sp < -DEADBAND
            moved = up | dn
            rows.append({
                "zone": area, "model": fam, "journal": "shadow" if shadow else "main (client)",
                "quarters": len(g), "missed": int((g["status"] == "MISSED").sum()), "settled": len(s),
                "trades": int(trade.sum()), "BUY": int((act == dec.BUY).sum()), "SELL": int((act == dec.SELL).sum()),
                "net_eur": float(s["pnl_eur"].fillna(0).sum()),
                "dir_hit_pct": float(100 * (np.sign(s["exp_spread"].astype(float))[moved] == np.sign(sp[moved])).mean())
                if moved.any() else np.nan,
                "up_q": int(up.sum()), "up_called_BUY": int((up & (act == dec.BUY)).sum()),
                "down_q": int(dn.sum()), "down_called_SELL": int((dn & (act == dec.SELL)).sum()),
                "hold_lean_pnl_1mwh": float(((lean * sp - cost)[hold & (lean != 0)]).sum()),
                "mae_exp": float((s["exp_spread"].astype(float) - sp).abs().mean()),
                "mae_q50": float((s["q50"].astype(float) - sp).abs().mean()),
                "mae_zero": float(sp.abs().mean()),
                "p_up_mean": float(s["p_up"].astype(float).mean()), "up_share": float(up.mean()),
                "p_down_mean": float(s["p_down"].astype(float).mean()), "down_share": float(dn.mean()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "week_review.csv", index=False)
    txt = [f"# Test week review {WEEK[0]} .. {WEEK[1]} (Denmark live journals)\n",
           "hold_lean_pnl_1mwh = what the HOLD quarters would have made trading 1 MWh in the model's "
           "leaning direction (negative = holding saved money).\n", md_table(df)]
    (out / "week_review.md").write_text("\n".join(txt), encoding="utf-8")
    log.info("week review written: %s\n%s", out / "week_review.md", df.to_string())


# ----------------------------------------------------------------------------- stage 1 (switch)
def gate_paused(r: pd.DataFrame, kind: str, days: int, margin: float) -> np.ndarray:
    """True where the switch pauses trading. Uses only quarters whose label was known at the
    decision time (no look-ahead)."""
    known = r["label_known_at"].to_numpy("datetime64[ns]")
    order = np.argsort(known, kind="stable")
    ks = known[order]
    if kind == "skill":
        a = (r["exp_spread"] - r["spread"]).abs().to_numpy(float)[order]
        b = r["spread"].abs().to_numpy(float)[order]
    else:
        a = r["net_raw_eur"].to_numpy(float)[order]
        b = np.zeros_like(a)
    ca, cb = np.concatenate([[0.0], np.nancumsum(a)]), np.concatenate([[0.0], np.nancumsum(b)])
    asof = r["as_of_trade"].to_numpy("datetime64[ns]")
    hi = np.searchsorted(ks, asof, side="right")
    lo = np.searchsorted(ks, asof - np.timedelta64(int(days * 86400), "s"), side="right")
    n = hi - lo
    sa, sb = ca[hi] - ca[lo], cb[hi] - cb[lo]
    if kind == "skill":
        return (n >= MIN_SETTLED) & (sa >= (1.0 - margin) * sb)
    return (n >= MIN_SETTLED) & (sa < 0)


def apply_gate(res: pd.DataFrame, cfg, kind, days, margin) -> tuple[pd.DataFrame, float]:
    if kind is None:
        return res, 0.0
    r = res.sort_values("as_of_trade").reset_index(drop=True)
    paused = gate_paused(r, kind, days, margin)
    base_cols = [c for c in r.columns if c not in ("gross_eur", "cost_eur", "net_eur", "net_raw_eur",
                                                     "mwh_raw", "local_day", "action_raw")]
    d = r[base_cols].copy()
    d["action"] = r["action_raw"].to_numpy(object)
    d["mwh"] = r["mwh_raw"].to_numpy(float)
    trade = d["action"] != dec.HOLD
    pz = paused & trade.to_numpy()
    d.loc[pz, "action"] = dec.HOLD
    d.loc[pz, "mwh"] = 0.0
    d.loc[pz, "reason"] = f"{kind} switch"
    area = d["area"].iloc[0] if "area" in d.columns else None
    d = P.apply_risk_overlays(d, cfg)
    if area is not None:
        d["area"] = area
    return d, float(paused.mean())


# ----------------------------------------------------------------------------- scoring
def score(r: pd.DataFrame, cfg) -> dict:
    r = r.sort_values("quarter_utc").reset_index(drop=True)
    tz = cfg["local_tz"]
    day = local_day(r["quarter_utc"], tz)
    act = r["action"]
    sp = r["spread"].astype(float)
    t0, t1 = r["quarter_utc"].min(), r["quarter_utc"].max()
    mid = t0 + (t1 - t0) / 2
    up, dn = sp > DEADBAND, sp < -DEADBAND
    last30 = r["quarter_utc"] >= t1 - pd.Timedelta(days=30)
    wk = (day >= WEEK[0]) & (day <= WEEK[1])
    mae_q50 = float((r["q50"] - sp).abs().mean()) if "q50" in r and r["q50"].notna().any() else np.nan
    return {
        "from": str(t0.date()), "to": str(t1.date()),
        "trades": int((act != dec.HOLD).sum()), "BUY": int((act == dec.BUY).sum()), "SELL": int((act == dec.SELL).sum()),
        "net_eur": float(r["net_eur"].sum()),
        "net_h1": float(r.loc[r["quarter_utc"] < mid, "net_eur"].sum()),
        "net_h2": float(r.loc[r["quarter_utc"] >= mid, "net_eur"].sum()),
        "max_dd": max_dd(r["net_eur"]),
        "net_last30": float(r.loc[last30, "net_eur"].sum()),
        "net_test_week": float(r.loc[wk, "net_eur"].sum()), "trades_test_week": int((act[wk] != dec.HOLD).sum()),
        "mae_exp": float((r["exp_spread"] - sp).abs().mean()), "mae_q50": mae_q50, "mae_zero": float(sp.abs().mean()),
        "mae_exp_last30": float((r.loc[last30, "exp_spread"] - sp[last30]).abs().mean()),
        "mae_zero_last30": float(sp[last30].abs().mean()),
        "p_up": float(r["p_up"].mean()), "up_share": float(up.mean()),
        "p_down": float(r["p_down"].mean()), "down_share": float(dn.mean()),
    }


def monthly(r: pd.DataFrame, cfg) -> pd.Series:
    m = pd.to_datetime(local_day(r["quarter_utc"], cfg["local_tz"])).dt.strftime("%Y-%m")
    return r.groupby(m.values)["net_eur"].sum()


def judge(row: dict, base_dd: float) -> dict:
    mae_ok = row["mae_exp"] < row["mae_zero"] and (np.isnan(row["mae_q50"]) or row["mae_q50"] < row["mae_zero"])
    c = {
        "c1_beats_zero": bool(mae_ok),
        "c2_both_halves": bool(row["net_h1"] > 0 and row["net_h2"] > 0),
        "c3_calibrated": bool(abs(row["p_up"] - row["up_share"]) <= 0.05 and abs(row["p_down"] - row["down_share"]) <= 0.05),
        "c4_both_sides": bool(row["BUY"] > 0 and row["SELL"] > 0),
        "c5_dd_ok": bool(row["max_dd"] >= base_dd - 1e-6),
    }
    c["passes"] = int(sum(c.values()))
    c["all_pass"] = c["passes"] == 5
    return c


# ----------------------------------------------------------------------------- runs
def run_one(store, cfg0, fb, cache, area, fam, cand, start, out_csv: Path) -> pd.DataFrame:
    cfg = with_overrides(cfg0, CANDIDATES[cand])
    t0 = time.time()
    log.info("==== replay %s %s %s from %s", area, fam, cand, start)
    o = P.walk_forward(store, cfg, area, start=start, fb=fb, family=fam, design_cache=cache)
    res = o["results"]
    res.to_csv(out_csv, index=False, compression="gzip")
    o["periods"].to_csv(out_csv.with_name(out_csv.name.replace(".csv.gz", "_periods.csv")), index=False)
    log.info("==== done %s %s %s in %.0f min: trades=%d net=%.0f EUR", area, fam, cand, (time.time() - t0) / 60,
             int((res["action"] != dec.HOLD).sum()), res["net_eur"].sum())
    return res


def load_run(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path, compression="gzip")
    for c in ("quarter_utc", "batch_start_utc", "deadline_utc", "as_of_trade", "label_known_at"):
        if c in r.columns:
            r[c] = pd.to_datetime(r[c])
    return r


def evaluate(res: pd.DataFrame, cfg, area, fam, cand) -> list[dict]:
    rows = []
    for g, kind, days, margin in GATES:
        r, pshare = apply_gate(res, cfg, kind, days, margin)
        s = score(r, cfg)
        rows.append({"zone": area, "model": fam, "training": cand, "switch": g, "paused_pct": 100 * pshare, **s,
                     "monthly": json.dumps({k: round(v) for k, v in monthly(r, cfg).items()})})
    return rows


def report(out: Path, rows: list[dict], start: str, elapsed_h: float, errors: list[str]) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        (out / "REPORT.md").write_text("# No replay finished\n\n" + "\n".join(errors), encoding="utf-8")
        return
    judged = []
    for (z, m), g in df.groupby(["zone", "model"], sort=False):
        b = g[(g["training"] == "base") & (g["switch"] == "none")]
        base_dd = float(b["max_dd"].iloc[0]) if len(b) else float(g["max_dd"].min())
        for _, r in g.iterrows():
            judged.append({**r.to_dict(), **judge(r.to_dict(), base_dd)})
    df = pd.DataFrame(judged)
    df.to_csv(out / "summary.csv", index=False)
    show = ["model", "training", "switch", "paused_pct", "trades", "BUY", "SELL", "net_eur", "net_h1", "net_h2",
            "max_dd", "net_last30", "net_test_week", "mae_exp", "mae_q50", "mae_zero", "p_up", "up_share", "passes"]
    lines = [f"# DK model improvement research - Denmark PC ({datetime.now():%Y-%m-%d %H:%M})\n",
             f"Walk-forward replay from {start} (retrain every "
             f"{df.attrs.get('retrain', 30)} days, each period trained only on data known before it). "
             f"Runtime {elapsed_h:.1f} h. Net P&L in EUR after costs and the live risk overlays.\n",
             "Training: base = live config; H90/H60 = recency half-life 90/60 days; W180 = last 180 days only; "
             "BAL = logistic class_weight balanced only.\n",
             "Switch: none; skillN = pause while the last N days' forecast error is not below predicting zero "
             "(m5 = must beat it by 5 %); pnlN = pause while the model's own trades lost money over N days.\n",
             "passes = number of plan criteria met (1 beats zero, 2 both halves positive, 3 calibrated, "
             "4 BUY and SELL, 5 drawdown no worse than base).\n"]
    best = {}
    for z, gz in df.groupby("zone", sort=False):
        lines.append(f"\n## {z}\n")
        for m, g in gz.groupby("model", sort=False):
            lines.append(f"\n### {z} {m}\n")
            lines.append(md_table(g.sort_values("net_eur", ascending=False)[show]))
        cur = gz[(gz["training"] == "base") & (gz["switch"] == "none")]
        ok = gz[gz["all_pass"]]
        pick = ok.sort_values("net_eur", ascending=False).head(1)
        lines.append("\n**Verdict:** ")
        if len(pick):
            p = pick.iloc[0]
            lines.append(f"best setup passing all 5 criteria: **{p['model']} / {p['training']} / switch {p['switch']}** "
                         f"net {p['net_eur']:,.0f} EUR (halves {p['net_h1']:,.0f} / {p['net_h2']:,.0f}), "
                         f"max DD {p['max_dd']:,.0f}, test week {p['net_test_week']:,.0f}.")
            best[z] = {k: (p[k].item() if hasattr(p[k], "item") else p[k]) for k in
                       ("model", "training", "switch", "net_eur", "net_h1", "net_h2", "max_dd", "net_test_week")}
            best[z]["overrides"] = CANDIDATES[p["training"]]
        else:
            nb = gz[(gz["net_h1"] > 0) & (gz["net_h2"] > 0)].sort_values("net_eur", ascending=False).head(1)
            lines.append("no setup passes all 5 criteria -> do NOT retrain the live model for this zone. ")
            if len(nb):
                p = nb.iloc[0]
                lines.append(f"Best with both halves positive: {p['model']} / {p['training']} / switch {p['switch']} "
                             f"net {p['net_eur']:,.0f} EUR (passes {p['passes']}/5).")
                best[z] = {"model": p["model"], "training": p["training"], "switch": p["switch"],
                           "net_eur": float(p["net_eur"]), "passes": int(p["passes"]), "all_pass": False,
                           "overrides": CANDIDATES[p["training"]]}
        if len(cur):
            for _, c in cur.iterrows():
                lines.append(f"\nCurrent setup {c['model']}/base: net {c['net_eur']:,.0f} EUR, "
                             f"test week {c['net_test_week']:,.0f}, passes {c['passes']}/5.")
        lines.append("\n")
    if errors:
        lines.append("\n## Errors\n\n" + "\n".join(f"- {e}" for e in errors) + "\n")
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "best.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
    log.info("REPORT written: %s", out / "REPORT.md")


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="private copy of the store (never the live file)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2026-05-01", help="first replayed delivery day (UTC)")
    ap.add_argument("--skip-smoke", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    if not out.is_absolute():
        out = BASE / out
    (out / "runs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out / "research.log", encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), fh], force=True)
    t_start = time.time()
    live_cfg = S.load(mode="batch")
    if Path(a.db).resolve() == Path(live_cfg.db_path).resolve():
        log.error("refusing to run on the live store - pass the copy")
        return 2
    cfg = S.load(db_path=a.db, mode="batch")
    log.info("research start: db=%s out=%s start=%s runs=%d", a.db, out, a.start, len(RUNS))

    # stage 0 - live journals (read-only)
    try:
        week_review(live_cfg, out)
    except Exception as e:
        log.error("week review failed: %s\n%s", e, traceback.format_exc())

    store = Store(Path(a.db), read_only=True)
    fb = IntradayFeatureBuilder(store, cfg)
    cache: dict = {}
    errors: list[str] = []

    # smoke test: one short period end-to-end incl. scoring and the switch re-scoring
    if not a.skip_smoke:
        try:
            sm = out / "smoke"
            sm.mkdir(exist_ok=True)
            st_day = (pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=14)).normalize()
            c = with_overrides(cfg, {"intraday.retrain_every_days": 30, **CANDIDATES["H90"]})
            log.info("==== smoke test (DK1 lgbm H90 from %s)", st_day.date())
            o = P.walk_forward(store, c, "DK1", start=st_day, fb=fb, family="lgbm", design_cache=cache)
            o["results"].to_csv(sm / "smoke.csv.gz", index=False, compression="gzip")
            rr = evaluate(load_run(sm / "smoke.csv.gz"), cfg, "DK1", "lgbm", "smoke")
            pd.DataFrame(rr).to_csv(sm / "smoke_scores.csv", index=False)
            log.info("==== smoke test OK")
        except Exception as e:
            log.error("SMOKE TEST FAILED - stopping before the long runs: %s\n%s", e, traceback.format_exc())
            store.close()
            return 1

    rows: list[dict] = []
    for area, fam, cand in RUNS:
        f = out / "runs" / f"{area}_{fam}_{cand}.csv.gz"
        try:
            if f.exists():
                log.info("skip (already done): %s", f.name)
                res = load_run(f)
            else:
                res = run_one(store, cfg, fb, cache, area, fam, cand, pd.Timestamp(a.start), f)
                res = load_run(f)
            rows += evaluate(res, cfg, area, fam, cand)
            report(out, rows, a.start, (time.time() - t_start) / 3600, errors)   # keep REPORT current
        except Exception as e:
            msg = f"{area} {fam} {cand}: {type(e).__name__}: {e}"
            errors.append(msg)
            log.error("run failed %s\n%s", msg, traceback.format_exc())
    store.close()
    report(out, rows, a.start, (time.time() - t_start) / 3600, errors)
    log.info("research finished in %.1f h, %d errors", (time.time() - t_start) / 3600, len(errors))
    return 0 if not errors else 3


if __name__ == "__main__":
    sys.exit(main())

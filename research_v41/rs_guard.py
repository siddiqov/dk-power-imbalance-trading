"""Crash-guard test (2026-10-05). RESEARCH ONLY - research copy of the data, writes only to
results/research_v41/guard/. No live file, model or setting is changed.

The live system (pipeline_id.predict_quarters) cancels a BUY when the model's q10 < -30 EUR/MWh
("BUY crash guard"). The walk-forward replays and the research never applied it. Live q10 is
below -30 in ~99.8% of quarters, so the guard blocks almost every model BUY.

This replays the production walk-forward (same retrains, threshold tuning, guards, fallback BUY,
risk overlays) ONCE per model, keeps every period's forecasts and pre-guard decisions, then
scores the same decisions with the crash guard OFF and at several thresholds - in the live order:
decide -> guards -> crash guard -> fallback BUY -> risk overlays.

  python research_v41/rs_guard.py            all three models (DK1 logreg, DK1 lgbm, DK2 lgbm)
  python research_v41/rs_guard.py DK1:lgbm   one model
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Nurex_V4_2"))
from v4_1_intraday import settings as S  # noqa: E402
from v4_1_intraday import pipeline_id as P  # noqa: E402
from v4_1_intraday.features_id import IntradayFeatureBuilder  # noqa: E402
from nurex42 import decision as dec  # noqa: E402
from nurex42.storage import Store  # noqa: E402

COPY = ROOT / "Nurex_V4_2" / "data" / "nurex42.research_copy.duckdb"
OUT = ROOT / "results" / "research_v41" / "guard"
OUT.mkdir(parents=True, exist_ok=True)
THRESHOLDS = [("live -30", -30.0), ("off", None), ("-60", -60.0), ("-100", -100.0), ("-150", -150.0)]
JOBS = [("DK1", "logreg"), ("DK1", "lgbm"), ("DK2", "lgbm")]


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "guard.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def replay(area, family, cfg):
    """Production walk-forward, keeping per period: rows, forecasts, pre-crash-guard decisions, params."""
    f = OUT / f"periods_{area}_{family}.pkl"
    if f.exists():
        log(f"[{area} {family}] using saved replay {f.name}")
        with open(f, "rb") as fh:
            return pickle.load(fh)
    st = Store(cfg.db_path, read_only=True)
    try:
        fb = IntradayFeatureBuilder(st, cfg)
        t = P.target_frame(fb, area, cfg)
        log(f"[{area} {family}] building point-in-time features for {len(t)} quarters")
        Xt, extras = P.design(fb, area, t, cfg)
    finally:
        st.close()
    ic = cfg["intraday"]
    p0 = (t["quarter_utc"].min() + pd.Timedelta(days=ic["min_train_days"])).normalize()
    last, step = t["quarter_utc"].max(), pd.Timedelta(days=ic["retrain_every_days"])
    periods = []
    while p0 <= last:
        p1 = p0 + step
        rows = t.index[(t["quarter_utc"] >= p0) & (t["quarter_utc"] < p1)]
        if len(rows):
            cut = t.loc[rows, "as_of_trade"].min()
            m, params, n_train = P.train_with_validation(Xt, extras, t, cfg, cut, family=family, area=area)
            pt = m.predict(Xt.loc[rows])
            d0 = dec.decide(pt, t.loc[rows, "quarter_utc"], m.stress, cfg, params)
            d0 = P.apply_guards(pt, d0, params.get("guards"))
            periods.append({"p0": p0, "rows": np.asarray(rows), "pt": pt, "d0": d0, "params": params})
            log(f"[{area} {family}] {p0.date()} train={n_train} buy={params.get('buy')} sell={params.get('sell')} "
                f"BUY signals={int((d0['action'] == 'BUY').sum())} SELL={int((d0['action'] == 'SELL').sum())}")
        p0 = p1
    out = {"area": area, "family": family, "t": t[["quarter_utc", "batch_start_utc", "deadline_utc", "as_of_trade",
                                                     "label_known_at", "spread"]], "periods": periods}
    with open(f, "wb") as fh:
        pickle.dump(out, fh)
    return out


def score(rep, cfg, thr):
    area, t = rep["area"], rep["t"]
    res, blocked = [], 0
    for p in rep["periods"]:
        rows, pt, d = p["rows"], p["pt"], p["d0"].copy()
        if thr is not None and "q10" in pt.columns:
            hit = (d["action"] == "BUY") & (pt["q10"] < thr)
            blocked += int(hit.sum())
            d.loc[hit, ["action", "reason"]] = ["HOLD", "buy suppressed: q10 crash risk"]
            d.loc[hit, ["mwh", "edge"]] = 0.0
        d = P.apply_fallback_buy(d, pt, p["params"], area, cfg)
        r = t.loc[rows].copy()
        r = r.join(pt[[c for c in ("exp_spread", "p_up", "p_down", "p_flat", "q10") if c in pt.columns]])
        r[["action", "mwh", "edge", "reason"]] = d[["action", "mwh", "edge", "reason"]]
        res.append(r)
    res = P.apply_risk_overlays(pd.concat(res, ignore_index=True), cfg)
    tr = res[res["action"] != "HOLD"]
    day = res.assign(day=res["quarter_utc"].dt.date).groupby("day")["net_eur"].sum()
    eq = day.cumsum()
    mon = res.assign(m=res["quarter_utc"].dt.to_period("M")).groupby("m")["net_eur"].sum()
    half = res["quarter_utc"] < res["quarter_utc"].quantile(0.5)
    buys, sells = tr[tr["action"] == "BUY"], tr[tr["action"] == "SELL"]
    mwh = tr["mwh"].sum()
    return {"net_eur": round(res["net_eur"].sum(), 0), "h1": round(res.loc[half, "net_eur"].sum(), 0),
            "h2": round(res.loc[~half, "net_eur"].sum(), 0), "trades": len(tr), "buys": len(buys), "sells": len(sells),
            "buy_net": round(buys["net_eur"].sum(), 0), "sell_net": round(sells["net_eur"].sum(), 0),
            "mwh": round(mwh, 1), "eur_per_mwh": round(res["net_eur"].sum() / mwh, 2) if mwh else 0.0,
            "max_dd": round(float((eq - eq.cummax()).min()), 0) if len(eq) else 0.0,
            "pos_months": f"{int((mon > 0).sum())}/{len(mon)}", "buys_blocked": blocked,
            "share_q10_below_thr": (round(float(np.mean(np.concatenate([p["pt"]["q10"].values for p in rep["periods"]]) < thr)) * 100, 1)
                                    if thr is not None else None)}


def main():
    jobs = [tuple(a.split(":")) for a in sys.argv[1:]] or JOBS
    cfg = S.load(db_path=str(COPY), mode="batch")
    log(f"==== crash-guard test start: {jobs} on {COPY.name}")
    rows = []
    for area, family in jobs:
        t0 = time.time()
        try:
            rep = replay(area, family, cfg)
        except Exception as e:
            log(f"[{area} {family}] FAILED {type(e).__name__}: {e}")
            continue
        for name, thr in THRESHOLDS:
            sc = score(rep, cfg, thr)
            rows.append({"area": area, "model": family, "crash_guard": name, **sc})
            log(f"RESULT {area} {family} guard={name}: " + json.dumps(sc))
        log(f"[{area} {family}] done in {(time.time() - t0) / 60:.0f} min")
        pd.DataFrame(rows).to_csv(OUT / "guard_results.csv", index=False)
    log("==== crash-guard test done")


if __name__ == "__main__":
    main()

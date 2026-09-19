"""Nurex V4.1 Intraday — command line (replaces the old T-60 trainer).

The model decides each 15-min quarter at intraday gate closure (delivery - 60 min)
using only information published before that moment. Data comes from the V4.2
point-in-time store (Nurex_V4_2/data/nurex42.duckdb).

  python train_v4_1.py update                 download new data (runs Nurex_V4_2/run.py update)
  python train_v4_1.py leaktest               corrupt-the-future leakage test (must print PASS)
  python train_v4_1.py replay [--area DK1]    walk-forward simulation + report in results/v4_1_intraday/
  python train_v4_1.py train  [--area DK1]    train the live models -> models_v4_1_intraday/
  python train_v4_1.py predict --area DK1 [--day 2026-09-17]   decisions for one local delivery day
  python train_v4_1.py all                    update + collect + leaktest + replay + train

  New data sources (credentials in Nurex_V4_2/.env):
  python train_v4_1.py collect [--source entsoe,umm,weather,frequency] [--start 2025-03-04]
                                              backfill / update ENTSO-E, UMM outages, weather, frequency
  python train_v4_1.py sources                coverage of every data source
  python train_v4_1.py probe-nordpool         test Nord Pool Intraday login + 60 s of live data
  python train_v4_1.py record-intraday        record Nord Pool intraday market data (run 24/7)

  Paper trading (locked decisions, judged honestly):
  python train_v4_1.py cycle                  update + collect + lock upcoming gates + settle (every 15 min)
  python train_v4_1.py lock                   only lock upcoming gates + settle (no downloads)
  python train_v4_1.py journal [--days 30]    locked-decision performance per zone

Simulation / research only - no orders are sent.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from v4_1_intraday import settings as S
from v4_1_intraday import leakage, pipeline_id as P, report_id as R
from v4_1_intraday.features_id import IntradayFeatureBuilder
from nurex42 import timeutil as tu
from nurex42.storage import Store

log = logging.getLogger("nurex41id")


def _store(cfg):
    if not cfg.db_path.exists():
        sys.exit(f"Database not found: {cfg.db_path}\nRun: cd Nurex_V4_2 && python run.py backfill")
    return Store(cfg.db_path, read_only=True)


def cmd_update(args, cfg):
    r = subprocess.run([sys.executable, "run.py", "update"], cwd=str(S.V42_ROOT))
    if r.returncode != 0:
        sys.exit("data update failed")


def cmd_leaktest(args, cfg):
    st = _store(cfg)
    probs = leakage.run(st, cfg, n_samples=args.samples)
    st.close()
    if probs:
        print("FAIL - features use information published after gate closure:")
        for p in probs:
            print("  ", p)
        sys.exit(1)
    print("PASS - no feature changed when post-decision data was corrupted")


def cmd_replay(args, cfg):
    st = _store(cfg)
    fb = IntradayFeatureBuilder(st, cfg)
    areas = [args.area] if args.area else cfg["areas"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    summary = {}
    for a in areas:
        out = P.walk_forward(st, cfg, a, fb=fb)
        path = cfg.path("reports_dir") / f"replay_{a}_{stamp}.md"
        rep = R.write(out, cfg, path)
        summary[a] = {"trading": rep["trading"], "forecast": rep["forecast"], "direction": rep["direction"]}
        tm = rep["trading"]
        print(f"[{a}] trades={tm['trades']} MWh={tm['mwh_traded']:.0f} net={tm['net_eur']:,.0f} EUR "
              f"({tm['net_eur_per_mwh']:.2f} EUR/MWh), at 2x costs {tm['net_eur_at_stress_costs']:,.0f} EUR, "
              f"max DD {tm['max_drawdown_eur']:,.0f} EUR -> {path}")
    (cfg.path("reports_dir") / f"replay_summary_{stamp}.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    st.close()


def cmd_train(args, cfg):
    st = _store(cfg)
    fb = IntradayFeatureBuilder(st, cfg)
    for a in [args.area] if args.area else cfg["areas"]:
        b = P.train_final(st, cfg, a, fb=fb)
        path = P.model_path(cfg, a)
        b.save(path)
        imp = b.model.feature_importance().head(15).round(2)
        print(f"[{a}] saved {path}  n_train={b.n_train}  trained_until={b.trained_until}")
        print(f"[{a}] decision params: {b.decision_params}")
        print(imp.to_string())
        (path.with_suffix(".json")).write_text(json.dumps({
            "area": a, "version": b.version, "n_train": b.n_train, "trained_until": str(b.trained_until),
            "decision_params": b.decision_params, "cost_eur_mwh": cfg.cost_per_mwh,
            "gate_lead_minutes": cfg["intraday"]["gate_lead_minutes"],
            "feature_importance_top15": imp.to_dict()}, indent=2, default=str), encoding="utf-8")
    st.close()


def cmd_predict(args, cfg):
    st = _store(cfg)
    b = P.IntradayBundle.load(P.model_path(cfg, args.area))
    day = args.day or tu.to_local(tu.utcnow(), cfg["local_tz"]).strftime("%Y-%m-%d")
    qs = tu.local_day_quarters(day, cfg["local_tz"])
    out = P.predict_quarters(st, cfg, b, qs)
    out["time_local"] = out["quarter_utc"].dt.tz_localize("UTC").dt.tz_convert(cfg["local_tz"]).dt.strftime("%H:%M")
    cols = ["time_local", "decision_final", "p_down", "p_flat", "p_up", "exp_spread", "q10", "q90",
            "action", "mwh", "edge", "spread_actual"]
    pd.set_option("display.width", 200)
    print(out[cols].round(2).to_string(index=False))
    path = cfg.path("reports_dir") / f"decisions_{args.area}_{day}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print("->", path)
    st.close()


SOURCES = ["entsoe", "umm", "weather", "frequency"]


def _last(st, sql):
    try:
        v = st.df(sql)["t"].iloc[0]
        return pd.Timestamp(v) if pd.notna(v) else None
    except Exception:
        return None


def cmd_collect(args, cfg):
    from v4_1_intraday.collectors import common, entsoe_ext, frequency, umm, weather
    srcs = args.source.split(",") if args.source else SOURCES
    st = Store(cfg.db_path)
    common.ensure_tables(st)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    start0 = pd.Timestamp(args.start or cfg["history_start"])
    ok = True
    try:
        for src in srcs:
            try:
                if src == "entsoe":
                    last = None if args.start else _last(st, "SELECT max(time_utc) t FROM entsoe_series WHERE series LIKE 'load:%'")
                    a = (last - pd.Timedelta(days=3)) if last is not None else start0
                    n = entsoe_ext.collect(st, a, now.ceil("D") + pd.Timedelta(days=2))
                elif src == "umm":
                    last = None if args.start else _last(st, "SELECT max(publication_utc) t FROM umm_events")
                    n = umm.collect(st, since=(last - pd.Timedelta(days=2)) if last is not None else start0)
                elif src == "weather":
                    last = None if args.start else _last(st, "SELECT max(time_utc) - INTERVAL 3 DAY t FROM weather_fc")
                    n = weather.collect(st, last if last is not None else start0, now + pd.Timedelta(days=2))
                elif src == "frequency":
                    last = None if args.start else _last(st, "SELECT max(time_utc) t FROM frequency")
                    n = frequency.collect(st, (last - pd.Timedelta(hours=6)) if last is not None else start0, now)
                else:
                    print(f"unknown source {src}")
                    continue
                print(f"[{src}] {n:,} rows stored")
            except Exception as e:
                ok = False
                print(f"[{src}] FAILED: {e}")
    finally:
        st.close()
    cmd_sources(args, cfg)
    return ok


def cmd_sources(args, cfg):
    from v4_1_intraday.collectors import nordpool_id
    st = Store(cfg.db_path, read_only=True)
    q = {
        "EDS imbalance": "SELECT count(*) n, min(time_utc) AS first_t, max(time_utc) AS last_t FROM imbalance",
        "EDS live system (PSRN)": "SELECT count(*) n, min(time_utc) AS first_t, max(time_utc) AS last_t FROM psrn",
        "ENTSO-E actual load": "SELECT count(*) n, min(time_utc) AS first_t, max(time_utc) AS last_t FROM entsoe_series WHERE series LIKE 'load:%'",
        "ENTSO-E load forecast": "SELECT count(*) n, min(time_utc) AS first_t, max(time_utc) AS last_t FROM entsoe_series WHERE series LIKE 'loadfc:%'",
        "ENTSO-E schedules": "SELECT count(*) n, min(time_utc) AS first_t, max(time_utc) AS last_t FROM entsoe_series WHERE series LIKE 'sched:%'",
        "ENTSO-E physical flows": "SELECT count(*) n, min(time_utc) AS first_t, max(time_utc) AS last_t FROM entsoe_series WHERE series LIKE 'phys:%'",
        "ENTSO-E NTC": "SELECT count(*) n, min(time_utc) AS first_t, max(time_utc) AS last_t FROM entsoe_series WHERE series LIKE 'ntc:%'",
        "ENTSO-E wind/solar fc": "SELECT count(*) n, min(time_utc) AS first_t, max(time_utc) AS last_t FROM entsoe_series WHERE series LIKE 'wsfc:%'",
        "UMM outages (by publication)": "SELECT count(*) n, min(publication_utc) AS first_t, max(publication_utc) AS last_t FROM umm_events",
        "Weather forecasts": "SELECT count(*) n, min(time_utc) AS first_t, max(time_utc) AS last_t FROM weather_fc",
        "Frequency (Nordic)": "SELECT count(*) n, min(time_utc) AS first_t, max(time_utc) AS last_t FROM frequency",
    }
    rows = []
    for name, sql in q.items():
        try:
            r = st.df(sql).iloc[0].to_dict()
            r = {"n": r["n"], "first": r["first_t"], "last": r["last_t"]}
        except Exception:
            r = {"n": 0, "first": None, "last": None}
        rows.append({"source": name, **r})
    st.close()
    for _, r in nordpool_id.coverage().iterrows():
        rows.append({"source": f"Nord Pool intraday {r['table']}", "n": r["rows"], "first": r["first"],
                     "last": r["last"]})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


def cmd_probe_nordpool(args, cfg):
    from v4_1_intraday.collectors import nordpool_id as npid
    c = npid.cfg_api()
    try:
        tok, user = npid.get_token(c)
    except Exception as e:
        sys.exit(f"LOGIN FAILED: {e}")
    print(f"login OK as {user}; token valid until {pd.Timestamp(npid.token_expiry(tok), unit='s')} UTC")
    before = npid.coverage()
    npid.record(stop_after=args.seconds)
    after = npid.coverage()
    print(after.to_string(index=False))
    st = npid.load("stats")
    if len(st):
        s = st.dropna(subset=["vwap"]).tail(5)
        print("sample statistics (raw API units):")
        print(s[["contract_id", "area_id", "last_price", "vwap", "last_qty", "turnover", "da_price"]].to_string(index=False))
        print(f"price check: vwap/{c['price_divisor']:.0f} should look like EUR/MWh; "
              f"da_price/{c['price_divisor']:.0f} should equal the day-ahead price")
    if int(after["rows"].sum()) <= int(before["rows"].sum()):
        print("WARNING: no new market data received - check host/area ids in config_v41.yaml (intraday_api)")


def cmd_record(args, cfg):
    from v4_1_intraday.collectors import nordpool_id as npid
    print("Recording Nord Pool intraday market data - leave this window open (Ctrl+C to stop).")
    npid.record(stop_after=None, subscribe_localview=not args.no_book)


def cmd_lock(args, cfg):
    from v4_1_intraday import journal as J
    st = _store(cfg)
    try:
        res = J.lock(st, cfg)
        n = J.settle(st, cfg)
    finally:
        st.close()
    print(f"locked: {res} | settled now: {n}")


def cmd_backfill(args, cfg):
    """Re-read recently published EDS imbalance prices into the store (fills late prices)."""
    from v4_1_intraday import backfill_imbalance as B
    return B.main(["--hours", str(getattr(args, "hours", 48) or 48)])


def cmd_cycle(args, cfg):
    """One paper-trading cycle. Data problems never stop the locking of the next gates."""
    r = subprocess.run([sys.executable, "run.py", "update"], cwd=str(S.V42_ROOT))
    if r.returncode != 0:
        print("WARNING: EDS update failed - locking with the data already stored")
    args.source = args.source if getattr(args, "source", None) else "entsoe,umm,weather,frequency"
    args.start = None
    try:
        cmd_collect(args, cfg)
    except Exception as e:
        print(f"WARNING: collect failed: {e}")
    # Settlement prices appear ~20 min after a quarter and the collector stores the row earlier
    # with an empty price, so re-read the recent window before settling the journal.
    try:
        cmd_backfill(args, cfg)
    except Exception as e:
        print(f"WARNING: imbalance backfill failed: {e}")
    cmd_lock(args, cfg)


def cmd_journal(args, cfg):
    from v4_1_intraday import journal as J
    s = J.summary(cfg, days=args.days)
    if s.empty:
        print("journal is empty - run `train_v4_1.py cycle` (scheduled every 15 min)")
        return
    print(s.to_string(index=False))
    print(f"journal file: {J.path(cfg)}")


def cmd_all(args, cfg):
    cmd_update(args, cfg)
    args.source, args.start = None, None
    cmd_collect(args, cfg)
    cmd_leaktest(args, cfg)
    cmd_replay(args, cfg)
    cmd_train(args, cfg)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("update")
    sp = sub.add_parser("leaktest"); sp.add_argument("--samples", type=int, default=5)
    sp = sub.add_parser("replay"); sp.add_argument("--area")
    sp = sub.add_parser("train"); sp.add_argument("--area")
    sp = sub.add_parser("predict"); sp.add_argument("--area", required=True); sp.add_argument("--day")
    sp = sub.add_parser("all"); sp.add_argument("--area"); sp.add_argument("--samples", type=int, default=5)
    sp = sub.add_parser("collect"); sp.add_argument("--source"); sp.add_argument("--start")
    sub.add_parser("sources")
    sp = sub.add_parser("probe-nordpool"); sp.add_argument("--seconds", type=int, default=60)
    sp = sub.add_parser("record-intraday"); sp.add_argument("--no-book", action="store_true")
    sp = sub.add_parser("cycle"); sp.add_argument("--source")
    sub.add_parser("lock")
    sp = sub.add_parser("backfill"); sp.add_argument("--hours", type=int, default=48)
    sp = sub.add_parser("journal"); sp.add_argument("--days", type=int, default=None)
    args = p.parse_args()
    cfg = S.load(args.config)
    {"update": cmd_update, "leaktest": cmd_leaktest, "replay": cmd_replay, "train": cmd_train,
     "predict": cmd_predict, "all": cmd_all, "collect": cmd_collect, "sources": cmd_sources,
     "probe-nordpool": cmd_probe_nordpool, "record-intraday": cmd_record,
     "cycle": cmd_cycle, "lock": cmd_lock, "backfill": cmd_backfill, "journal": cmd_journal}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()

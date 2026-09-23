"""Nurex V4.2 command line.

  python run.py seed-legacy --legacy ..\\energy_data.db   import real data already downloaded by V3/V4
  python run.py backfill                    full paginated download from Energi Data Service (+ ENTSO-E)
  python run.py update                      incremental download (last 3 days + new data)
  python run.py coverage                    data completeness report
  python run.py schedule --day 2026-09-17   client batch schedule for a delivery day
  python run.py replay [--area DK1]         walk-forward simulation on history + report
  python run.py train                       train the models used for live paper trading
  python run.py cycle                       one live paper-trading cycle
  python run.py live [--every 5]            run cycles forever (update data + cycle every N minutes)
  python run.py dashboard                   RETIRED (2026-09-22) - V4.1 dashboard uses port 5006
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from nurex42 import evaluation, live, pipeline
from nurex42 import timeutil as tu
from nurex42.collectors import eds
from nurex42.config import ROOT, load_settings
from nurex42.features import DA_15MIN_START, FeatureBuilder
from nurex42.storage import Store

log = logging.getLogger("nurex42")


def cmd_seed(args, cfg):
    from nurex42.collectors.legacy_seed import seed
    s = Store(cfg.db_path)
    print(seed(s, args.legacy))
    print(s.coverage().to_string(index=False))
    s.close()


def _collect_all(cfg, start, end, incremental: bool, only: list[str] | None = None):
    s = Store(cfg.db_path)
    areas = cfg["areas"]
    try:
        keys_to_run = ["imbalance", "dayahead", "forecast", "psrn", "mfrr_market"]
        if only:
            keys_to_run = [k for k in keys_to_run if k in only]

        for key in keys_to_run:
            a = eds.incremental_start(s, key, start) if incremental else pd.Timestamp(start)
            b = end
            if key == "dayahead":
                a = max(a, DA_15MIN_START - pd.Timedelta(days=1))
                b = end + pd.Timedelta(days=2)                     # tomorrow's prices once published
            if key == "forecast":
                b = end + pd.Timedelta(days=2)
            try:
                n = eds.collect(s, key, areas, a, b)
                log.info("%s: %d rows stored", key, n)
            except Exception as e:
                log.error("%s failed: %s", key, e)

        if not incremental and (not only or "elspot" in only):
            try:  # hourly day-ahead prices before the 15-minute go-live
                eds.collect(s, "elspot", areas, start, DA_15MIN_START)
            except Exception as e:
                log.error("elspot failed: %s", e)

        if not only or "entsoe" in only:
            try:
                from nurex42.collectors import entsoe_client
                last = s.df("SELECT max(time_utc) AS t FROM entsoe_series")["t"].iloc[0]
                ea = (pd.Timestamp(last) - pd.Timedelta(days=3)) if (incremental and pd.notna(last)) else start
                entsoe_client.collect(s, ea, end + pd.Timedelta(days=2))
            except Exception as e:
                log.warning("ENTSO-E skipped: %s", e)

        print(s.coverage().to_string(index=False))
    finally:
        s.close()


def cmd_backfill(args, cfg):
    start = pd.Timestamp(args.start or cfg["history_start"])
    only = args.only.split(",") if args.only else None
    _collect_all(cfg, start, tu.utcnow().ceil("D"), incremental=False, only=only)


def cmd_update(args, cfg):
    _collect_all(cfg, pd.Timestamp(cfg["history_start"]), tu.utcnow().ceil("D"), incremental=True)


def cmd_coverage(args, cfg):
    s = Store(cfg.db_path, read_only=True)
    print(s.coverage().to_string(index=False))
    s.close()


def cmd_schedule(args, cfg):
    day = args.day or str(tu.to_local(tu.utcnow(), cfg["local_tz"]).date() + pd.Timedelta(days=1))
    t = tu.daily_batch_table(day, cfg)
    print(t[["batch_start_utc_local", "last_q_local", "n_quarters", "deadline_utc_local"]]
          .rename(columns={"batch_start_utc_local": "delivery_from", "last_q_local": "last_quarter",
                           "deadline_utc_local": "submit_by"}).to_string(index=False))


def cmd_positions(args, cfg):
    s = Store(cfg.db_path, read_only=True)
    try:
        now_local = tu.to_local(tu.utcnow(), cfg["local_tz"])
        start_utc = tu.local_time_on_day(now_local.normalize().tz_localize(None), "00:00", cfg["local_tz"])
        
        pos = s.df("SELECT * FROM positions WHERE book='live' AND area=? AND quarter_utc >= ? ORDER BY quarter_utc", [args.area, start_utc])
        plan = s.df("SELECT * FROM plan WHERE run_id=(SELECT run_id FROM runs WHERE book='live' AND area=? AND kind='cycle' ORDER BY as_of_utc DESC LIMIT 1) ORDER BY quarter_utc", [args.area])
        
        if pos.empty and plan.empty:
            print(f"[{args.area}] No live plan or positions.")
            return
            
        cols = ["quarter_utc", "action", "mwh", "exp_spread", "reason"]
        pos["zone"] = "LOCKED"
        plan["zone"] = "PLAN"
        
        view = pd.concat([pos.reindex(columns=cols + ["zone"]), plan.reindex(columns=cols + ["zone"])], ignore_index=True)
        view = view.drop_duplicates("quarter_utc", keep="first").sort_values("quarter_utc")
        view["delivery_local"] = pd.to_datetime(view["quarter_utc"]).dt.tz_localize("UTC").dt.tz_convert(cfg["local_tz"]).dt.strftime("%Y-%m-%d %H:%M")
        
        print(f"\n[{args.area}] Live positions and plan from {now_local.strftime('%Y-%m-%d %H:%M %Z')}:")
        print(view[["delivery_local", "zone", "action", "mwh", "exp_spread", "reason"]].to_string(index=False))
    finally:
        s.close()


def cmd_replay(args, cfg):
    s = Store(cfg.db_path)
    fb = FeatureBuilder(s, cfg)
    areas = [args.area] if args.area else cfg["areas"]
    summary = {}
    for area in areas:
        try:
            out = pipeline.walk_forward(s, cfg, area, start=args.start, end=args.end, fb=fb)
        except Exception as e:
            log.exception("[%s] replay failed: %s", area, e)
            continue
        path = ROOT / "reports" / f"replay_{area}_{pd.Timestamp.now():%Y%m%d_%H%M}.md"
        rep = evaluation.write_report(out, cfg, path)
        summary[area] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rep["trading"].items()}
        print(f"\n[{area}] report: {path}")
        print(json.dumps(summary[area], indent=2, default=str))
        print(rep["baselines"].to_string(index=False))
    s.close()


def cmd_train(args, cfg):
    s = Store(cfg.db_path)
    fb = FeatureBuilder(s, cfg)
    for area in ([args.area] if args.area else cfg["areas"]):
        try:
            b = pipeline.train_final(s, cfg, area, fb=fb)
        except Exception as e:
            log.error("[%s] training failed: %s", area, e)
            continue
        b.save(live.model_path(area))
        print(f"[{area}] model saved -> {live.model_path(area)} | trained on {b.n_train} quarters | "
              f"labels until {b.trained_until} | decision params {b.decision_params}")
    s.close()


def cmd_cycle(args, cfg):
    s = Store(cfg.db_path)
    try:
        print(json.dumps(live.cycle(s, cfg), indent=2, default=str))
    finally:
        s.close()


def cmd_live(args, cfg):
    last_train = None
    while True:
        t0 = time.time()
        try:
            cmd_update(args, cfg)
            today = pd.Timestamp.now().date()
            if args.retrain_daily and last_train != today:
                cmd_train(argparse.Namespace(area=None), cfg)
                last_train = today
            cmd_cycle(args, cfg)
        except Exception as e:
            log.exception("cycle failed: %s", e)
        time.sleep(max(30, args.every * 60 - (time.time() - t0)))


def cmd_dashboard(args, cfg):
    # RETIRED 2026-09-22: the V4.2 dashboard is no longer used. Port 5006 is reserved for the
    # V4.1-only dashboard (dashboard_v41_live.py). Old launchers are in _retired_dashboard/.
    if not getattr(args, "force_retired", False):
        print("The V4.2 dashboard (nurex42/dashboard.py) is retired. Use the V4.1 dashboard instead.\n"
              "To start it anyway: python run.py dashboard --force-retired")
        return
    port = str(cfg["dashboard"]["port"])
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "nurex42" / "dashboard.py"),
                    "--server.port", port, "--server.headless", "true"], check=False)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Nurex V4.2")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("seed-legacy"); sp.add_argument("--legacy", default=str(ROOT.parent / "energy_data.db"))
    sp = sub.add_parser("backfill"); sp.add_argument("--start", default=None); sp.add_argument("--only", default=None)
    sub.add_parser("update")
    sub.add_parser("coverage")
    sp = sub.add_parser("schedule"); sp.add_argument("--day", default=None)
    sp = sub.add_parser("positions"); sp.add_argument("--area", required=True)
    sp = sub.add_parser("replay"); sp.add_argument("--area"); sp.add_argument("--start"); sp.add_argument("--end")
    sp = sub.add_parser("train"); sp.add_argument("--area")
    sub.add_parser("cycle")
    sp = sub.add_parser("live"); sp.add_argument("--every", type=int, default=5)
    sp.add_argument("--retrain-daily", action="store_true")
    sub.add_parser("dashboard").add_argument("--force-retired", action="store_true")
    args = p.parse_args()
    cfg = load_settings(args.config)
    {"seed-legacy": cmd_seed, "backfill": cmd_backfill, "update": cmd_update, "coverage": cmd_coverage,
     "schedule": cmd_schedule, "positions": cmd_positions, "replay": cmd_replay, "train": cmd_train, "cycle": cmd_cycle,
     "live": cmd_live, "dashboard": cmd_dashboard}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()

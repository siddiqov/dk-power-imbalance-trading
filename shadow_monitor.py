"""Nurex V4.1 — Phase 7 Shadow-Mode Live Monitor.

Reads the locked-decision journal and generates daily P&L summaries,
signal quality metrics, drawdown alerts, and weekly client reports —
without submitting any BRP orders.

Usage:
  python shadow_monitor.py               daily check + weekly report on Sundays
  python shadow_monitor.py --report week write this week's client report now
  python shadow_monitor.py --status      quick one-line status print

Windows Task Scheduler: run daily at 23:45 local time via
  scripts_v41/run_phase7_shadow_daily.bat
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ── locate project root ──────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from v4_1_intraday import settings as S

cfg = S.load(None)

# ── paths ────────────────────────────────────────────────────────────────────
JOURNAL_PATH  = HERE / cfg.get("journal_path", "data/v41_journal.sqlite")
LOG_DIR       = HERE / "logs"
RESULTS_DIR   = HERE / cfg.get("reports_dir", "results/v4_1_intraday")
WEEKLY_DIR    = RESULTS_DIR / "weekly"
DAILY_LOG     = LOG_DIR / "v41_shadow_daily.log"
ALERT_LOG     = LOG_DIR / "v41_shadow_alerts.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)
WEEKLY_DIR.mkdir(parents=True, exist_ok=True)

# ── alert thresholds ─────────────────────────────────────────────────────────
MAX_DRAWDOWN_EUR   = -5_000   # alert if rolling 7-day drawdown exceeds this
QUIET_DAYS         = 5        # alert if no trades locked for this many days
MIN_HIT_RATE       = 0.45     # alert if 7-day direction hit rate drops below
WEEKLY_REPORT_DAY  = 6        # Sunday (weekday() == 6)


# ═══════════════════════════════════════════════════════════════════════════════
#  Journal reader
# ═══════════════════════════════════════════════════════════════════════════════

def load_journal(days: int | None = None) -> pd.DataFrame:
    """Return settled journal rows as a DataFrame."""
    if not JOURNAL_PATH.exists():
        return pd.DataFrame()
    con = sqlite3.connect(JOURNAL_PATH)
    try:
        q = "SELECT * FROM decisions WHERE settlement_eur IS NOT NULL"
        if days:
            since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
            q += f" AND locked_utc >= '{since}'"
        q += " ORDER BY locked_utc"
        df = pd.read_sql_query(q, con)
    finally:
        con.close()
    if df.empty:
        return df
    df["locked_utc"]   = pd.to_datetime(df["locked_utc"])
    df["delivery_utc"] = pd.to_datetime(df["delivery_utc"])
    df["delivery_local"] = (df["delivery_utc"]
                            .dt.tz_localize("UTC")
                            .dt.tz_convert("Europe/Copenhagen")
                            .dt.date)
    return df


def load_all_locked() -> pd.DataFrame:
    """Return ALL locked rows (settled + unsettled) for quiet-model check."""
    if not JOURNAL_PATH.exists():
        return pd.DataFrame()
    con = sqlite3.connect(JOURNAL_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT area, action, locked_utc FROM decisions ORDER BY locked_utc DESC LIMIT 500", con
        )
    finally:
        con.close()
    if df.empty:
        return df
    df["locked_utc"] = pd.to_datetime(df["locked_utc"])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (df.groupby(["delivery_local", "area"])
              .agg(trades=("settlement_eur", "count"),
                   mwh=("mwh", "sum"),
                   net_eur=("settlement_eur", "sum"))
              .reset_index())


def rolling_stats(df: pd.DataFrame, window_days: int = 7) -> dict:
    if df.empty:
        return {}
    cutoff = (datetime.utcnow() - timedelta(days=window_days)).date()
    w = df[df["delivery_local"] >= cutoff]
    if w.empty:
        return {}
    net_eur = float(w["settlement_eur"].sum())
    trades  = int(len(w))
    mwh     = float(w["mwh"].sum())
    hit_rate = None
    if "imbalance_spread_actual" in w.columns and "action" in w.columns:
        t = w[w["action"].isin(["BUY", "SELL"])]
        if len(t):
            correct = (
                ((t["action"] == "BUY")  & (t["imbalance_spread_actual"] > 0)) |
                ((t["action"] == "SELL") & (t["imbalance_spread_actual"] < 0))
            ).sum()
            hit_rate = float(correct) / len(t)
    cum  = w.sort_values("delivery_local")["settlement_eur"].cumsum()
    peak = cum.cummax()
    dd   = float((cum - peak).min()) if len(cum) else 0.0
    return {"window_days": window_days, "net_eur": net_eur, "trades": trades,
            "mwh": mwh, "hit_rate": hit_rate, "max_drawdown_eur": dd}


def cumulative_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return {"since":         str(df["delivery_local"].min()),
            "through":       str(df["delivery_local"].max()),
            "total_trades":  int(len(df)),
            "total_mwh":     float(df["mwh"].sum()),
            "total_eur":     float(df["settlement_eur"].sum())}


def quiet_model_check(days: int = QUIET_DAYS) -> dict:
    locked = load_all_locked()
    if locked.empty:
        return {"last_trade_utc": None, "days_quiet": None, "alert": True}
    trades = locked[locked["action"].isin(["BUY", "SELL"])]
    if trades.empty:
        return {"last_trade_utc": None, "days_quiet": None, "alert": True}
    last  = trades["locked_utc"].max()
    quiet = (datetime.utcnow() - last.to_pydatetime()).total_seconds() / 86400
    return {"last_trade_utc": str(last), "days_quiet": round(quiet, 1), "alert": quiet >= days}


# ═══════════════════════════════════════════════════════════════════════════════
#  Alerts
# ═══════════════════════════════════════════════════════════════════════════════

def check_alerts(stats7: dict, quiet: dict) -> list[str]:
    alerts = []
    if stats7.get("max_drawdown_eur", 0) <= MAX_DRAWDOWN_EUR:
        alerts.append(
            f"DRAWDOWN  7-day rolling drawdown {stats7['max_drawdown_eur']:,.0f} EUR "
            f"exceeds threshold {MAX_DRAWDOWN_EUR:,} EUR")
    if stats7.get("hit_rate") is not None and stats7["hit_rate"] < MIN_HIT_RATE:
        alerts.append(
            f"HIT_RATE  7-day direction accuracy {stats7['hit_rate']:.1%} "
            f"below minimum {MIN_HIT_RATE:.0%}")
    if quiet.get("alert"):
        days_q = quiet.get("days_quiet")
        msg = f"{days_q:.1f} days" if days_q is not None else "unknown (journal empty)"
        alerts.append(f"QUIET     No trades locked for {msg} — check live cycle (run.py)")
    return alerts


def write_alerts(alerts: list[str], ts: str):
    if not alerts:
        return
    lines = [f"\n{'='*60}", f"ALERT  {ts}"] + [f"  {a}" for a in alerts]
    with open(ALERT_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
#  Daily log entry
# ═══════════════════════════════════════════════════════════════════════════════

def write_daily_entry(ts: str, cum: dict, stats7: dict, quiet: dict, alerts: list[str]):
    entry = {"ts": ts, "cumulative": cum, "last_7_days": stats7,
             "quiet_check": quiet, "alerts": alerts}
    with open(DAILY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Weekly client report (Markdown)
# ═══════════════════════════════════════════════════════════════════════════════

def write_weekly_report(df: pd.DataFrame, cum: dict, stats7: dict) -> Path:
    now   = datetime.utcnow()
    week  = now.strftime("W%W_%Y")
    path  = WEEKLY_DIR / f"shadow_week_{week}.md"
    cutoff = (now - timedelta(days=7)).date()
    recent = df[df["delivery_local"] >= cutoff].copy() if not df.empty else pd.DataFrame()
    daily  = daily_summary(recent) if not recent.empty else pd.DataFrame()

    lines = [
        f"# Nurex V4.1 Shadow Monitor — Week {week}",
        f"*Generated {now.strftime('%Y-%m-%d %H:%M')} UTC (no BRP submission — paper trading)*",
        "",
        "## Cumulative Performance (journal start → now)",
        "",
    ]
    if cum:
        lines += [
            "| Metric | Value |", "|--------|-------|",
            f"| Period | {cum['since']} → {cum['through']} |",
            f"| Total trades | {cum['total_trades']:,} |",
            f"| Total MWh | {cum['total_mwh']:,.0f} MWh |",
            f"| **Net P&L** | **{cum['total_eur']:,.0f} EUR** |",
        ]
    else:
        lines.append("*No settled trades yet.*")

    lines += ["", "## Last 7 Days", ""]
    if stats7:
        hit = f"{stats7['hit_rate']:.1%}" if stats7.get("hit_rate") is not None else "N/A"
        lines += [
            "| Metric | Value |", "|--------|-------|",
            f"| Trades | {stats7['trades']:,} |",
            f"| MWh | {stats7['mwh']:,.0f} MWh |",
            f"| Net P&L | {stats7['net_eur']:,.0f} EUR |",
            f"| Max drawdown | {stats7['max_drawdown_eur']:,.0f} EUR |",
            f"| Direction hit rate | {hit} |",
        ]
    else:
        lines.append("*No settled trades in last 7 days.*")

    if not daily.empty:
        lines += ["", "### Daily Breakdown (last 7 days)", "",
                  "| Date | Area | Trades | MWh | Net EUR |",
                  "|------|------|--------|-----|---------|"]
        for _, row in daily.iterrows():
            lines.append(f"| {row['delivery_local']} | {row['area']} | {row['trades']} "
                         f"| {row['mwh']:,.0f} | {row['net_eur']:,.0f} |")

    lines += [
        "", "## Model Status", "",
        "| Item | Status |", "|------|--------|",
        "| DK1 model | models_v4_1_intraday/v4_1_intraday_DK1.pkl |",
        "| DK2 model | models_v4_1_intraday/v4_1_intraday_DK2.pkl |",
        "| Live cycle | run.py (Nurex_V41_Cycle task, every 15 min) |",
        "| BRP submission | **Not active** (Phase 6 pending) |",
        "", "## Notes", "",
        "- Shadow mode: decisions are locked and settled against actual EDS imbalance prices,",
        "  but no BRP orders are submitted.",
        "- P&L figures are *indicative* — real trading includes bid/ask spread and BRP settlement fees.",
        "- Weekly reports are generated automatically every Sunday at 23:45 CET.",
        "", "---",
        "*Nurex V4.1 Intraday Trading System | Danish Energy Trading*",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Main commands
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_daily():
    ts     = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    df     = load_journal()
    cum    = cumulative_stats(df)
    stats7 = rolling_stats(df, window_days=7)
    quiet  = quiet_model_check()

    print(f"\n{'='*60}")
    print(f"Nurex V4.1 Shadow Monitor  {ts}")
    print(f"{'='*60}")

    if cum:
        print(f"\nCumulative: {cum['total_trades']:,} trades  "
              f"{cum['total_mwh']:,.0f} MWh  {cum['total_eur']:,.0f} EUR")
        print(f"  Period: {cum['since']} → {cum['through']}")

    if stats7:
        hit = f"{stats7['hit_rate']:.1%}" if stats7.get("hit_rate") is not None else "N/A"
        print(f"\nLast 7 days: {stats7['trades']} trades  "
              f"{stats7['mwh']:,.0f} MWh  {stats7['net_eur']:,.0f} EUR  "
              f"DD {stats7['max_drawdown_eur']:,.0f} EUR  hit {hit}")

    if quiet.get("last_trade_utc"):
        print(f"\nLast trade locked: {quiet['last_trade_utc']}  ({quiet['days_quiet']:.1f} days ago)")
    else:
        print("\nNo trades locked yet.")

    alerts = check_alerts(stats7, quiet)
    write_alerts(alerts, ts)
    write_daily_entry(ts, cum, stats7, quiet, alerts)

    if datetime.utcnow().weekday() == WEEKLY_REPORT_DAY:
        path = write_weekly_report(df, cum, stats7)
        print(f"\nWeekly report: {path}")

    print(f"\nDaily log:  {DAILY_LOG}")
    if alerts:
        print(f"Alert log:  {ALERT_LOG}")
    print("="*60)


def cmd_status():
    df    = load_journal()
    cum   = cumulative_stats(df)
    quiet = quiet_model_check()
    if cum:
        print(f"Nurex V4.1 shadow | {cum['since']} → {cum['through']} | "
              f"{cum['total_trades']:,} trades | {cum['total_eur']:,.0f} EUR | "
              f"last trade {quiet.get('days_quiet', '?')}d ago")
    else:
        print("Nurex V4.1 shadow | journal empty — cycle not running?")


def cmd_week():
    df     = load_journal()
    cum    = cumulative_stats(df)
    stats7 = rolling_stats(df, window_days=7)
    path   = write_weekly_report(df, cum, stats7)
    print(f"Weekly report written: {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--status", action="store_true", help="quick one-line status")
    p.add_argument("--report", choices=["week"], help="force-write a report now")
    args = p.parse_args()
    if args.status:
        cmd_status()
    elif args.report == "week":
        cmd_week()
    else:
        cmd_daily()


if __name__ == "__main__":
    main()

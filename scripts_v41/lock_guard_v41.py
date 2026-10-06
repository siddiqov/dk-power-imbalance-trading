"""Nurex V4.1 lock guard (2026-10-06): makes sure no batch misses its deadline.

Runs a few minutes after the cycle's lock (scheduled task Nurex_V41_LockGuard, hourly at :37).
1. Reads the journals (SQLite - always readable, no database lock needed) and finds the batch whose
   deadline is in the next 15 minutes.
2. If every zone (and the shadow model) is already locked -> logs "ok" and exits (the normal case).
3. If not (cycle crashed, was skipped or was too slow) -> runs the normal lock, retrying every 20 s
   until 40 s before the deadline (the database may be busy for a moment with a download).
Locking twice is impossible: the journal keeps the first decision per quarter (INSERT OR IGNORE).
Log: logs/v41_lockguard.log
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "Nurex_V4_2")]
import pandas as pd  # noqa: E402
from v4_1_intraday import settings as S  # noqa: E402
from v4_1_intraday import journal as J  # noqa: E402
from nurex42.storage import Store  # noqa: E402


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    with open(ROOT / "logs" / "v41_lockguard.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def now_utc():
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def due_quarters(cfg, now):
    """Quarters whose batch deadline falls in (now, now + 15 min]."""
    qs = pd.Series(pd.date_range(now.floor("15min"), now + pd.Timedelta(hours=9), freq="15min"))
    dl = pd.Series(pd.DatetimeIndex(cfg.deadlines(qs)))
    m = (dl > now) & (dl <= now + pd.Timedelta(minutes=15))
    return qs[m.values].reset_index(drop=True), (dl[m.values].min() if m.any() else None)


def missing(cfg, qs):
    """(journal, area) pairs that do not have all of qs locked yet."""
    out = []
    for shadow, areas in ((False, list(cfg["areas"])), (True, J.shadow_areas(cfg))):
        for a in areas:
            j = J.read(cfg, area=a, start=qs.min(), end=qs.max() + pd.Timedelta(minutes=15), shadow=shadow)
            have = set(pd.to_datetime(j["quarter_utc"])) if len(j) else set()
            if not set(qs).issubset(have):
                out.append(("shadow " if shadow else "") + a)
    return out


def main():
    cfg = S.load()
    now = now_utc()
    qs, deadline = due_quarters(cfg, now)
    if not len(qs):
        log("no batch deadline in the next 15 min - nothing to check")
        return 0
    miss = missing(cfg, qs)
    if not miss:
        log(f"ok: batch {qs.min()}..{qs.max()} UTC already locked (deadline {deadline} UTC)")
        return 0
    log(f"NOT LOCKED: {miss} for batch {qs.min()}..{qs.max()} UTC, deadline {deadline} UTC - locking now")
    attempt = 0
    while now_utc() < deadline - pd.Timedelta(seconds=40):
        attempt += 1
        try:
            st = Store(cfg.db_path, read_only=True)
            try:
                r = J.lock(st, cfg)
                try:
                    rs = J.lock(st, cfg, shadow=True)
                except Exception as e:
                    rs = f"failed: {e}"
            finally:
                st.close()
            miss = missing(cfg, qs)
            log(f"attempt {attempt}: main {r} | shadow {rs} | still missing: {miss or 'none'}")
            if not miss:
                return 0
        except Exception as e:
            log(f"attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}")
        time.sleep(20)
    log(f"GAVE UP at the deadline - still missing {missing(cfg, qs)}")
    return 2


if __name__ == "__main__":
    sys.exit(main())

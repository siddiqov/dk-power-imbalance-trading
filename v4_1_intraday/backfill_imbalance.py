"""Re-fetch recently published imbalance prices and fill them into the V4.2 store.

Why this exists
---------------
Energi Data Service publishes a quarter's row early, carrying the spot price and aFRR
volumes but with `ImbalancePriceEUR` still null; the imbalance price itself lands ~20 min
after the quarter ends. The collector inserts that early row and never revisits it, so a
quarter captured in that window keeps a NULL imbalance price permanently. Those quarters
can never settle, never produce a PnL, and silently drop out of every total.

This tool re-reads a recent window from EDS and updates rows whose price has since been
published. It only writes a row when EDS has a non-null imbalance price, so it can never
overwrite good data with a null.

It deliberately depends on nothing but duckdb + requests: not on nurex42 (currently
bytecode only) and not on the collector module whose source was lost on 2026-09-19.

Usage
-----
    python -m v4_1_intraday.backfill_imbalance              # last 48 hours
    python -m v4_1_intraday.backfill_imbalance --hours 168  # last week
    python -m v4_1_intraday.backfill_imbalance --dry-run    # report only, write nothing
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests
import yaml

EDS_URL = "https://api.energidataservice.dk/dataset/ImbalancePrice"
V42_ROOT = Path(__file__).resolve().parent.parent / "Nurex_V4_2"

# EDS field -> store column. Only fields the store actually keeps.
FIELDS = {
    "PriceArea": "area",
    "TimeUTC": "time_utc",
    "SpotPriceEUR": "spot_eur",
    "ImbalancePriceEUR": "imbalance_eur",
    "SatisfiedDemand": "satisfied_demand_mw",
    "DominatingDirection": "dominating_direction",
    "aFRRUpMW": "afrr_up_mw",
    "aFRRDownMW": "afrr_down_mw",
    "aFRRVWAUpEUR": "afrr_vwa_up_eur",
    "aFRRVWADownEUR": "afrr_vwa_down_eur",
    "mFRRMarginalPriceUpEUR": "mfrr_price_up_eur",
    "mFRRMarginalPriceDownEUR": "mfrr_price_down_eur",
}


def db_path() -> Path:
    """Read db_path from the V4.2 config, resolved like IntradaySettings does."""
    with open(V42_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    p = Path(raw["db_path"])
    return p if p.is_absolute() else V42_ROOT / p


def fetch(hours: int, areas: list) -> list:
    """Recent EDS rows that carry a published imbalance price."""
    import datetime as dt
    start = (dt.datetime.utcnow() - dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M")
    params = {
        "start": start,
        "limit": 0,
        "sort": "TimeUTC DESC",
        "filter": '{"PriceArea":[' + ",".join('"%s"' % a for a in areas) + "]}",
    }
    r = requests.get(EDS_URL, params=params, timeout=60)
    r.raise_for_status()
    recs = r.json().get("records", [])
    # A null price means "not published yet" - skip, never write it over a stored value.
    return [x for x in recs if x.get("ImbalancePriceEUR") is not None]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hours", type=int, default=48, help="how far back to re-read (default 48)")
    ap.add_argument("--areas", default="DK1,DK2")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    ap.add_argument("--retries", type=int, default=3, help="retries when the store is locked")
    a = ap.parse_args(argv)
    areas = [x.strip() for x in a.areas.split(",") if x.strip()]

    import duckdb
    import pandas as pd

    recs = fetch(a.hours, areas)
    if not recs:
        print("EDS returned no rows with a published imbalance price - nothing to do.")
        return 0
    df = pd.DataFrame(recs)[list(FIELDS)].rename(columns=FIELDS)
    df["time_utc"] = pd.to_datetime(df["time_utc"])
    df["source"] = "eds_backfill"
    df["ingested_at"] = pd.Timestamp.utcnow().tz_localize(None)
    path = db_path()
    print("EDS: %d priced quarters in the last %dh  |  store: %s" % (len(df), a.hours, path))

    last, con = None, None
    for attempt in range(a.retries):
        try:
            con = duckdb.connect(str(path), read_only=a.dry_run)
            break
        except Exception as e:          # single writer: a collector or the dashboard has it
            last, con = e, None
            if attempt < a.retries - 1:
                time.sleep(3)
    if con is None:
        print("Store is busy, no changes made: %s" % last, file=sys.stderr)
        return 1

    try:
        con.register("incoming", df)
        gaps = con.execute("""
            SELECT count(*) FROM imbalance s JOIN incoming i
              ON s.area = i.area AND s.time_utc = i.time_utc
            WHERE s.imbalance_eur IS NULL""").fetchone()[0]
        missing = con.execute("""
            SELECT count(*) FROM incoming i
            WHERE NOT EXISTS (SELECT 1 FROM imbalance s
                              WHERE s.area = i.area AND s.time_utc = i.time_utc)""").fetchone()[0]
        print("stored rows with a missing price that EDS can now fill: %d" % gaps)
        print("quarters absent from the store entirely:                %d" % missing)

        if a.dry_run:
            print("dry run - nothing written.")
            return 0
        if gaps == 0 and missing == 0:
            print("store is already up to date.")
            return 0

        collist = ", ".join(df.columns)
        con.execute("BEGIN")
        con.execute("""
            DELETE FROM imbalance WHERE (area, time_utc) IN (
                SELECT s.area, s.time_utc FROM imbalance s JOIN incoming i
                  ON s.area = i.area AND s.time_utc = i.time_utc
                WHERE s.imbalance_eur IS NULL)""")
        con.execute("""
            INSERT INTO imbalance (%s)
            SELECT %s FROM incoming i
            WHERE NOT EXISTS (SELECT 1 FROM imbalance s
                              WHERE s.area = i.area AND s.time_utc = i.time_utc)""" % (collist, collist))
        con.execute("COMMIT")
        left = con.execute("""
            SELECT count(*) FROM imbalance
            WHERE imbalance_eur IS NULL
              AND time_utc >= (SELECT min(time_utc) FROM incoming)""").fetchone()[0]
        print("filled %d gap(s), inserted %d new quarter(s)." % (gaps, missing))
        print("rows still without a price in that window: %d "
              "(quarters EDS has not published yet)" % left)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())

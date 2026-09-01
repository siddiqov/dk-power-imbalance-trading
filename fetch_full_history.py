# ==============================================================================
# fetch_full_history.py
# Fetches the complete date range March 2025 -> now in monthly batches,
# upserting all records into energy_data.db via data_retrieval.DataPipeline.
# This avoids the 25,000-record API page limit by issuing one request per month.
# ==============================================================================

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import duckdb
import warnings
warnings.filterwarnings('ignore')

from data_retrieval import DataPipeline

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
AREA       = 'DK1'
START_DATE = datetime(2025, 3, 4)
END_DATE   = datetime.now()
DB_PATH    = 'energy_data.db'

# -----------------------------------------------------------------------
# Build monthly windows
# -----------------------------------------------------------------------
def monthly_windows(start, end):
    windows = []
    cur = start.replace(day=1)
    while cur <= end:
        window_start = max(start, cur)
        window_end   = min(end, cur + relativedelta(months=1) - timedelta(seconds=1))
        windows.append((window_start, window_end))
        cur += relativedelta(months=1)
    return windows

# -----------------------------------------------------------------------
# Check current DB state
# -----------------------------------------------------------------------
def db_summary():
    conn = duckdb.connect(DB_PATH)
    tables = ['imbalance_prices', 'day_ahead_prices', 'afrr_activation',
              'mfrr_activation', 'electricity_balance', 'forecasts_hour']
    print("\n  Current DuckDB state:")
    for tbl in tables:
        try:
            r = conn.execute(
                f"SELECT COUNT(*) as n, MIN(time_utc) as mn, MAX(time_utc) as mx FROM {tbl}"
            ).fetchdf()
            print(f"    {tbl:<25} {r.n[0]:>7,} rows   {r.mn[0]} -> {r.mx[0]}")
        except:
            print(f"    {tbl:<25} (empty / missing)")
    conn.close()

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == '__main__':
    windows = monthly_windows(START_DATE, END_DATE)
    pipeline = DataPipeline(DB_PATH)

    print(f"\n{'='*65}")
    print(f"  FULL HISTORY FETCH  |  {AREA}  |  {START_DATE.date()} -> {END_DATE.date()}")
    print(f"  Monthly batches: {len(windows)}")
    print(f"{'='*65}")

    db_summary()

    total = {'imbalance_prices': 0, 'day_ahead_prices': 0,
             'electricity_balance': 0, 'forecasts_hour': 0,
             'afrr_activation': 0, 'mfrr_activation': 0}

    for i, (w_start, w_end) in enumerate(windows, 1):
        label = w_start.strftime('%Y-%m')
        print(f"\n  [{i:02d}/{len(windows)}] Batch {label}  ({w_start.date()} -> {w_end.date()})")
        try:
            results = pipeline.fetch_all_data(
                AREA, w_start, w_end, export_csv=False
            )
            for tbl, cnt in results.items():
                total[tbl] = total.get(tbl, 0) + cnt
                print(f"    {tbl}: +{cnt:,}")
        except Exception as e:
            print(f"    ERROR in batch {label}: {e}")

    print(f"\n{'='*65}")
    print("  FETCH COMPLETE — TOTALS UPSERTED")
    print(f"{'='*65}")
    for tbl, cnt in total.items():
        print(f"    {tbl}: {cnt:,} records processed")

    db_summary()
    print("\n  Done. Run run_backtest_pipeline.py to train on the full dataset.")

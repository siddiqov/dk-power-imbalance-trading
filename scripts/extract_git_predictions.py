#!/usr/bin/env python3
"""
scripts/extract_git_predictions.py
----------------------------------
Extracts complete historical prediction records across all Git branches and commits
for results/96Q_future_table_DK1.csv and results/96Q_future_table_DK2.csv.

Key features:
1. Non-destructive Git inspection: Uses `git log --all` and `git show <commit>:<file>` via stdout
   (never calls `git checkout` or touches uncommitted working directory changes).
2. Schema normalization & data sanitization: Normalizes columns, cleans numeric values (strips €, DKK),
   converts `--`/missing to SQL NULL, standardizes ISO-8601 timestamps.
3. Version tagging: Accurately categorizes commits into 'v2', 'v3', or 'legacy'.
4. Structured SQLite database: Saves to data/historical_predictions.db with relational tables,
   foreign keys, composite indexes, and Supabase-ready deduplicated views.
"""

import os
import sys
import re
import csv
import io
import sqlite3
import subprocess
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join("data", "historical_predictions.db")
TARGET_FILES = [
    "results/96Q_future_table_DK1.csv",
    "results/96Q_future_table_DK2.csv"
]

COL_MAPPING = {
    "quarter": ["quarter", "quarter_str", "q"],
    "time_dk": ["time_dk", "time_cet", "delivery_time_dk", "time_copenhagen", "time"],
    "time_utc": ["time_utc", "delivery_time_utc"],
    "spot_price_eur": ["spot_price_eur", "spot_price", "spot_eur", "spot"],
    "deep_bilstm_eur": ["deep_bilstm_eur", "deep_bilstm", "bilstm_eur", "bilstm"],
    "transformer_tft_eur": ["transformer_tft_eur", "transformer_tft", "tft_eur", "tft"],
    "transfer_lgb_eur": ["transfer_lgb_eur", "transfer_lgb", "lightgbm_eur", "lgb_eur", "lgb"],
    "hierarchical_eur": ["hierarchical_eur", "hierarchical_lgb_xgb_eur", "hierarchical"],
    "pure15m_catboost_eur": ["pure15m_catboost_eur", "catboost_eur", "catboost"],
    "meta_ensemble_eur": ["meta_ensemble_eur", "ensemble_eur", "meta_ensemble"],
    "meta_ensemble_dkk": ["meta_ensemble_dkk", "ensemble_dkk"],
    "actual_settled_imbalance_eur": [
        "actual_settled_imbalance_eur", "actual_settled_eur", 
        "settled_imbalance_eur", "settled_price_eur", "actual_imbalance_eur"
    ],
    "predicted_direction": ["predicted_direction", "direction", "pred_direction"],
    "agent_action": ["agent_action", "action", "recommended_action", "trade_action"],
    "status": ["status", "trade_status"]
}


def clean_numeric(val: Any) -> Optional[float]:
    """Clean string representation of float: strip €, DKK, commas, whitespace. Return None for missing."""
    if val is None:
        return None
    val_str = str(val).strip()
    if not val_str or val_str in ("--", "nan", "None", "null", "N/A", "NA", "-", "—"):
        return None
    
    clean = val_str.replace("€", "").replace("DKK", "").replace("dkk", "").replace("EUR", "").replace("eur", "")
    clean = re.sub(r'\s+', '', clean)
    try:
        return float(clean)
    except ValueError:
        return None


def clean_datetime(val: Any) -> Optional[str]:
    """Clean datetime to standard ISO-8601 string: YYYY-MM-DDTHH:MM:SS."""
    if val is None:
        return None
    val_str = str(val).strip()
    if not val_str or val_str in ("--", "nan", "None", "null"):
        return None
    
    # Standardize separator to T
    val_str = val_str.replace(" ", "T")
    if len(val_str) == 16:  # YYYY-MM-DDTHH:MM
        val_str += ":00"
    return val_str


def tag_version(commit_msg: str) -> str:
    """Tag pipeline version based on commit message."""
    msg_lower = commit_msg.lower()
    if "feat(v3)" in msg_lower or "fix(v3)" in msg_lower or "v3" in msg_lower:
        return "v3"
    elif "feat(v2)" in msg_lower or "fix(v2)" in msg_lower or "v2" in msg_lower:
        return "v2"
    elif "v1" in msg_lower:
        return "v1"
    else:
        return "legacy"


def init_database(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite database with required tables, indexes, and views."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            commit_hash TEXT NOT NULL,
            short_hash TEXT NOT NULL,
            commit_date TEXT NOT NULL,
            commit_message TEXT NOT NULL,
            version_pipeline TEXT NOT NULL,
            file_path TEXT NOT NULL,
            bidding_zone TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            extracted_at TEXT NOT NULL,
            UNIQUE(commit_hash, bidding_zone)
        );
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            bidding_zone TEXT NOT NULL,
            quarter TEXT,
            time_utc TEXT,
            time_dk TEXT,
            spot_price_eur REAL,
            deep_bilstm_eur REAL,
            transformer_tft_eur REAL,
            transfer_lgb_eur REAL,
            hierarchical_eur REAL,
            pure15m_catboost_eur REAL,
            meta_ensemble_eur REAL,
            meta_ensemble_dkk REAL,
            actual_settled_imbalance_eur REAL,
            predicted_direction TEXT,
            agent_action TEXT,
            status TEXT,
            version_pipeline TEXT,
            commit_hash TEXT,
            FOREIGN KEY (snapshot_id) REFERENCES prediction_snapshots(snapshot_id) ON DELETE CASCADE
        );
        """)
        
        # Composite Indexes
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pred_history_zone_time_ver 
        ON predictions_history(bidding_zone, time_utc, version_pipeline);
        """)
        
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pred_history_commit_zone 
        ON predictions_history(commit_hash, bidding_zone);
        """)
        
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pred_snapshots_commit_zone 
        ON prediction_snapshots(commit_hash, bidding_zone);
        """)
        
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pred_snapshots_version 
        ON prediction_snapshots(version_pipeline);
        """)
        
        # Supabase-ready deduplicated view
        conn.execute("""
        CREATE VIEW IF NOT EXISTS v_latest_predictions AS
        SELECT p.*
        FROM predictions_history p
        JOIN (
            SELECT bidding_zone, time_utc, version_pipeline, MAX(id) as max_id
            FROM predictions_history
            GROUP BY bidding_zone, time_utc, version_pipeline
        ) latest ON p.id = latest.max_id;
        """)

    return conn


def get_commit_history(target_files: List[str]) -> List[Dict[str, str]]:
    """Query git log across all branches for commits that modified target files."""
    cmd = [
        "git", "log", "--all",
        "--format=%H|%ad|%s",
        "--date=iso",
        "--"
    ] + target_files
    
    proc = subprocess.run(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True, 
        encoding="utf-8", 
        errors="replace", 
        check=True
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    
    commits = []
    seen_hashes = set()
    for line in lines:
        parts = line.split("|", 2)
        if len(parts) == 3:
            chash, cdate, cmsg = parts
            if chash not in seen_hashes:
                seen_hashes.add(chash)
                commits.append({
                    "commit_hash": chash,
                    "short_hash": chash[:7],
                    "commit_date": cdate,
                    "commit_message": cmsg,
                    "version_pipeline": tag_version(cmsg)
                })
    return commits


def fetch_file_content_at_commit(commit_hash: str, file_path: str) -> Optional[str]:
    """Extract file content directly from git stdout using git show without modifying working tree."""
    cmd = ["git", "show", f"{commit_hash}:{file_path}"]
    proc = subprocess.run(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True, 
        encoding="utf-8", 
        errors="replace"
    )
    if proc.returncode != 0:
        # File might not have existed at this commit
        return None
    return proc.stdout


def map_columns(header: List[str]) -> Dict[str, str]:
    """Map raw CSV header columns to normalized database column names."""
    norm_to_raw = {}
    normalized_raw = {re.sub(r'[^a-z0-9_]', '', col.lower().strip().replace(' ', '_')): col for col in header}
    
    for db_col, aliases in COL_MAPPING.items():
        for alias in aliases:
            norm_alias = re.sub(r'[^a-z0-9_]', '', alias.lower())
            if norm_alias in normalized_raw:
                norm_to_raw[db_col] = normalized_raw[norm_alias]
                break
    return norm_to_raw


def process_commit_file(conn: sqlite3.Connection, commit_info: Dict[str, str], file_path: str, extracted_at: str) -> int:
    """Process a single file at a specific commit and insert records into SQLite."""
    content = fetch_file_content_at_commit(commit_info["commit_hash"], file_path)
    if not content:
        return 0
    
    zone = "DK1" if "DK1" in file_path else ("DK2" if "DK2" in file_path else "UNKNOWN")
    
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        return 0
    
    col_map = map_columns(reader.fieldnames)
    
    # Read and parse rows
    rows_to_insert = []
    for raw_row in reader:
        row_dict = {}
        for db_col in COL_MAPPING.keys():
            raw_col_name = col_map.get(db_col)
            raw_val = raw_row.get(raw_col_name) if raw_col_name else None
            
            if db_col in ("spot_price_eur", "deep_bilstm_eur", "transformer_tft_eur", 
                          "transfer_lgb_eur", "hierarchical_eur", "pure15m_catboost_eur", 
                          "meta_ensemble_eur", "meta_ensemble_dkk", "actual_settled_imbalance_eur"):
                row_dict[db_col] = clean_numeric(raw_val)
            elif db_col in ("time_dk", "time_utc"):
                row_dict[db_col] = clean_datetime(raw_val)
            else:
                row_dict[db_col] = str(raw_val).strip() if raw_val is not None else None
        
        rows_to_insert.append(row_dict)
    
    row_count = len(rows_to_insert)
    if row_count == 0:
        return 0
    
    cursor = conn.cursor()
    
    # Insert snapshot record (or update if already exists)
    cursor.execute("""
    INSERT INTO prediction_snapshots 
    (commit_hash, short_hash, commit_date, commit_message, version_pipeline, file_path, bidding_zone, row_count, extracted_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(commit_hash, bidding_zone) DO UPDATE SET
        row_count = excluded.row_count,
        extracted_at = excluded.extracted_at
    """, (
        commit_info["commit_hash"],
        commit_info["short_hash"],
        commit_info["commit_date"],
        commit_info["commit_message"],
        commit_info["version_pipeline"],
        file_path,
        zone,
        row_count,
        extracted_at
    ))
    
    snapshot_id = cursor.lastrowid
    if not snapshot_id:
        cursor.execute("SELECT snapshot_id FROM prediction_snapshots WHERE commit_hash = ? AND bidding_zone = ?", 
                       (commit_info["commit_hash"], zone))
        snapshot_id = cursor.fetchone()[0]
    
    # Delete existing predictions for this snapshot to prevent duplicates on rerun
    cursor.execute("DELETE FROM predictions_history WHERE snapshot_id = ?", (snapshot_id,))
    
    # Insert predictions history rows
    insert_records = []
    for r in rows_to_insert:
        insert_records.append((
            snapshot_id,
            zone,
            r["quarter"],
            r["time_utc"],
            r["time_dk"],
            r["spot_price_eur"],
            r["deep_bilstm_eur"],
            r["transformer_tft_eur"],
            r["transfer_lgb_eur"],
            r["hierarchical_eur"],
            r["pure15m_catboost_eur"],
            r["meta_ensemble_eur"],
            r["meta_ensemble_dkk"],
            r["actual_settled_imbalance_eur"],
            r["predicted_direction"],
            r["agent_action"],
            r["status"],
            commit_info["version_pipeline"],
            commit_info["commit_hash"]
        ))
    
    cursor.executemany("""
    INSERT INTO predictions_history (
        snapshot_id, bidding_zone, quarter, time_utc, time_dk,
        spot_price_eur, deep_bilstm_eur, transformer_tft_eur, transfer_lgb_eur,
        hierarchical_eur, pure15m_catboost_eur, meta_ensemble_eur, meta_ensemble_dkk,
        actual_settled_imbalance_eur, predicted_direction, agent_action, status,
        version_pipeline, commit_hash
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, insert_records)
    
    conn.commit()
    return row_count


def run_extraction():
    """Main extraction routine."""
    print("=" * 80)
    print("  HISTORICAL PREDICTIONS GIT EXTRACTION TO SQLITE")
    print(f"  Target DB: {DB_PATH}")
    print("=" * 80)
    
    conn = init_database(DB_PATH)
    commits = get_commit_history(TARGET_FILES)
    print(f"\nFound {len(commits)} unique Git commits that modified target forecast tables.\n")
    
    extracted_at = datetime.now(timezone.utc).isoformat()
    
    total_snapshots = 0
    total_rows = 0
    
    for idx, c in enumerate(commits, 1):
        print(f"[{idx:02d}/{len(commits):02d}] Commit: {c['short_hash']} | Date: {c['commit_date'][:19]} | Ver: {c['version_pipeline']}")
        print(f"     Msg: {c['commit_message'][:70]}")
        
        for fpath in TARGET_FILES:
            zone = "DK1" if "DK1" in fpath else "DK2"
            rows = process_commit_file(conn, c, fpath, extracted_at)
            if rows > 0:
                print(f"     -> {zone}: Extracted {rows} rows from {fpath}")
                total_snapshots += 1
                total_rows += rows
            else:
                print(f"     -> {zone}: Not present in commit")
    
    print("\n" + "=" * 80)
    print("  EXTRACTION COMPLETED - SUMMARY REPORT")
    print("=" * 80)
    
    cursor = conn.cursor()
    
    # Breakdown by version
    print("\n--- Snapshots & Rows by Pipeline Version ---")
    cursor.execute("""
    SELECT version_pipeline, COUNT(DISTINCT commit_hash) as commits, COUNT(*) as snapshots, SUM(row_count) as total_rows
    FROM prediction_snapshots
    GROUP BY version_pipeline
    ORDER BY version_pipeline DESC
    """)
    for row in cursor.fetchall():
        print(f"  Pipeline {row[0]:<8} | Commits: {row[1]:2d} | Snapshots: {row[2]:2d} | Rows: {row[3]:,d}")
    
    # Breakdown by bidding zone
    print("\n--- Rows by Bidding Zone ---")
    cursor.execute("""
    SELECT bidding_zone, COUNT(*) as rows, COUNT(DISTINCT snapshot_id) as snapshots
    FROM predictions_history
    GROUP BY bidding_zone
    ORDER BY bidding_zone
    """)
    for row in cursor.fetchall():
        print(f"  Zone {row[0]:<4} | Snapshots: {row[2]:2d} | Total Rows: {row[1]:,d}")
        
    # Settled vs Unsettled
    print("\n--- Settlement Status Breakdown ---")
    cursor.execute("""
    SELECT 
        CASE WHEN actual_settled_imbalance_eur IS NOT NULL THEN 'Settled (Genuine API Value)' ELSE 'Pending / Unsettled (NULL)' END as status_cat,
        COUNT(*) as cnt
    FROM predictions_history
    GROUP BY status_cat
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:<32} : {row[1]:,d} rows")
        
    # Total overall
    cursor.execute("SELECT COUNT(*) FROM prediction_snapshots")
    snap_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM predictions_history")
    hist_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM v_latest_predictions")
    view_count = cursor.fetchone()[0]
    
    print(f"\nTotal Snapshots Stored: {snap_count}")
    print(f"Total Prediction Rows:  {hist_count:,d}")
    print(f"Deduplicated View Rows: {view_count:,d}")
    print(f"SQLite DB File Size:    {os.path.getsize(DB_PATH) / 1024:.1f} KB")
    print("=" * 80 + "\n")
    
    conn.close()


if __name__ == "__main__":
    run_extraction()

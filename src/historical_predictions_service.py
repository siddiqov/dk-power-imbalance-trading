"""
src/historical_predictions_service.py
-------------------------------------
Service layer for querying and analyzing historical predictions from SQLite
(data/historical_predictions.db). Supports custom date ranges, zone filters,
pipeline versions, snapshot inspection, and model accuracy benchmarking (MAE).
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "historical_predictions.db")


def get_db_connection() -> sqlite3.Connection:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Historical predictions database not found at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_all_snapshots() -> List[Dict[str, Any]]:
    """Retrieve all available commit snapshots for dropdown selection."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT snapshot_id, commit_hash, short_hash, commit_date, commit_message,
           version_pipeline, file_path, bidding_zone, row_count, extracted_at
    FROM prediction_snapshots
    ORDER BY snapshot_id ASC
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def query_historical_predictions(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    zone: Optional[str] = None,
    version: Optional[str] = None,
    snapshot_id: Optional[int] = None,
    status: Optional[str] = None,
    view_mode: str = "all",
    limit: int = 5000
) -> Dict[str, Any]:
    """
    Query historical prediction records with custom filters.
    view_mode: 'all' (queries predictions_history) or 'deduplicated' (queries v_latest_predictions)
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    conditions = []
    params = []

    table_name = "v_latest_predictions" if view_mode == "deduplicated" else "predictions_history"

    if start_date:
        # Match time_dk >= 'YYYY-MM-DD'
        conditions.append("time_dk >= ?")
        params.append(f"{start_date}T00:00:00" if "T" not in start_date else start_date)

    if end_date:
        # Match time_dk <= 'YYYY-MM-DD 23:59:59'
        conditions.append("time_dk <= ?")
        params.append(f"{end_date}T23:59:59" if "T" not in end_date else end_date)

    if zone and zone.upper() in ("DK1", "DK2"):
        conditions.append("bidding_zone = ?")
        params.append(zone.upper())

    if version and version.lower() in ("v2", "v3"):
        conditions.append("version_pipeline = ?")
        params.append(version.lower())

    if snapshot_id and view_mode != "deduplicated":
        conditions.append("snapshot_id = ?")
        params.append(int(snapshot_id))

    if status:
        if status.lower() == "settled":
            conditions.append("actual_settled_imbalance_eur IS NOT NULL")
        elif status.lower() in ("pending", "unsettled"):
            conditions.append("actual_settled_imbalance_eur IS NULL")

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    query = f"""
    SELECT id, snapshot_id, bidding_zone, quarter, time_utc, time_dk,
           spot_price_eur, deep_bilstm_eur, transformer_tft_eur, transfer_lgb_eur,
           hierarchical_eur, pure15m_catboost_eur, meta_ensemble_eur, meta_ensemble_dkk,
           actual_settled_imbalance_eur, predicted_direction, agent_action, status,
           version_pipeline, commit_hash
    FROM {table_name}
    {where_clause}
    ORDER BY time_dk ASC, bidding_zone ASC, snapshot_id ASC
    LIMIT ?
    """
    params.append(limit)

    cursor.execute(query, params)
    raw_records = [dict(r) for r in cursor.fetchall()]

    # Compute summary metrics & MAE
    total_records = len(raw_records)
    settled_records = [r for r in raw_records if r["actual_settled_imbalance_eur"] is not None]
    settled_count = len(settled_records)
    pending_count = total_records - settled_count

    # Model accuracy on settled records
    mae_metrics = {}
    models_to_check = [
        ("transformer_tft_eur", "Transformer-TFT"),
        ("transfer_lgb_eur", "Transfer-LightGBM"),
        ("deep_bilstm_eur", "Deep-BiLSTM"),
        ("pure15m_catboost_eur", "Pure15m-CatBoost"),
        ("meta_ensemble_eur", "Meta-Ensemble"),
        ("hierarchical_eur", "Hierarchical-LGBM+XGB")
    ]

    for col_name, display_name in models_to_check:
        errors = [
            abs(r[col_name] - r["actual_settled_imbalance_eur"])
            for r in settled_records
            if r[col_name] is not None
        ]
        if errors:
            mae_metrics[display_name] = round(sum(errors) / len(errors), 2)
        else:
            mae_metrics[display_name] = None

    # Determine best performing model
    valid_maes = {k: v for k, v in mae_metrics.items() if v is not None}
    best_model = min(valid_maes, key=valid_maes.get) if valid_maes else "N/A"

    # Format TSV payload for 1-click Excel clipboard copy
    tsv_headers = [
        "Commit", "Version", "Zone", "Quarter", "Delivery Time (DK)", "Delivery Time (UTC)",
        "Spot (€/MWh)", "TFT (€/MWh)", "LGB (€/MWh)", "BiLSTM (€/MWh)", "CatBoost (€/MWh)",
        "Meta-Ensemble (€/MWh)", "Actual Settled (€/MWh)", "Direction", "Action", "Status"
    ]
    tsv_lines = ["\t".join(tsv_headers)]

    formatted_records = []
    for r in raw_records:
        short_hash = r["commit_hash"][:7] if r["commit_hash"] else "---"
        t_dk = r["time_dk"].replace("T", " ") if r["time_dk"] else ""
        t_utc = r["time_utc"].replace("T", " ") if r["time_utc"] else ""

        spot_str = f"{r['spot_price_eur']:.2f}" if r["spot_price_eur"] is not None else "--"
        tft_str = f"{r['transformer_tft_eur']:.2f}" if r["transformer_tft_eur"] is not None else "--"
        lgb_str = f"{r['transfer_lgb_eur']:.2f}" if r["transfer_lgb_eur"] is not None else "--"
        bilstm_str = f"{r['deep_bilstm_eur']:.2f}" if r["deep_bilstm_eur"] is not None else "--"
        catboost_str = f"{r['pure15m_catboost_eur']:.2f}" if r["pure15m_catboost_eur"] is not None else "--"
        ensemble_str = f"{r['meta_ensemble_eur']:.2f}" if r["meta_ensemble_eur"] is not None else "--"
        settled_str = f"{r['actual_settled_imbalance_eur']:.2f}" if r["actual_settled_imbalance_eur"] is not None else "--"

        tsv_lines.append("\t".join([
            short_hash, r["version_pipeline"].upper(), r["bidding_zone"], r["quarter"] or "",
            t_dk, t_utc, spot_str, tft_str, lgb_str, bilstm_str, catboost_str,
            ensemble_str, settled_str, r["predicted_direction"] or "",
            r["agent_action"] or "", r["status"] or ""
        ]))

        formatted_records.append({
            "id": r["id"],
            "snapshot_id": r["snapshot_id"],
            "short_hash": short_hash,
            "commit_hash": r["commit_hash"],
            "version_pipeline": r["version_pipeline"],
            "bidding_zone": r["bidding_zone"],
            "quarter": r["quarter"],
            "time_dk": t_dk,
            "time_utc": t_utc,
            "spot_price_eur": r["spot_price_eur"],
            "transformer_tft_eur": r["transformer_tft_eur"],
            "transfer_lgb_eur": r["transfer_lgb_eur"],
            "deep_bilstm_eur": r["deep_bilstm_eur"],
            "pure15m_catboost_eur": r["pure15m_catboost_eur"],
            "meta_ensemble_eur": r["meta_ensemble_eur"],
            "actual_settled_imbalance_eur": r["actual_settled_imbalance_eur"],
            "predicted_direction": r["predicted_direction"],
            "agent_action": r["agent_action"],
            "status": r["status"]
        })

    conn.close()

    return {
        "success": True,
        "total_records": total_records,
        "settled_count": settled_count,
        "pending_count": pending_count,
        "view_mode": view_mode,
        "mae_metrics": mae_metrics,
        "best_model": best_model,
        "records": formatted_records,
        "tsv_payload": "\n".join(tsv_lines)
    }

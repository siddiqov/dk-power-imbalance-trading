#!/usr/bin/env python3
# ==============================================================================
# scripts/publish_tournament_portal.py
# Pushes the latest quarters from data/tournament/trades_all.csv to Supabase
# for the Bidder Portal:
#   - DK1: StormHunter (DK1_idt_shadow)
#   - DK2: StormHunter (DK2_stormhunter), StormFusion (DK2_fusion), NeuralFlash AI (DK2_chronos)
# ==============================================================================

import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from dotenv import load_dotenv
load_dotenv(BASE / ".env")

from src.supabase_publisher import SupabasePublisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TournamentPublisher")

# Model mapping: internal model id -> (area, portal model_name)
MODEL_MAPPING = {
    "DK1_idt_shadow": ("DK1", "StormHunter"),
    "DK2_stormhunter": ("DK2", "StormHunter"),
    "DK2_fusion": ("DK2", "StormFusion"),
    "DK2_chronos": ("DK2", "NeuralFlash AI"),
}


def publish_from_trades_all(csv_path: Path = None, days_back: int = 3) -> int:
    """Reads trades_all.csv and publishes recent quarters for the target models."""
    csv_file = csv_path or (BASE / "data" / "tournament" / "trades_all.csv")
    if not csv_file.exists():
        logger.warning(f"File not found: {csv_file}")
        return 0

    publisher = SupabasePublisher()
    if not getattr(publisher, "client", None) and not getattr(publisher, "use_rest", False):
        logger.info("Supabase credentials not configured. Skipping remote publish.")
        return 0

    df = pd.read_csv(csv_file)
    if df.empty:
        logger.warning("trades_all.csv is empty")
        return 0

    # Filter to recent dates
    df["dt"] = pd.to_datetime(df["day_local"], errors="coerce")
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days_back)
    df_recent = df[df["dt"] >= cutoff].copy()

    total_pushed = 0

    for internal_id, (area, portal_model_name) in MODEL_MAPPING.items():
        m_df = df_recent[(df_recent["model"] == internal_id) & (df_recent["area"] == area)].copy()
        if m_df.empty:
            continue

        # Group by delivery day to upsert
        for day_str, day_group in m_df.groupby("day_local"):
            trades = []
            for _, row in day_group.iterrows():
                try:
                    t = pd.to_datetime(row["quarter_local"])
                    q_idx = t.hour * 4 + t.minute // 15 + 1
                except Exception:
                    continue

                act = str(row.get("action") or "HOLD").upper()
                if act == "BUY":
                    decision = "BUY Spot (Long)"
                    direction = 1
                elif act == "SELL":
                    decision = "SELL Spot (Short)"
                    direction = -1
                else:
                    decision = "HOLD"
                    direction = 0

                spot_val = float(row.get("da_eur") or row.get("fill_eur") or 0.0)
                edge_val = float(row.get("edge") or 0.0) if pd.notna(row.get("edge")) else float(row.get("q50") or 0.0)
                pred_imb = round(spot_val + edge_val, 2)

                status_val = str(row.get("status") or "LOCKED")
                if status_val == "SETTLED" and pd.notna(row.get("imbalance_eur")):
                    status_val = "SETTLED_AUDITED"

                p_up = float(row.get("p_up") or 0.0)
                p_down = float(row.get("p_down") or 0.0)

                pnl = row.get("pnl_eur")
                net_pnl = float(pnl) if pd.notna(pnl) else None

                imb_eur = row.get("imbalance_eur")
                actual_settled = float(imb_eur) if pd.notna(imb_eur) else None

                trade_dict = {
                    "quarter": f"Q{q_idx}",
                    "action": decision,
                    "direction": direction,
                    "status": status_val,
                    "mwh": float(row.get("mwh") or 0.0),
                    "spot_price_eur": spot_val,
                    "pred_imbalance_eur": pred_imb,
                    "pred_spread_eur": edge_val,
                    "p_up": p_up,
                    "p_down": p_down,
                    "q50_price_eur": float(row.get("q50") or 0.0) if pd.notna(row.get("q50")) else None,
                    "actual_settled_eur": actual_settled,
                    "net_pnl_eur": net_pnl,
                }
                trades.append(trade_dict)

            if trades:
                count = publisher.upsert_predictions(
                    trades=trades,
                    price_area=area,
                    delivery_date=str(day_str),
                    model_name=portal_model_name,
                    market_mode="INTRADAY_D0",
                )
                total_pushed += count

    logger.info(f"✅ Published {total_pushed} quarters from trades_all.csv to Supabase")
    return total_pushed


if __name__ == "__main__":
    publish_from_trades_all()

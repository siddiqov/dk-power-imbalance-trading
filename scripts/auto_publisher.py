#!/usr/bin/env python3
# ==============================================================================
# scripts/auto_publisher.py
# 24/7 Continuous Background Publisher for Power Trading Bidder Portal
#
# Runs every 15 minutes automatically:
# - Ingests authentic Energi Data Service records (DayAheadPrices & ImbalancePrice)
# - Evaluates all 6 tournament models across DK1 & DK2
# - Locks Pre-Market Bids & Predictions (Intraday + Day-Ahead) to Supabase & SQLite
# - Reconciles live settlements and realized PnL every 15 minutes
# ==============================================================================

import os
import sys
import time
import logging
from datetime import datetime, timedelta

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.push_predictions import push_predictions
from src.supabase_publisher import ALL_MODELS

# Setup dual logging (Terminal + File)
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
log_file = os.path.join(PROJECT_ROOT, "logs", "auto_publisher.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("AutoPublisher")


def get_seconds_until_next_quarter():
    """Calculates seconds until the next 15-minute clock boundary (:00, :15, :30, :45) + 30s buffer."""
    now = datetime.now()
    minute = now.minute
    next_minute = ((minute // 15) + 1) * 15
    if next_minute == 60:
        target = now.replace(minute=0, second=30, microsecond=0) + timedelta(hours=1)
    else:
        target = now.replace(minute=next_minute, second=30, microsecond=0)
    delay = (target - now).total_seconds()
    return max(30, delay)


def main():
    logger.info("=" * 70)
    logger.info("🚀 24/7 POWER TRADING MULTI-MODEL AUDIT LEDGER PUBLISHER STARTED")
    logger.info(f"   Project Root: {PROJECT_ROOT}")
    logger.info(f"   Log File:     {log_file}")
    logger.info(f"   Active Models: {', '.join(ALL_MODELS)}")
    logger.info("=" * 70)

    iteration = 1
    while True:
        cycle_start = datetime.now()
        logger.info(f"\n[Cycle #{iteration}] Starting multi-model ledger update at {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}...")

        try:
            # 1. Multi-Model Intraday & Day-Ahead push + Settlement Reconciliation
            total_pushed = push_predictions(
                delivery_date=None,
                price_areas=["DK1", "DK2"],
                models=ALL_MODELS,
                market_modes=["INTRADAY_D0", "DAY_AHEAD_D1"],
                reconcile=True,
            )

            # 2. Generate and lock local Day-Ahead Auction Bidding Sheets
            try:
                from src.day_ahead_auction_engine import DayAheadAuctionEngine
                for area in ["DK1", "DK2"]:
                    auction_engine = DayAheadAuctionEngine(price_area=area)
                    auction_res = auction_engine.generate_fixed_auction_bids()
                    logger.info(
                        f"  📋 [Day-Ahead Auction] Locked 96Q Bids for {area} "
                        f"({auction_res['delivery_date']}): {auction_res['summary_metrics']['total_bidded_mw']} MW bidded."
                    )
            except Exception as e_auc:
                logger.warning(f"  ⚠️ Day-Ahead local sheet generation note: {e_auc}")

            logger.info(f"[Cycle #{iteration}] ✅ Cycle complete! Updated {total_pushed} records across Supabase & Local Ledger.")

        except KeyboardInterrupt:
            logger.info("\n🛑 Publisher stopped by user.")
            break
        except Exception as e:
            logger.error(f"[Cycle #{iteration}] ❌ Error during publisher cycle: {e}", exc_info=True)
            logger.info("Retrying in 60 seconds...")
            time.sleep(60)
            continue

        iteration += 1
        sleep_sec = get_seconds_until_next_quarter()
        next_run_time = datetime.now() + timedelta(seconds=sleep_sec)
        logger.info(f"⏳ Sleeping {int(sleep_sec)}s until next quarter boundary ({next_run_time.strftime('%H:%M:%S')})...\n")

        try:
            time.sleep(sleep_sec)
        except KeyboardInterrupt:
            logger.info("\n🛑 Publisher stopped by user.")
            break


if __name__ == "__main__":
    main()

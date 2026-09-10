#!/usr/bin/env python3
# ==============================================================================
# scripts/auto_publisher.py
# 24/7 Continuous Background Publisher for Power Trading Bidder Portal
#
# Runs every 15 minutes automatically:
# - Fetches latest 15-minute market records & forecasts
# - Evaluates TFT trading strategy with gate closure rules
# - Upserts DK1 & DK2 quarter predictions to Supabase
# - Bidder's portal on Vercel updates automatically in real-time
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
    logger.info("🚀 24/7 POWER TRADING BIDDER PUBLISHER SERVICE STARTED")
    logger.info(f"   Project Root: {PROJECT_ROOT}")
    logger.info(f"   Log File:     {log_file}")
    logger.info("=" * 70)

    # Initial immediate run on startup
    iteration = 1
    while True:
        cycle_start = datetime.now()
        logger.info(f"\n[Cycle #{iteration}] Starting prediction update at {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}...")

        try:
            # Push predictions for DK1 and DK2
            push_predictions(delivery_date=None, price_areas=["DK1", "DK2"])
            logger.info(f"[Cycle #{iteration}] Update successful! Supabase is up to date.")
        except KeyboardInterrupt:
            logger.info("\n🛑 Publisher stopped by user.")
            break
        except Exception as e:
            logger.error(f"[Cycle #{iteration}] ❌ Error during prediction push: {e}", exc_info=True)
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

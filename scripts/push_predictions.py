#!/usr/bin/env python3
# ==============================================================================
# scripts/push_predictions.py
# Standalone script to push TFT model predictions to Supabase for Bidder Portal.
#
# Usage:
#   python scripts/push_predictions.py                  # Push today's predictions
#   python scripts/push_predictions.py --date 2026-09-11  # Push specific date
#
# Can be run manually or via cron every 15 minutes:
#   */15 * * * * cd /path/to/power-trading && python scripts/push_predictions.py
# ==============================================================================

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.tournament_tables_v2 import TournamentTableGenerator
from src.commercial_strategy_v3 import V3CommercialStrategyEngine
from src.supabase_publisher import SupabasePublisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def push_predictions(delivery_date: str = None, price_areas=("DK1", "DK2")):
    """
    Generate TFT predictions for the given delivery date and push to Supabase.

    1. Fetches the 96-quarter table from Energi Data Service
    2. Runs V3CommercialStrategyEngine to get predictions + decisions
    3. Upserts to Supabase quarter_predictions table
    """
    if delivery_date is None:
        # Default: tomorrow (predictions are for next delivery day)
        delivery_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    publisher = SupabasePublisher()
    model_name = "Transformer-TFT"

    total_pushed = 0

    for area in price_areas:
        logger.info(f"{'='*60}")
        logger.info(f"Processing {area} for delivery date {delivery_date}")
        logger.info(f"{'='*60}")

        try:
            # Step 1: Get 96-quarter table from Energi Data Service
            table_gen = TournamentTableGenerator(price_area=area)

            # Use get_future_table() for today/tomorrow, backtest for historical
            today_str = datetime.now().strftime("%Y-%m-%d")
            tomorrow_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

            if delivery_date in (today_str, tomorrow_str):
                logger.info(f"  Fetching live future table for {area}...")
                df = table_gen.get_future_table()
            else:
                logger.info(f"  Fetching backtest table for {area} ({delivery_date})...")
                df = table_gen.get_backtest_table(date_str=delivery_date)

            if df is None or df.empty:
                logger.warning(f"  ⚠️  No data available for {area} on {delivery_date}")
                continue

            logger.info(f"  Got {len(df)} quarters from data source")

            # Step 2: Run V3 Strategy Engine
            strategy = V3CommercialStrategyEngine(
                price_area=area, capital=100000.0, base_volume_mwh=2.0
            )
            summary = strategy.evaluate_trading_ledger(
                df, model_name=model_name, market_mode="INTRADAY_D0"
            )

            trades = summary.get("trades", [])
            if not trades:
                logger.warning(f"  ⚠️  No trades generated for {area}")
                continue

            logger.info(f"  Generated {len(trades)} trade signals")

            # Step 3: Push to Supabase
            count = publisher.upsert_predictions(
                trades=trades,
                price_area=area,
                delivery_date=delivery_date,
                model_name=model_name,
            )
            total_pushed += count

        except Exception as e:
            logger.error(f"  ❌ Failed for {area}: {e}", exc_info=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Done! Pushed {total_pushed} total quarter predictions to Supabase")
    logger.info(f"{'='*60}")
    return total_pushed


def main():
    parser = argparse.ArgumentParser(
        description="Push TFT model predictions to Supabase for Bidder Portal"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Delivery date in YYYY-MM-DD format (default: tomorrow)",
    )
    parser.add_argument(
        "--areas",
        type=str,
        nargs="+",
        default=["DK1", "DK2"],
        help="Price areas to process (default: DK1 DK2)",
    )
    args = parser.parse_args()
    push_predictions(delivery_date=args.date, price_areas=args.areas)


if __name__ == "__main__":
    main()

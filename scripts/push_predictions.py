#!/usr/bin/env python3
# ==============================================================================
# scripts/push_predictions.py
# Standalone script to push multi-model predictions & trade audit ledger to Supabase.
#
# Usage:
#   python scripts/push_predictions.py                  # Push today & tomorrow for all 6 models
#   python scripts/push_predictions.py --date 2026-09-12  # Push specific date
#   python scripts/push_predictions.py --models Transformer-TFT  # Push specific model
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
from src.supabase_publisher import SupabasePublisher, ALL_MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def push_predictions(
    delivery_date: str = None,
    price_areas=("DK1", "DK2"),
    models=None,
    market_modes=("INTRADAY_D0", "DAY_AHEAD_D1"),
    reconcile: bool = True,
):
    """
    Generate predictions for all requested models and market modes,
    and push to Supabase. Also reconciles occurred settlement quarters.
    """
    if models is None:
        models = ALL_MODELS

    today_str = datetime.now().strftime("%Y-%m-%d")
    tomorrow_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    dates_to_push = [delivery_date] if delivery_date else [today_str, tomorrow_str]

    publisher = SupabasePublisher()
    total_pushed = 0

    for current_date in dates_to_push:
        for area in price_areas:
            logger.info(f"{'='*60}")
            logger.info(f"Processing {area} for delivery date {current_date}")
            logger.info(f"{'='*60}")

            try:
                # Step 1: Get 96-quarter table from Energi Data Service
                table_gen = TournamentTableGenerator(price_area=area)

                if current_date in (today_str, tomorrow_str):
                    logger.info(f"  Fetching live future table for {area} ({current_date})...")
                    df = table_gen.get_future_table(date_str=current_date)
                else:
                    logger.info(f"  Fetching backtest table for {area} ({current_date})...")
                    df = table_gen.get_backtest_table(date_str=current_date)

                if df is None or df.empty:
                    logger.warning(f"  ⚠️ No data available for {area} on {current_date}")
                    continue

                logger.info(f"  Got {len(df)} quarters from data source")

                strategy = V3CommercialStrategyEngine(
                    price_area=area, capital=20000.0, base_volume_mwh=2.0
                )

                # Step 2: Loop through requested models and market modes
                for mode in market_modes:
                    for model_name in models:
                        logger.info(f"  Evaluating [{mode}] model={model_name}...")
                        summary = strategy.evaluate_trading_ledger(
                            df, model_name=model_name, market_mode=mode
                        )

                        trades = summary.get("trades", [])
                        if not trades:
                            continue

                        # Step 3: Push to Supabase
                        count = publisher.upsert_predictions(
                            trades=trades,
                            price_area=area,
                            delivery_date=current_date,
                            model_name=model_name,
                            market_mode=mode,
                        )
                        total_pushed += count

                # Step 4: Reconcile past quarters against authentic Energinet Imbalance settlement
                if reconcile and current_date == today_str:
                    logger.info(f"  Auditing settled quarters for {area} ({today_str})...")
                    publisher.reconcile_settled_quarters(price_area=area, delivery_date=today_str)

            except Exception as e:
                logger.error(f"  ❌ Failed for {area} ({current_date}): {e}", exc_info=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Done! Pushed {total_pushed} total quarter records to Supabase")
    logger.info(f"{'='*60}")
    return total_pushed


def main():
    parser = argparse.ArgumentParser(
        description="Push multi-model predictions & ledger to Supabase for Bidder Portal"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Delivery date in YYYY-MM-DD format (default: today and tomorrow)",
    )
    parser.add_argument(
        "--areas",
        type=str,
        nargs="+",
        default=["DK1", "DK2"],
        help="Price areas to process (default: DK1 DK2)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to process (default: all 6 tournament models)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["INTRADAY_D0", "DAY_AHEAD_D1"],
        help="Market modes to process (default: INTRADAY_D0 DAY_AHEAD_D1)",
    )
    parser.add_argument(
        "--no-reconcile",
        action="store_true",
        help="Skip settlement reconciliation against Energinet",
    )
    args = parser.parse_args()

    push_predictions(
        delivery_date=args.date,
        price_areas=args.areas,
        models=args.models,
        market_modes=args.modes,
        reconcile=not args.no_reconcile,
    )


if __name__ == "__main__":
    main()

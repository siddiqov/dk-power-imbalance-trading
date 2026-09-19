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
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.tournament_tables_v2 import TournamentTableGenerator
from src.commercial_strategy_v3_1 import V31CommercialStrategyEngine
from src.supabase_publisher import SupabasePublisher, ALL_MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _adapter_df_to_trades(df: pd.DataFrame) -> list:
    """Convert a v4_1_intraday dashboard_adapter day_decisions() DataFrame to
    the trade-dict format expected by SupabasePublisher.upsert_predictions().

    Source column mapping:
        'LOCKED'      -> status='LOCKED'          (journal committed at gate closure)
        'MISSED'      -> status='MISSED'           (gate passed before journal ran)
        'recomputed'  -> status='LOCKED_PENDING'   (computed now, not yet in journal)
    """
    if df is None or df.empty:
        return []

    STATUS_MAP = {
        "LOCKED": "LOCKED",
        "MISSED": "MISSED",
        "recomputed": "LOCKED_PENDING",
        "what-if": "LOCKED_PENDING",
        "risk-managed": "LOCKED_PENDING",
    }

    trades = []
    for _, row in df.iterrows():
        # Derive quarter index from time_dk_str (e.g. '2026-09-19 06:00')
        try:
            t = pd.to_datetime(row.get("time_dk_str") or row.get("quarter_utc"))
            q_idx = t.hour * 4 + t.minute // 15 + 1
        except Exception:
            q_idx = 1

        action = str(row.get("action", "HOLD"))
        if action == "BUY":
            decision = "BUY Spot (Long)"
            direction = 1
        elif action == "SELL":
            decision = "SELL Spot (Short)"
            direction = -1
        else:
            decision = "HOLD"
            direction = 0

        src = str(row.get("source", "recomputed"))
        status = STATUS_MAP.get(src, "LOCKED_PENDING")
        if src == "MISSED":
            decision = "HOLD"
            direction = 0

        mwh = float(row.get("mwh", 0.0) or 0.0)
        exp_spread = float(row.get("exp_spread", 0.0) or 0.0)
        spot = float(row.get("spot_actual", 0.0) or 0.0)
        pred_imb = round(spot + exp_spread, 2)

        p_up = float(row.get("p_up", 0.0) or 0.0)
        p_flat = float(row.get("p_flat", 0.0) or 0.0)
        p_down = float(row.get("p_down", 0.0) or 0.0)
        q10 = float(row.get("q10", 0.0) or 0.0)
        q50 = float(row.get("q50", 0.0) or 0.0)
        q90 = float(row.get("q90", 0.0) or 0.0)

        pnl = row.get("pnl_eur")
        net_pnl = float(pnl) if pnl is not None and not (isinstance(pnl, float) and np.isnan(pnl)) else None

        trades.append({
            "quarter": f"Q{q_idx}",
            "time_dk": str(row.get("time_dk_str", "")),
            "spot_price_eur": spot,
            "pred_imbalance_eur": pred_imb,
            "pred_spread_eur": exp_spread,
            "p_up": p_up,
            "p_flat": p_flat,
            "p_down": p_down,
            "p_up_spike": 0.0,
            "action": decision,
            "direction": direction,
            "volume_mwh": mwh,
            "q10_price_eur": q10,
            "q50_price_eur": q50,
            "q90_price_eur": q90,
            "status": status,
            "net_pnl_eur": net_pnl,
        })
    return trades


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

                strategy = V31CommercialStrategyEngine(
                    price_area=area, capital=20000.0, base_volume_mwh=2.0, profile="tier3_aggressive"
                )

                # Approach A (Midnight Boundary Bridge): Load Day D table to seed physical inertia across midnight
                try:
                    t_col = "time_dk" if "time_dk" in df.columns else "time_utc"
                    first_dt = pd.to_datetime(df.iloc[0][t_col])
                    prev_date_str = (first_dt - timedelta(days=1)).strftime("%Y-%m-%d")
                    if prev_date_str == today_str:
                        df_prev_day = table_gen.get_future_table(date_str=today_str)
                    else:
                        df_prev_day = table_gen.get_backtest_table(date_str=prev_date_str)
                except Exception:
                    df_prev_day = None

                # Step 2: Loop through requested models and market modes
                for mode in market_modes:
                    for model_name in models:
                        if model_name == "V4.1-HighAlpha":
                            # Use the v4_1_intraday adapter (matches dashboard_v4_1.py exactly).
                            # Aggressive threshold = validated margin * 0.25, pmin reduced by 0.06.
                            # Locked decisions from the journal override recomputed ones automatically.
                            try:
                                from v4_1_intraday import dashboard_adapter as v41id
                                _params = (v41id.decision_params(area, "aggressive")
                                           if hasattr(v41id, "decision_params") else None)
                                df_decisions = v41id.day_decisions(area, current_date, params=_params)
                                trades = _adapter_df_to_trades(df_decisions)
                                logger.info(
                                    f"  [V4.1-HighAlpha] adapter returned {len(df_decisions)} quarters "
                                    f"({sum(1 for t in trades if t['status'] == 'LOCKED')} LOCKED, "
                                    f"{sum(1 for t in trades if t['status'] == 'MISSED')} MISSED) "
                                    f"for {area} {current_date}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"  [V4.1-HighAlpha] adapter failed ({e}), falling back to V41InferenceEngine"
                                )
                                from src.v4_1_inference_engine import V41InferenceEngine
                                v41_engine = V41InferenceEngine(price_area=area)
                                trades = v41_engine.evaluate_96q_schedule(df, current_date)
                        elif model_name == "V3.2-FlowAware":
                            from src.v3_2_inference_engine import V32InferenceEngine
                            v32_engine = V32InferenceEngine(price_area=area)
                            trades = v32_engine.evaluate_96q_schedule(df, current_date)
                        else:
                            summary = strategy.evaluate_trading_ledger(
                                df, 
                                model_name=model_name, 
                                market_mode=mode, 
                                profile="tier3_aggressive",
                                approach="A", 
                                df_prev_day=df_prev_day
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

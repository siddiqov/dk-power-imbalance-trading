# ==============================================================================
# src/realtime_tracker_v2.py
# Real-Time Day-Ahead Arbitrage Planner & Live Intraday ROC Tracker
#
# Functions:
# 1. generate_day_ahead_trading_plan(target_date, capital):
#    Pulls tomorrow's Day-Ahead Spot prices (cleared at 12:45 D-1) and forecasts,
#    predicts all 96 quarters of imbalance prices, and locks in trade commitments.
# 2. evaluate_live_intraday_settlement(target_date, capital):
#    Fetches real-time settled quarters from Energinet, computes quarter-by-quarter
#    Gross PnL, fees, and Live Real-Time ROC (%) on the portfolio.
# ==============================================================================

import os
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.data_ingestion_v2 import V2DataEngine
from src.feature_engineering_v2 import V2FeatureEngineer
from src.trading_agent_v2 import CommercialTradingAgent


class RealTimeDayAheadTracker:
    """
    Manages live Day-Ahead bidding schedules and intraday real-time ROC tracking.
    """

    def __init__(self, price_area='DK1', initial_capital=100000.0):
        self.price_area = price_area
        self.initial_capital = initial_capital
        self.engine = V2DataEngine()
        self.fe = V2FeatureEngineer()
        self.agent = CommercialTradingAgent(initial_capital=initial_capital)

    def generate_day_ahead_plan(self, target_date=None, best_model_res=None):
        """
        Generates the 96-quarter trade commitment schedule for tomorrow (e.g. Sept 2nd).
        """
        if target_date is None:
            target_date = datetime.now().date() + timedelta(days=1)

        print(f"\n=======================================================")
        print(f"  GENERATING DAY-AHEAD TRADING SCHEDULE FOR: {target_date} ({self.price_area})")
        print(f"=======================================================")

        # Pull spot prices for target date
        df_spot = self.engine.fetch_api_dataset(
            "DayAheadPrices",
            start_date=datetime.combine(target_date, datetime.min.time()),
            end_date=datetime.combine(target_date, datetime.max.time()),
            limit=500
        )

        if df_spot.empty:
            # Fallback to Elspotprices
            df_spot = self.engine.fetch_api_dataset(
                "Elspotprices",
                start_date=datetime.combine(target_date, datetime.min.time()),
                end_date=datetime.combine(target_date, datetime.max.time()),
                limit=500
            )

        if not df_spot.empty:
            df_spot = df_spot[df_spot["PriceArea"] == self.price_area].copy()
            df_spot["time_utc"] = pd.to_datetime(df_spot["HourUTC" if "HourUTC" in df_spot.columns else "TimeUTC"])
            df_spot.rename(columns={"SpotPriceEUR": "spot_price_eur"}, inplace=True)
            df_spot.sort_values("time_utc", inplace=True)

        return df_spot

    def calculate_live_roc_status(self, trade_schedule_df, actual_settled_df):
        """
        Calculates real-time live ROC (%) as quarters settle throughout the day.
        """
        merged = pd.merge(trade_schedule_df, actual_settled_df, on="time_utc", how="left", suffixes=("", "_actual"))
        settled_trades = merged.dropna(subset=["imbalance_price_eur_actual"])

        capital = self.initial_capital
        gross_pnl = 0.0
        fees = 0.0
        slippage = 0.0

        for _, row in settled_trades.iterrows():
            action = row.get("action", "HOLD")
            vol = row.get("volume_mwh", 0.0)
            p_spot = row["spot_price_eur"]
            p_imb = row["imbalance_price_eur_actual"]
            spread = p_imb - p_spot

            if action == "LONG_SPOT":
                trade_gross = vol * spread
            elif action == "SHORT_SPOT":
                trade_gross = vol * (-spread)
            else:
                trade_gross = 0.0

            trade_fee = vol * (self.agent.exchange_fee + self.agent.tso_fee)
            trade_slip = vol * self.agent.slippage

            gross_pnl += trade_gross
            fees += trade_fee
            slippage += trade_slip

        operating_pnl = gross_pnl - fees - slippage
        tax = max(operating_pnl * self.agent.tax_rate, 0.0)
        net_profit = operating_pnl - tax
        current_capital = capital + net_profit
        live_roc = (net_profit / capital) * 100.0

        return {
            "quarters_settled": len(settled_trades),
            "total_quarters": len(trade_schedule_df),
            "current_capital": current_capital,
            "net_profit": net_profit,
            "live_roc_pct": live_roc,
            "gross_pnl": gross_pnl,
            "fees_paid": fees + slippage
        }

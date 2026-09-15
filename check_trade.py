import sys
import os
import pandas as pd
import json
sys.path.append(os.path.abspath('.'))

from src.tournament_tables_v2 import TournamentTableGenerator
from src.commercial_strategy_v3_1 import V31CommercialStrategyEngine

area = "DK1"
table_gen = TournamentTableGenerator(price_area=area)
date_str = "2026-09-14"

# Use the exact logic from dashboard_v3_1.py
target_df = pd.DataFrame()
try:
    target_df = table_gen.generate_and_save_future_table(date_str=date_str)
except Exception as e:
    target_df = table_gen.get_backtest_table(date_str=date_str)

if target_df.empty:
    target_df = table_gen.get_future_table(date_str=date_str)

print("Target DF shape:", target_df.shape)
if not target_df.empty:
    strategy = V31CommercialStrategyEngine(price_area=area)
    s_v31 = strategy.evaluate_trading_ledger(target_df, model_name="Transfer-BiLSTM", market_mode="INTRADAY_D0")
    trades = s_v31.get("trades", [])
    if len(trades) > 0:
        print("KEYS:", trades[0].keys())
        print(trades[0])

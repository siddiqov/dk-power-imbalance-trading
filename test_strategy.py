import sys
import os
import pandas as pd
sys.path.append(os.path.abspath('.'))

from src.tournament_tables_v2 import TournamentTableGenerator
from src.commercial_strategy_v3_1 import V31CommercialStrategyEngine

area = "DK1"
table_gen = TournamentTableGenerator(price_area=area)
date_str = "2026-09-14"

# Generate target DataFrame for today
try:
    target_df = table_gen.generate_and_save_future_table(date_str=date_str)
except Exception as e:
    print("Future table generation failed:", e)
    target_df = table_gen.get_backtest_table(date_str=date_str)

print("Target DF length:", len(target_df))

if not target_df.empty:
    strategy = V31CommercialStrategyEngine(price_area=area)
    # Evaluate ledger
    s_v31 = strategy.evaluate_trading_ledger(target_df, model_name="Transfer-BiLSTM", market_mode="INTRADAY_D0")
    
    trades = s_v31.get("trades", [])
    print(f"Generated {len(trades)} trades.")
    if len(trades) > 0:
        print("First trade sample:")
        t = trades[0]
        print(f"Time: {t.get('time_str')}, P(Up): {t.get('p_up_pct')}%, V3.1 Decision: {t.get('action')}")

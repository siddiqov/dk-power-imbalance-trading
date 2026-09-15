import sys
import os
import pandas as pd
sys.path.append(os.path.abspath('.'))

from src.tournament_tables_v2 import TournamentTableGenerator

table_gen = TournamentTableGenerator(price_area="DK1")
date_str = pd.Timestamp.now(tz="Europe/Copenhagen").strftime("%Y-%m-%d")
try:
    df = table_gen.generate_and_save_future_table(date_str=date_str)
    print(df.columns)
except Exception as e:
    df = table_gen.get_backtest_table(date_str=date_str)
    print(df.columns)

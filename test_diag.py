import sys, os, traceback
sys.path.insert(0, r'C:\Users\Hafeez\Documents\Nurex_Trading\Basic_Approach')

print("=== Step 1: Import TournamentTableGenerator ===")
try:
    from src.tournament_tables_v2 import TournamentTableGenerator, _RESULTS_DIR
    print(f"[OK] Imported. _RESULTS_DIR={_RESULTS_DIR}")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 2: get_backtest_table('2026-09-22') ===")
try:
    tg = TournamentTableGenerator(price_area="DK1")
    df = tg.get_backtest_table(date_str="2026-09-22")
    print(f"[OK] Returned {len(df)} rows, empty={df.empty}")
    if not df.empty:
        print(f"  Columns: {list(df.columns)}")
        print(f"  First row: {df.iloc[0].to_dict()}")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 3: generate_and_save_future_table('2026-09-22') ===")
try:
    df2 = tg.generate_and_save_future_table(date_str="2026-09-22")
    print(f"[OK] Returned {len(df2)} rows, empty={df2.empty}")
except Exception as e:
    print(f"[WARN] generate_and_save_future_table raised: {e}")
    print("  (Will fall back to get_backtest_table)")

print("\n=== Step 4: V31CommercialStrategyEngine.evaluate_trading_ledger ===")
try:
    from src.commercial_strategy_v3_1 import V31CommercialStrategyEngine
    strat = V31CommercialStrategyEngine(price_area="DK1")
    res = strat.evaluate_trading_ledger(df, model_name="Transfer-BiLSTM", market_mode="INTRADAY_D0")
    trades = res.get("trades", [])
    print(f"[OK] evaluate_trading_ledger returned {len(trades)} trades")
    if trades:
        print(f"  First trade keys: {list(trades[0].keys()) if isinstance(trades[0], dict) else type(trades[0])}")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)

import pandas as pd
df_trades = pd.DataFrame(trades)
print(f"\n=== Step 5: df_trades ===")
print(f"  empty={df_trades.empty}, shape={df_trades.shape}")
if not df_trades.empty:
    print(f"  Columns: {list(df_trades.columns)}")

print("\n=== Step 6: V41FeatureEngine.build_feature_matrix ===")
try:
    from src.feature_engine_v4_1 import V41FeatureEngine
    fe = V41FeatureEngine(price_area="DK1")
    feat_df = fe.build_feature_matrix(df_trades)
    print(f"[OK] build_feature_matrix returned shape={feat_df.shape}, empty={feat_df.empty}")
except Exception as e:
    traceback.print_exc()

print("\n=== DONE ===")

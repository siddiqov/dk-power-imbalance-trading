# train_v3.py
import sys, os
sys.path.append(os.path.abspath('.'))

import pandas as pd
from src.data_ingestion_v2 import V2DataEngine
from src.model_trainer_v3 import V3QuantileModelSuite

def main():
    engine = V2DataEngine()
    
    for area in ["DK1", "DK2"]:
        print(f"\n=======================================================")
        print(f"  TRAINING V3 QUANTILE & DESK SUITE FOR {area}")
        print(f"=======================================================")
        df_1h, df_15m = engine.load_paradigm3_dual_models(area)
        
        # Combine or prepare training matrix
        df_train = df_15m if not df_15m.empty else df_1h
        if df_train.empty:
            print(f"[WARN] No local training data found for {area}, loading historical table...")
            df_train = pd.read_csv(f"results/96Q_backtest_table_{area}.csv")
            
        suite = V3QuantileModelSuite(price_area=area)
        suite.train_models(df_train)
        print(f"[SUCCESS] V3 Models for {area} trained and saved successfully!")

if __name__ == '__main__':
    main()

import sys
import os
sys.path.append(os.path.abspath('.'))
from src.data_ingestion_v2 import V2DataEngine

engine = V2DataEngine()
df_1h, df_15m = engine.load_paradigm3_dual_models("DK1")
print(df_15m.columns.tolist())

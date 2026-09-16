import sys, os
sys.path.append(os.path.abspath('.'))
from train_v3_2_flow_aware import fetch_real_v3_1_baseline_scores
df = fetch_real_v3_1_baseline_scores('DK1')
print(df.columns.tolist())
import joblib
import sys
import os
sys.path.append(os.path.abspath('.'))

data = joblib.load('models_v3_1/v3_1_suite_DK1.pkl')
print(data['intraday']['_feature_cols'])

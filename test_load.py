import joblib
import sys
import os

sys.path.append(os.path.abspath('.'))

try:
    data = joblib.load('models_v3_1/v3_1_suite_DK1.pkl')
    print("Keys in loaded suite:", data.keys() if isinstance(data, dict) else type(data))
    if isinstance(data, dict):
        if 'intraday' in data:
            print("Intraday keys:", data['intraday'].keys() if isinstance(data['intraday'], dict) else type(data['intraday']))
except Exception as e:
    print("Error:", e)

import os
import pandas as pd
from entsoe import EntsoePandasClient
import traceback

ENTSOE_TOKEN = os.environ.get("ENTSOE_TOKEN", "")
client = EntsoePandasClient(api_key=ENTSOE_TOKEN)

start_ts = pd.Timestamp('2026-09-14', tz='Europe/Copenhagen')
end_ts = start_ts + pd.Timedelta(days=1)

print("Querying Scheduled Exchanges (DE -> DK1)...")
try:
    scheduled = client.query_scheduled_exchanges('DE_LU', 'DK_1', start=start_ts, end=end_ts, day_ahead=True)
    print(scheduled.head(10))
except Exception as e:
    print("Error:", e)
    traceback.print_exc()

print("\nQuerying Physical Flows (DE -> DK1)...")
try:
    physical = client.query_crossborder_flows('DE_LU', 'DK_1', start=start_ts, end=end_ts)
    print(physical.head(10))
except Exception as e:
    print("Error:", e)

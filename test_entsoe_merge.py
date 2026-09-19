import os
import pandas as pd
from entsoe import EntsoePandasClient

ENTSOE_TOKEN = os.environ.get("ENTSOE_TOKEN", "")
client = EntsoePandasClient(api_key=ENTSOE_TOKEN)

date_str = '2026-09-14'
start_ts = pd.Timestamp(date_str, tz='Europe/Copenhagen')
end_ts = start_ts + pd.Timedelta(days=1)

de_prices = client.query_day_ahead_prices('DE_LU', start=start_ts, end=end_ts)
de_prices_df = de_prices.reset_index()
de_prices_df.columns = ['time_dk_obj', 'de_spot_eur']
de_prices_df['time_dk_obj'] = de_prices_df['time_dk_obj'].dt.tz_localize(None)

print(de_prices_df.head())

flows = client.query_crossborder_flows('DE_LU', 'DK_1', start=start_ts, end=end_ts)
flows_df = flows.reset_index()
flows_df.columns = ['time_dk_obj', 'physical_flow_mw']
flows_df['time_dk_obj'] = flows_df['time_dk_obj'].dt.tz_localize(None)

print(flows_df.head())

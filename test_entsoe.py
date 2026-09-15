from entsoe import EntsoePandasClient
import pandas as pd

client = EntsoePandasClient(api_key="01bb4846-6f4c-4e0f-8333-6c709b316594")

start = pd.Timestamp.now(tz='Europe/Copenhagen').floor('D')
end = start + pd.Timedelta(days=1)

try:
    print("Testing DE Prices...")
    de_prices = client.query_day_ahead_prices('DE_LU', start=start, end=end)
    print(de_prices.head())
    
    print("Testing DK1-DE Cross border physical flows...")
    flows = client.query_crossborder_flows('DE_LU', 'DK_1', start=start, end=end)
    print(flows.head())
except Exception as e:
    print("Error:", e)

import requests
import json
import time
import pandas as pd
import duckdb

print('Fetching ElectricityProdex5MinRealtime for DK1 (2026)...')
url = 'https://api.energidataservice.dk/dataset/ElectricityProdex5MinRealtime'
all_records = []

# Fetch in monthly batches
dates = pd.date_range('2026-01-01', '2026-08-30', freq='MS')
for d in dates:
    w_start = d.strftime('%Y-%m-%dT00:00')
    next_month = d + pd.DateOffset(months=1)
    w_end = next_month.strftime('%Y-%m-%dT00:00')
    params = {
        'filter': json.dumps({'PriceArea': 'DK1'}),
        'start': w_start,
        'end': w_end,
        'limit': 20000
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            recs = r.json().get('records', [])
            print(f'  {w_start[:7]}: {len(recs):,} records')
            all_records.extend(recs)
        else:
            print(f'  {w_start[:7]}: status {r.status_code}')
    except Exception as e:
        print(f'  {w_start[:7]}: error {e}')
    time.sleep(0.3)

if all_records:
    df = pd.DataFrame(all_records)
    df['time_utc'] = pd.to_datetime(df['Minutes5UTC'])
    df['time_dk'] = pd.to_datetime(df['Minutes5DK'])
    df['price_area'] = df['PriceArea']
    df['wind_offshore'] = pd.to_numeric(df['OffshoreWindPower'], errors='coerce').fillna(0)
    df['wind_onshore'] = pd.to_numeric(df['OnshoreWindPower'], errors='coerce').fillna(0)
    df['total_wind'] = df['wind_offshore'] + df['wind_onshore']
    df['solar'] = pd.to_numeric(df['SolarPower'], errors='coerce').fillna(0)
    
    ex_de = pd.to_numeric(df.get('ExchangeGermany', 0), errors='coerce').fillna(0)
    ex_nl = pd.to_numeric(df.get('ExchangeNetherlands', 0), errors='coerce').fillna(0)
    ex_gb = pd.to_numeric(df.get('ExchangeGreatBritain', 0), errors='coerce').fillna(0)
    ex_no = pd.to_numeric(df.get('ExchangeNorway', 0), errors='coerce').fillna(0)
    ex_se = pd.to_numeric(df.get('ExchangeSweden', 0), errors='coerce').fillna(0)
    ex_belt = pd.to_numeric(df.get('ExchangeGreatBelt', 0), errors='coerce').fillna(0)
    
    df['exchange_continent'] = ex_de + ex_nl
    df['exchange_great_belt'] = ex_belt
    df['exchange_nordic'] = ex_no + ex_se
    df['exchange_gb'] = ex_gb
    df['net_exchange'] = ex_de + ex_nl + ex_gb + ex_no + ex_se + ex_belt
    
    prod_lt100 = pd.to_numeric(df.get('ProductionLt100MW', 0), errors='coerce').fillna(0)
    prod_ge100 = pd.to_numeric(df.get('ProductionGe100MW', 0), errors='coerce').fillna(0)
    df['total_load'] = prod_lt100 + prod_ge100 + df['total_wind'] + df['solar'] - df['net_exchange']
    
    # Resample 5-min to 15-min
    numeric_cols = ['total_load', 'total_wind', 'wind_offshore', 'wind_onshore', 'solar',
                    'exchange_continent', 'exchange_great_belt', 'exchange_nordic',
                    'exchange_gb', 'net_exchange']
    df_15m = df.set_index('time_utc')[numeric_cols].resample('15min').mean().reset_index()
    df_15m['time_dk'] = df_15m['time_utc'].dt.tz_localize('UTC').dt.tz_convert('Europe/Copenhagen').dt.tz_localize(None)
    df_15m['price_area'] = 'DK1'
    
    cols = ['id', 'time_utc', 'time_dk', 'price_area', 'total_load', 'total_wind', 'wind_offshore',
            'wind_onshore', 'solar', 'exchange_continent', 'exchange_great_belt', 'exchange_nordic',
            'exchange_gb', 'net_exchange']
    
    df_15m['id'] = df_15m.apply(lambda x: int(hash(f"{x['time_utc']}{x['price_area']}") % 10**15), axis=1)
    df_insert = df_15m[cols].dropna(subset=['time_utc'])
    
    conn = duckdb.connect('energy_data.db')
    conn.register('df_insert', df_insert)
    conn.execute('INSERT OR REPLACE INTO electricity_balance (id, time_utc, time_dk, price_area, total_load, total_wind, wind_offshore, wind_onshore, solar, exchange_continent, exchange_great_belt, exchange_nordic, exchange_gb, net_exchange) SELECT * FROM df_insert')
    count = conn.execute("SELECT COUNT(*) as n, MIN(time_utc) as mn, MAX(time_utc) as mx FROM electricity_balance WHERE price_area='DK1'").fetchdf()
    print('Updated electricity_balance:', count.to_dict('records'))
    conn.close()

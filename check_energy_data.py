import pandas as pd
import sqlite3

conn = sqlite3.connect('energy_data.db')
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables in energy_data.db:", tables['name'].tolist())

df = pd.read_sql_query("SELECT COUNT(*) as cnt FROM intraday_15m_dk", conn)
print("Rows in intraday_15m_dk:", df['cnt'].iloc[0])

# Load a sample to do the backtest
df_sample = pd.read_sql_query("SELECT * FROM intraday_15m_dk ORDER BY time_utc DESC LIMIT 10", conn)
print(df_sample.columns.tolist())

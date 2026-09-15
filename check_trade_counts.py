import pandas as pd
import sqlite3

conn = sqlite3.connect('trades_journal.db')
df = pd.read_sql_query("SELECT COUNT(*) as cnt FROM trade_orders WHERE status = 'SETTLED'", conn)
print("Settled Trades in trades_journal:", df['cnt'].iloc[0])

df_sample = pd.read_sql_query("SELECT * FROM trade_orders WHERE status = 'SETTLED' LIMIT 5", conn)
print(df_sample[['delivery_date', 'locked_spot_price_eur', 'actual_settled_price_eur', 'net_pnl_eur']])

# Check trading_history.db
conn2 = sqlite3.connect('trading_history.db')
df2 = pd.read_sql_query("SELECT COUNT(*) as cnt FROM trade_ledger", conn2)
print("Trades in trading_history.db:", df2['cnt'].iloc[0])

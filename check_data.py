import duckdb
conn = duckdb.connect('energy_data.db')
print("Sample Forecasts:")
print(conn.execute('SELECT * FROM forecasts_hour LIMIT 2').fetchdf())
print("\nSample Balance:")
print(conn.execute('SELECT * FROM electricity_balance LIMIT 2').fetchdf())
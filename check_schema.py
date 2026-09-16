import duckdb
conn = duckdb.connect('energy_data.db')
print("Forecasts_hour:")
print(conn.execute('DESCRIBE forecasts_hour;').fetchall())
print("\nElectricity_balance:")
print(conn.execute('DESCRIBE electricity_balance;').fetchall())
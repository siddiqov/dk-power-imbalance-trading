import duckdb
conn = duckdb.connect('energy_data.db')
print(conn.execute('SHOW TABLES;').fetchall())
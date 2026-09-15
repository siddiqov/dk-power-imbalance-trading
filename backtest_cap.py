import pandas as pd
import numpy as np

# Load the historical dataset
df = pd.read_csv('imbalance_price_processed.csv')

# Drop NaNs in target columns
df = df.dropna(subset=['spot_price_eur', 'imbalance_price_eur'])

print(f"Total historical quarters analyzed: {len(df):,}")

# Calculate the actual spread (profit/loss per MWh if you were to BUY at Spot)
df['long_pnl'] = df['imbalance_price_eur'] - df['spot_price_eur']

# Create bins for Spot Price
bins = [0, 50, 100, 150, 200, 250, 300, 400, 1000]
labels = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-400', '400+']
df['spot_bracket'] = pd.cut(df['spot_price_eur'], bins=bins, labels=labels)

# Calculate stats per bracket
results = df.groupby('spot_bracket')['long_pnl'].agg(['count', 'mean', 'min']).reset_index()
results.columns = ['Spot Bracket', 'Number of Occurrences', 'Average Net PnL (Long)', 'Worst Single Loss']

print("\n--- HISTORICAL LONG PNL EXPECTED VALUE BY SPOT PRICE BRACKET ---")
print(results.to_string(index=False))

# Calculate optimal price cap (Grid Search simulation)
# If a model randomly guessed, what is the PnL if we apply a cap?
caps = [150, 200, 250, 300, 500]
print("\n--- GRID SEARCH: TOTAL PNL (assuming naive long) BY PRICE CAP ---")
for cap in caps:
    filtered_df = df[df['spot_price_eur'] <= cap]
    total_pnl = filtered_df['long_pnl'].sum()
    print(f"Cap: EUR {cap:<3} -> Total Historical PnL: EUR {total_pnl:,.2f}")


import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trading_history.db')

def init_db():
    """Initializes the SQLite database and the trade_ledger table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create the ledger table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trade_ledger (
            date TEXT,
            price_area TEXT,
            quarter INTEGER,
            time_dk TEXT,
            spot_price_eur REAL,
            model_prediction_eur REAL,
            actual_settled_eur REAL,
            action TEXT,
            volume_mwh REAL,
            fees_eur REAL,
            net_pnl_eur REAL,
            model_used TEXT,
            PRIMARY KEY (date, price_area, quarter, model_used)
        )
    ''')
    conn.commit()
    conn.close()

def save_daily_ledger(df, date_str, area, model_used="Stacking-MetaEnsemble"):
    """
    Saves a Pandas DataFrame of the daily ledger into the database.
    UPSERTS data to ensure we don't duplicate rows for the same date/area.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    
    # Clean up any existing records for this date, area, and model to do a clean overwrite
    cursor = conn.cursor()
    cursor.execute('''
        DELETE FROM trade_ledger 
        WHERE date = ? AND price_area = ? AND model_used = ?
    ''', (date_str, area, model_used))
    
    # Append the dataframe
    db_df = pd.DataFrame({
        'date': date_str,
        'price_area': area,
        'quarter': df['quarter'].astype(str).str.extract(r'(\d+)')[0].astype(int),
        'time_dk': df['time_dk'],
        'spot_price_eur': df['spot_price_eur'].astype(float),
        'model_prediction_eur': df['meta_ensemble_eur'].astype(float) if 'meta_ensemble_eur' in df.columns else df.get('pred_imbalance_eur', 0.0),
        'actual_settled_eur': pd.to_numeric(df.get('actual_settled_imbalance_eur', '--').astype(str).str.replace('€', '').str.replace(',', '').str.strip(), errors='coerce'),
        'action': df['agent_action'].astype(str).apply(lambda x: "BUY" if "BUY" in x else ("SELL" if "SELL" in x else "HOLD")) if 'agent_action' in df.columns else df.get('action', 'HOLD'),
        'volume_mwh': df.get('volume', 2.0).astype(float) if 'volume' in df.columns else 2.0,
        'fees_eur': 0.0,
        'net_pnl_eur': 0.0,
        'model_used': model_used,
        'deep_bilstm_eur': df.get('deep_bilstm_eur', float('nan')).astype(float),
        'transformer_tft_eur': df.get('transformer_tft_eur', float('nan')).astype(float),
        'transfer_lgb_eur': df.get('transfer_lgb_eur', float('nan')).astype(float),
        'hierarchical_eur': df.get('hierarchical_eur', float('nan')).astype(float),
        'pure15m_catboost_eur': df.get('pure15m_catboost_eur', float('nan')).astype(float)
    })
    
    # Calculate PnL if not explicitly passed
    for i, row in db_df.iterrows():
        if pd.notnull(row['actual_settled_eur']):
            direction = 1 if row['action'] == "BUY" else (-1 if row['action'] == "SELL" else 0)
            spread = row['actual_settled_eur'] - row['spot_price_eur']
            gross = spread * direction * row['volume_mwh']
            fees = (0.06 + 0.20 + 0.25) * row['volume_mwh'] if direction != 0 else 0.0
            tax = (gross - fees) * 0.22 if (gross - fees) > 0 else 0.0
            net = gross - fees - tax
            db_df.at[i, 'net_pnl_eur'] = net
            db_df.at[i, 'fees_eur'] = fees
    
    db_df.to_sql('trade_ledger', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()
    print(f"Saved {len(db_df)} records for {date_str} ({area}) to DB.")

def fetch_daily_ledger(date_str, area, model_used="Stacking-MetaEnsemble"):
    """
    Fetches the daily ledger from the database and constructs a structure
    compatible with the visualization engine.
    """
    if not os.path.exists(DB_PATH):
        return None
        
    conn = sqlite3.connect(DB_PATH)
    query = '''
        SELECT * FROM trade_ledger 
        WHERE date = ? AND price_area = ? AND model_used = ?
        ORDER BY quarter ASC
    '''
    df = pd.read_sql(query, conn, params=(date_str, area, model_used))
    conn.close()
    
    if df.empty:
        return None
        
    trade_log_df = pd.DataFrame({
        "quarter": df['quarter'],
        "spot_price": df['spot_price_eur'],
        "actual_imbalance": df['actual_settled_eur'],
        "actual_spread": df['actual_settled_eur'] - df['spot_price_eur'],
        "action": df['action'],
        "spread_capture": [(a - s) if pd.notnull(a) else 0 for a, s in zip(df['actual_settled_eur'], df['spot_price_eur'])]
    })
    
    trade_log_df.loc[trade_log_df["action"] == "BUY", "action"] = "LONG_SPOT"
    trade_log_df.loc[trade_log_df["action"] == "SELL", "action"] = "SHORT_SPOT"
    
    capital_curve = [100000.0]
    for pnl in df['net_pnl_eur']:
        capital_curve.append(capital_curve[-1] + (pnl if pd.notnull(pnl) else 0.0))
        
    pred_spreads = df['model_prediction_eur'] - df['spot_price_eur']
    
    summary = {
        "gross_pnl": df['net_pnl_eur'].sum() + df['fees_eur'].sum(),
        "fees_paid": df['fees_eur'].sum(),
        "slippage_paid": df[df['action'] != 'HOLD']['volume_mwh'].sum() * 0.25,
        "tax_paid": 0,
        "net_profit": df['net_pnl_eur'].sum(),
        "return_pct": (df['net_pnl_eur'].sum() / 100000.0) * 100,
        "initial_capital": 100000.0,
        "trade_log": trade_log_df,
        "capital_curve": capital_curve,
    }
    
    trading_summaries = [{
        "model_name": model_used,
        "pred_spread": pred_spreads.to_numpy(),
        "summary": summary
    }]
    
    return trading_summaries

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from src.db_manager import save_daily_ledger

def archive_today():
    """
    Reads the latest 96Q_future_table_{area}.csv and archives it into the database.
    Can be run via cron at 23:55 daily.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    print(f"Archiving trades for {date_str}...")
    
    for area in ["DK1", "DK2"]:
        csv_path = f"results/96Q_future_table_{area}.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Ensure it has the data
            if not df.empty:
                save_daily_ledger(df, date_str, area)
                print(f"[{area}] Successfully archived.")
            else:
                print(f"[{area}] CSV is empty.")
        else:
            print(f"[{area}] CSV not found: {csv_path}")

if __name__ == "__main__":
    archive_today()

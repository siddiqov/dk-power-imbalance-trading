# ============================================
# data_retrieval.py - COMPLETE (FIXED)
# ============================================

import requests
import pandas as pd
import duckdb
from datetime import datetime, timedelta
import time
import os
import sys
import warnings

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

warnings.filterwarnings('ignore')


# ============================================
# DATABASE MANAGER - FIXED
# ============================================

class DatabaseManager:
    """Handles all database operations with DuckDB"""

    def __init__(self, db_path="energy_data.db"):
        self.db_path = db_path
        # Only create tables if they don't exist
        self._ensure_database_exists()

    def _ensure_database_exists(self):
        """Create tables only if they don't exist (does NOT drop existing)"""
        conn = duckdb.connect(self.db_path)

        # Check if tables already exist
        result = conn.execute("""
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_schema = 'main' AND table_name = 'imbalance_prices'
        """).fetchdf()

        tables_exist = result['count'][0] > 0

        if tables_exist:
            conn.close()
            return

        # Tables don't exist - create them
        print("📦 Creating database tables...")

        # 1. Day-Ahead Prices
        conn.execute("""
            CREATE TABLE IF NOT EXISTS day_ahead_prices (
                id BIGINT PRIMARY KEY,
                time_utc TIMESTAMP,
                time_dk TIMESTAMP,
                price_area VARCHAR,
                price_dkk FLOAT,
                price_eur FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Imbalance Prices
        conn.execute("""
            CREATE TABLE IF NOT EXISTS imbalance_prices (
                id BIGINT PRIMARY KEY,
                time_utc TIMESTAMP,
                time_dk TIMESTAMP,
                price_area VARCHAR,
                imbalance_price_dkk FLOAT,
                imbalance_price_eur FLOAT,
                spot_price_eur FLOAT,
                satisfied_demand FLOAT,
                dominating_direction VARCHAR,
                afrr_up_mw FLOAT,
                afrr_down_mw FLOAT,
                mfrr_marginal_up_dkk FLOAT,
                mfrr_marginal_down_dkk FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 3. Electricity Balance
        conn.execute("""
            CREATE TABLE IF NOT EXISTS electricity_balance (
                id BIGINT PRIMARY KEY,
                time_utc TIMESTAMP,
                time_dk TIMESTAMP,
                price_area VARCHAR,
                total_load FLOAT,
                total_wind FLOAT,
                wind_offshore FLOAT,
                wind_onshore FLOAT,
                solar FLOAT,
                exchange_continent FLOAT,
                exchange_great_belt FLOAT,
                exchange_nordic FLOAT,
                exchange_gb FLOAT,
                net_exchange FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 4. Forecasts
        conn.execute("""
            CREATE TABLE IF NOT EXISTS forecasts_hour (
                id BIGINT PRIMARY KEY,
                time_utc TIMESTAMP,
                time_dk TIMESTAMP,
                price_area VARCHAR,
                forecast_type VARCHAR,
                forecast_day_ahead FLOAT,
                forecast_intraday FLOAT,
                forecast_5hour FLOAT,
                forecast_1hour FLOAT,
                forecast_current FLOAT,
                timestamp_utc TIMESTAMP,
                timestamp_dk TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 5. AFRR Activation
        conn.execute("""
            CREATE TABLE IF NOT EXISTS afrr_activation (
                id BIGINT PRIMARY KEY,
                time_utc TIMESTAMP,
                time_dk TIMESTAMP,
                price_area VARCHAR,
                afrr_activated_mw FLOAT,
                afrr_price_eur FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 6. MFRR Activation
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mfrr_activation (
                id BIGINT PRIMARY KEY,
                time_utc TIMESTAMP,
                time_dk TIMESTAMP,
                price_area VARCHAR,
                mfrr_up_mw FLOAT,
                mfrr_down_mw FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 7. ML Features
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_features (
                id BIGINT PRIMARY KEY,
                time_utc TIMESTAMP,
                time_dk TIMESTAMP,
                price_area VARCHAR,
                day_ahead_price FLOAT,
                imbalance_price FLOAT,
                total_load FLOAT,
                total_wind FLOAT,
                wind_forecast FLOAT,
                solar_forecast FLOAT,
                wind_error FLOAT,
                solar_error FLOAT,
                net_exchange FLOAT,
                price_momentum FLOAT,
                hour INT,
                day_of_week INT,
                month INT,
                target_60min FLOAT,
                target_direction INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 8. Model Predictions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                id BIGINT PRIMARY KEY,
                time_utc TIMESTAMP,
                time_dk TIMESTAMP,
                price_area VARCHAR,
                model_name VARCHAR,
                actual_imbalance FLOAT,
                predicted_imbalance FLOAT,
                confidence FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 9. Model Comparison
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_comparison (
                id BIGINT PRIMARY KEY,
                model_name VARCHAR,
                price_area VARCHAR,
                cv_mean_r2 FLOAT,
                cv_std_r2 FLOAT,
                cv_mean_mae FLOAT,
                cv_std_mae FLOAT,
                training_time FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_time_utc ON day_ahead_prices(time_utc);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_time_imbalance ON imbalance_prices(time_utc);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_time_balance ON electricity_balance(time_utc);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_area ON day_ahead_prices(price_area);")

        conn.close()
        print("  Database tables created")

    def get_connection(self):
        """Get a database connection"""
        return duckdb.connect(self.db_path)

    def insert_dataframe(self, table_name, df):
        """Insert a dataframe into a table"""
        if df.empty:
            return 0

        conn = self.get_connection()
        try:
            if 'id' not in df.columns:
                df['id'] = df.apply(lambda x: int(
                    hash(f"{x.get('time_utc', '')}{x.get('price_area', '')}") % 10 ** 15
                ), axis=1)

            table_info = conn.execute(f"PRAGMA table_info({table_name})").fetchdf()
            table_cols = [c for c in table_info['name'].tolist() if c not in ['created_at']]

            insert_cols = [col for col in table_cols if col in df.columns]

            if not insert_cols:
                return 0

            conn.register('temp_df', df)
            col_str = ', '.join(insert_cols)
            select_cols = ', '.join([f'temp_df."{col}"' for col in insert_cols])
            query = f"INSERT OR REPLACE INTO {table_name} ({col_str}) SELECT {select_cols} FROM temp_df"
            conn.execute(query)

            return len(df)
        except Exception as e:
            print(f"      Insert error: {e}")
            return 0
        finally:
            conn.close()

    def query(self, sql):
        """Execute a query and return results as dataframe"""
        conn = self.get_connection()
        try:
            return conn.execute(sql).fetchdf()
        finally:
            conn.close()

    def get_table_stats(self):
        """Get statistics for all tables"""
        conn = self.get_connection()
        try:
            tables = conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
            """).fetchdf()

            stats = {}
            for _, row in tables.iterrows():
                table_name = row['table_name']
                try:
                    count = conn.execute(f"SELECT COUNT(*) as count FROM {table_name}").fetchdf()['count'][0]
                    stats[table_name] = count
                except:
                    stats[table_name] = 0

            return stats
        finally:
            conn.close()

    def export_to_csv(self, table_name, area=None):
        """Export a table to CSV"""
        if area:
            df = self.query(f"SELECT * FROM {table_name} WHERE price_area = '{area}'")
        else:
            df = self.query(f"SELECT * FROM {table_name}")

        if df.empty:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{table_name}_{timestamp}.csv"
        filepath = os.path.join("data/csv", filename)
        os.makedirs("data/csv", exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"      Exported {len(df):,} rows to {filepath}")
        return filepath


# ============================================
# ENERGY DATA FETCHER (same as before)
# ============================================

class EnerginetDataFetcher:
    """Fetches data from Energinet Data Service API"""

    def __init__(self):
        self.base_url = "https://api.energidataservice.dk"
        self.batch_size = 50000
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'EnergyTrading/1.0'})
        print("  EnerginetDataFetcher initialized")

    def fetch_dataset(self, dataset_name, start_date, end_date, area=None, limit=None):
        """Generic dataset fetcher"""
        url = f"{self.base_url}/dataset/{dataset_name}"

        params = {
            'limit': limit or self.batch_size
        }

        if start_date:
            params['start'] = start_date.strftime('%Y-%m-%d')
        if end_date:
            # Ensure full day data is fetched (end in EDS is exclusive timestamp)
            params['end'] = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')

        try:
            print(
                f"    Fetching: {dataset_name} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            response = self.session.get(url, params=params, timeout=60)

            if response.status_code == 429:
                print("    Rate limited. Waiting 65 seconds...")
                time.sleep(65)
                return self.fetch_dataset(dataset_name, start_date, end_date, area, limit)

            if response.status_code != 200:
                print(f"    API Error {response.status_code} for {dataset_name}")
                return pd.DataFrame()

            data = response.json()
            records = data.get('records', [])

            if not records:
                print(f"    ⚠ No records found for {dataset_name}")
                return pd.DataFrame()

            df = pd.DataFrame(records)

            if area and 'PriceArea' in df.columns:
                df = df[df['PriceArea'] == area]
            elif area and 'Area' in df.columns:
                df = df[df['Area'] == area]

            print(f"     Retrieved {len(df):,} records")
            return df

        except Exception as e:
            print(f"     Error fetching {dataset_name}: {e}")
            return pd.DataFrame()

    def fetch_day_ahead_prices(self, start_date, end_date, area=None):
        df = self.fetch_dataset('DayAheadPrices', start_date, end_date, area)
        if df.empty:
            df = self.fetch_dataset('Elspotprices', start_date, end_date, area)
        if df.empty:
            return df

        try:
            time_col_utc = 'TimeUTC' if 'TimeUTC' in df.columns else 'HourUTC'
            time_col_dk = 'TimeDK' if 'TimeDK' in df.columns else 'HourDK'
            price_dkk_col = 'DayAheadPriceDKK' if 'DayAheadPriceDKK' in df.columns else 'SpotPriceDKK'
            price_eur_col = 'DayAheadPriceEUR' if 'DayAheadPriceEUR' in df.columns else 'SpotPriceEUR'

            df['time_utc'] = pd.to_datetime(df[time_col_utc])
            df['time_dk'] = pd.to_datetime(df[time_col_dk])
            df['price_area'] = df['PriceArea']
            df['price_dkk'] = pd.to_numeric(df[price_dkk_col], errors='coerce')
            df['price_eur'] = pd.to_numeric(df[price_eur_col], errors='coerce')
            df = df[['time_utc', 'time_dk', 'price_area', 'price_dkk', 'price_eur']]
            df = df.dropna(subset=['price_dkk'])
            print(f"     Day-Ahead: {len(df):,} records")
            return df
        except Exception as e:
            print(f"     Error processing Day-Ahead: {e}")
            return pd.DataFrame()

    def fetch_imbalance_prices(self, start_date, end_date, area=None):
        df = self.fetch_dataset('ImbalancePrice', start_date, end_date, area)
        if df.empty:
            return df

        try:
            df['time_utc'] = pd.to_datetime(df['TimeUTC'])
            df['time_dk'] = pd.to_datetime(df['TimeDK'])
            df['price_area'] = df['PriceArea']
            df['imbalance_price_dkk'] = pd.to_numeric(df['ImbalancePriceDKK'], errors='coerce')
            df['imbalance_price_eur'] = pd.to_numeric(df['ImbalancePriceEUR'], errors='coerce')
            df['spot_price_eur'] = pd.to_numeric(df['SpotPriceEUR'], errors='coerce')
            df['satisfied_demand'] = pd.to_numeric(df['SatisfiedDemand'], errors='coerce')
            df['dominating_direction'] = df.get('DominatingDirection', 'Unknown')
            df['afrr_up_mw'] = pd.to_numeric(df['aFRRUpMW'], errors='coerce')
            df['afrr_down_mw'] = pd.to_numeric(df['aFRRDownMW'], errors='coerce')
            df['mfrr_marginal_up_dkk'] = pd.to_numeric(df['mFRRMarginalPriceUpDKK'], errors='coerce')
            df['mfrr_marginal_down_dkk'] = pd.to_numeric(df['mFRRMarginalPriceDownDKK'], errors='coerce')

            df = df[['time_utc', 'time_dk', 'price_area',
                     'imbalance_price_dkk', 'imbalance_price_eur',
                     'spot_price_eur', 'satisfied_demand', 'dominating_direction',
                     'afrr_up_mw', 'afrr_down_mw',
                     'mfrr_marginal_up_dkk', 'mfrr_marginal_down_dkk']]
            df = df.dropna(subset=['imbalance_price_dkk'])

            print(f"     Imbalance: {len(df):,} records")
            return df
        except Exception as e:
            print(f"     Error processing Imbalance: {e}")
            return pd.DataFrame()

    def fetch_electricity_balance(self, start_date, end_date, area=None):
        """Fetch REAL 15-minute electricity balance data"""
        df = self.fetch_dataset('ElectricityBalanceNonv', start_date, end_date, area)
        if df.empty:
            return df

        try:
            df['time_utc'] = pd.to_datetime(df['HourUTC'])
            df['time_dk'] = pd.to_datetime(df['HourDK'])
            df['price_area'] = df['PriceArea']

            df['total_load'] = pd.to_numeric(df['TotalLoad'], errors='coerce')
            df['wind_offshore'] = pd.to_numeric(df['OffshoreWindPower'], errors='coerce')
            df['wind_onshore'] = pd.to_numeric(df['OnshoreWindPower'], errors='coerce')
            df['solar'] = pd.to_numeric(df['SolarPower'], errors='coerce')
            df['total_wind'] = df['wind_offshore'].fillna(0) + df['wind_onshore'].fillna(0)

            df['exchange_continent'] = pd.to_numeric(df['ExchangeContinent'], errors='coerce')
            df['exchange_great_belt'] = pd.to_numeric(df['ExchangeGreatBelt'], errors='coerce')
            df['exchange_nordic'] = pd.to_numeric(df['ExchangeNordicCountries'], errors='coerce')
            df['exchange_gb'] = pd.to_numeric(df['ExchangeGreatBritain'], errors='coerce')
            df['net_exchange'] = df['exchange_continent'].fillna(0) + \
                                 df['exchange_great_belt'].fillna(0) + \
                                 df['exchange_nordic'].fillna(0) + \
                                 df['exchange_gb'].fillna(0)

            df = df[['time_utc', 'time_dk', 'price_area',
                     'total_load', 'total_wind', 'wind_offshore', 'wind_onshore',
                     'solar', 'exchange_continent', 'exchange_great_belt',
                     'exchange_nordic', 'exchange_gb', 'net_exchange']]

            print(f"      Electricity Balance (15-min): {len(df):,} records")
            return df

        except Exception as e:
            print(f"      Error processing Electricity Balance: {e}")
            return pd.DataFrame()

    def fetch_forecasts(self, start_date, end_date, area=None):
        df = self.fetch_dataset('Forecasts_Hour', start_date, end_date, area)
        if df.empty:
            return df

        try:
            df['time_utc'] = pd.to_datetime(df['HourUTC'])
            df['time_dk'] = pd.to_datetime(df['HourDK'])
            df['price_area'] = df['PriceArea']
            df['forecast_type'] = df['ForecastType']

            if 'TimestampUTC' in df.columns:
                df['timestamp_utc'] = pd.to_datetime(df['TimestampUTC'])
            if 'TimestampDK' in df.columns:
                df['timestamp_dk'] = pd.to_datetime(df['TimestampDK'])

            df['forecast_day_ahead'] = pd.to_numeric(df.get('Forecast Day Ahead', df.get('ForecastDayAhead', 0)),
                                                     errors='coerce')
            df['forecast_intraday'] = pd.to_numeric(df.get('Forecast Intraday', df.get('ForecastIntraday', 0)),
                                                    errors='coerce')
            df['forecast_5hour'] = pd.to_numeric(df.get('Forecast 5 Hour', df.get('Forecast5Hour', 0)), errors='coerce')
            df['forecast_1hour'] = pd.to_numeric(df.get('Forecast 1 Hour', df.get('Forecast1Hour', 0)), errors='coerce')
            df['forecast_current'] = pd.to_numeric(df.get('Forecast Current', df.get('ForecastCurrent', 0)),
                                                   errors='coerce')

            df = df[['time_utc', 'time_dk', 'price_area', 'forecast_type',
                     'forecast_day_ahead', 'forecast_intraday', 'forecast_5hour',
                     'forecast_1hour', 'forecast_current',
                     'timestamp_utc', 'timestamp_dk']]

            print(f"      Forecasts: {len(df):,} records")
            return df
        except Exception as e:
            print(f"      Error processing forecasts: {e}")
            return pd.DataFrame()

    def fetch_afrr(self, start_date, end_date, area=None):
        df = self.fetch_dataset('AfrrEnergyActivation', start_date, end_date, area)
        if df.empty:
            return df

        try:
            df['time_utc'] = pd.to_datetime(df['TimeMsUTC'])
            df['time_dk'] = pd.to_datetime(df['TimeMsDK'])
            df['price_area'] = df['PriceArea']
            df['afrr_activated_mw'] = pd.to_numeric(df['aFRR_Activated'], errors='coerce')
            df['afrr_price_eur'] = pd.to_numeric(df['aFRR_ActivatedEUR'], errors='coerce')
            df = df[['time_utc', 'time_dk', 'price_area', 'afrr_activated_mw', 'afrr_price_eur']]
            df = df.dropna(subset=['afrr_activated_mw'])

            print(f"     AFRR: {len(df):,} records")
            return df
        except Exception as e:
            print(f"      Error processing AFRR: {e}")
            return pd.DataFrame()

    def fetch_mfrr(self, start_date, end_date, area=None):
        df = self.fetch_dataset('MfrrEnergyActivationMarket', start_date, end_date, area)
        if df.empty:
            return df

        try:
            df['time_utc'] = pd.to_datetime(df['TimeUTC'])
            df['time_dk'] = pd.to_datetime(df['TimeDK'])
            df['price_area'] = df['PriceArea']
            df['mfrr_up_mw'] = pd.to_numeric(df['TotalmFRRUpMW'], errors='coerce').fillna(0)
            df['mfrr_down_mw'] = pd.to_numeric(df['TotalmFRRDownMW'], errors='coerce').fillna(0)
            df = df[['time_utc', 'time_dk', 'price_area', 'mfrr_up_mw', 'mfrr_down_mw']]

            print(f"     MFRR: {len(df):,} records")
            return df
        except Exception as e:
            print(f"     Error processing MFRR: {e}")
            return pd.DataFrame()


# ============================================
# DATA PIPELINE
# ============================================

class DataPipeline:
    """Orchestrates data fetching and storage"""

    def __init__(self, db_path="energy_data.db"):
        self.db = DatabaseManager(db_path)
        self.fetcher = EnerginetDataFetcher()

    def fetch_all_data(self, area='DK1', start_date=None, end_date=None, export_csv=True):
        if start_date is None:
            start_date = datetime(2025, 4, 23)
        if end_date is None:
            end_date = datetime.now()

        print(f"\n  Fetching ALL data for {area}")
        print(f"   Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print("=" * 50)

        results = {}

        print("\n     Fetching Day-Ahead prices...")
        day_ahead = self.fetcher.fetch_day_ahead_prices(start_date, end_date, area)
        if not day_ahead.empty:
            count = self.db.insert_dataframe('day_ahead_prices', day_ahead)
            results['day_ahead_prices'] = count

        print("\n     Fetching Imbalance prices...")
        imbalance = self.fetcher.fetch_imbalance_prices(start_date, end_date, area)
        if not imbalance.empty:
            count = self.db.insert_dataframe('imbalance_prices', imbalance)
            results['imbalance_prices'] = count

        print("\n     Fetching Electricity Balance...")
        balance = self.fetcher.fetch_electricity_balance(start_date, end_date, area)
        if not balance.empty:
            count = self.db.insert_dataframe('electricity_balance', balance)
            results['electricity_balance'] = count

        print("\n     Fetching Forecasts...")
        forecasts = self.fetcher.fetch_forecasts(start_date, end_date, area)
        if not forecasts.empty:
            count = self.db.insert_dataframe('forecasts_hour', forecasts)
            results['forecasts_hour'] = count

        print("\n     Fetching AFRR...")
        afrr = self.fetcher.fetch_afrr(start_date, end_date, area)
        if not afrr.empty:
            count = self.db.insert_dataframe('afrr_activation', afrr)
            results['afrr_activation'] = count

        print("\n     Fetching MFRR...")
        mfrr = self.fetcher.fetch_mfrr(start_date, end_date, area)
        if not mfrr.empty:
            count = self.db.insert_dataframe('mfrr_activation', mfrr)
            results['mfrr_activation'] = count

        print("\n" + "=" * 50)
        print("  DATA FETCHING COMPLETE")
        print("=" * 50)

        total_records = 0
        for table, count in results.items():
            print(f"   {table}: {count:,} records")
            total_records += count

        print(f"\n   TOTAL: {total_records:,} records")

        if export_csv:
            print("\n     Exporting to CSV...")
            for table in results.keys():
                self.db.export_to_csv(table, area)

        return results


if __name__ == "__main__":
    pipeline = DataPipeline()
    start_date = datetime(2025, 3, 4)
    end_date = datetime.now()
    results = pipeline.fetch_all_data('DK1', start_date, end_date)
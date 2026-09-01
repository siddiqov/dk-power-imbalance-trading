# ============================================
# model_training.py - REGIME-SWITCHING MODEL
# CORRECTED: Regime definition based on actual direction
# ============================================

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
import time
import os
import warnings

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


# ============================================
# FEATURE ENGINEER - CORRECT REGIME DEFINITION
# ============================================

class FeatureEngineer:
    """Creates features for REGIME-SWITCHING prediction with CORRECT regime definition"""

    def __init__(self, db):
        self.db = db
        self.feature_columns = None

    def create_features(self, area, start_date, end_date, prediction_horizon=4):
        """
        Create features for regime prediction
        Regimes: 0=Downward Regulation, 1=Neutral, 2=Upward Regulation

        prediction_horizon: number of 15-min periods ahead (4 = 60 min)
        """
        print(f"\n🔧 Creating REGIME-SWITCHING features for {area}")
        print(f"   Predicting {prediction_horizon * 15} minutes ahead")
        print("-" * 50)

        # 1. Get Imbalance Prices
        print("   Querying imbalance_prices...")
        imbalance = self.db.query(f"""
            SELECT time_utc, time_dk, 
                   imbalance_price_dkk as imbalance_price,
                   satisfied_demand,
                   afrr_up_mw,
                   afrr_down_mw,
                   dominating_direction
            FROM imbalance_prices
            WHERE price_area = '{area}'
              AND time_utc BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY time_utc
        """)
        print(f"    Imbalance: {len(imbalance):,} records")

        if imbalance.empty:
            print("      No Imbalance data found!")
            return pd.DataFrame()

        # Remove duplicates
        imbalance = imbalance.drop_duplicates(subset=['time_utc'])
        print(f"    After deduplication: {len(imbalance):,} records")

        features = imbalance.copy()
        features = features.set_index('time_utc')

        # 2. Get Day-Ahead Prices
        print("   Querying day_ahead_prices...")
        day_ahead = self.db.query(f"""
            SELECT time_utc, price_dkk as day_ahead_price
            FROM day_ahead_prices
            WHERE price_area = '{area}'
              AND time_utc BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY time_utc
        """)
        print(f"    Day-Ahead: {len(day_ahead):,} records")

        if not day_ahead.empty:
            day_ahead = day_ahead.drop_duplicates(subset=['time_utc'])
            day_ahead = day_ahead.set_index('time_utc').sort_index()
            day_ahead_15min = day_ahead.resample('15T').ffill()
            features = features.join(day_ahead_15min, how='left')
            features['day_ahead_price'] = features['day_ahead_price'].ffill().fillna(0)
            print("      Day-Ahead resampled from hourly to 15-min")
        else:
            features['day_ahead_price'] = 0

        # 3. Get Electricity Balance (REAL wind, solar, load, exchange)
        print("   Querying electricity_balance...")
        balance = self.db.query(f"""
            SELECT time_utc, 
                   total_load,
                   total_wind,
                   wind_offshore,
                   wind_onshore,
                   solar,
                   net_exchange
            FROM electricity_balance
            WHERE price_area = '{area}'
              AND time_utc BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY time_utc
        """)
        print(f"    Electricity Balance: {len(balance):,} records")

        if not balance.empty:
            balance = balance.drop_duplicates(subset=['time_utc'])
            balance = balance.set_index('time_utc').sort_index()
            features = features.join(balance, how='left')

            for col in ['total_load', 'total_wind', 'wind_offshore', 'wind_onshore', 'solar', 'net_exchange']:
                features[col] = features[col].ffill().fillna(0)
            print("      Electricity Balance joined (already 15-min)")
        else:
            print("    ⚠️ No Electricity Balance data - using zeros")
            features['total_load'] = 0
            features['total_wind'] = 0
            features['wind_offshore'] = 0
            features['wind_onshore'] = 0
            features['solar'] = 0
            features['net_exchange'] = 0

        # 4. Get AFRR Activation
        print("   Querying afrr_activation...")
        afrr = self.db.query(f"""
            SELECT time_utc, afrr_activated_mw as afrr_activated
            FROM afrr_activation
            WHERE price_area = '{area}'
              AND time_utc BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY time_utc
        """)

        if not afrr.empty:
            afrr = afrr.drop_duplicates(subset=['time_utc'])
            afrr = afrr.set_index('time_utc').sort_index()
            features = features.join(afrr, how='left')
            features['afrr_activated'] = features['afrr_activated'].ffill().fillna(0)
            print("      AFRR joined (already 15-min)")
        else:
            features['afrr_activated'] = 0

        # 5. Get MFRR Activation
        print("   Querying mfrr_activation...")
        mfrr = self.db.query(f"""
            SELECT time_utc, mfrr_up_mw, mfrr_down_mw
            FROM mfrr_activation
            WHERE price_area = '{area}'
              AND time_utc BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY time_utc
        """)

        if not mfrr.empty:
            mfrr = mfrr.drop_duplicates(subset=['time_utc'])
            mfrr = mfrr.set_index('time_utc').sort_index()
            mfrr_15min = mfrr.resample('15T').ffill()
            features = features.join(mfrr_15min, how='left')
            features['mfrr_up'] = features['mfrr_up_mw'].ffill().fillna(0)
            features['mfrr_down'] = features['mfrr_down_mw'].ffill().fillna(0)
            print("      MFRR resampled from hourly to 15-min")
        else:
            features['mfrr_up'] = 0
            features['mfrr_down'] = 0

        # 6. Get Forecasts
        print("   Querying forecasts_hour...")

        wind_forecast = self.db.query(f"""
            SELECT time_utc, 
                   AVG(forecast_current) as wind_forecast
            FROM forecasts_hour
            WHERE price_area = '{area}'
              AND forecast_type LIKE '%Wind%'
              AND time_utc BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY time_utc
            ORDER BY time_utc
        """)

        if not wind_forecast.empty:
            wind_forecast = wind_forecast.drop_duplicates(subset=['time_utc'])
            wind_forecast = wind_forecast.set_index('time_utc').sort_index()
            wind_forecast_15min = wind_forecast.resample('15T').ffill()
            features = features.join(wind_forecast_15min, how='left')
            features['wind_forecast'] = features['wind_forecast'].ffill().fillna(0)
            print("      Wind Forecast resampled from hourly to 15-min")
        else:
            features['wind_forecast'] = 0

        solar_forecast = self.db.query(f"""
            SELECT time_utc, 
                   AVG(forecast_current) as solar_forecast
            FROM forecasts_hour
            WHERE price_area = '{area}'
              AND forecast_type LIKE '%Solar%'
              AND time_utc BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY time_utc
            ORDER BY time_utc
        """)

        if not solar_forecast.empty:
            solar_forecast = solar_forecast.drop_duplicates(subset=['time_utc'])
            solar_forecast = solar_forecast.set_index('time_utc').sort_index()
            solar_forecast_15min = solar_forecast.resample('15T').ffill()
            features = features.join(solar_forecast_15min, how='left')
            features['solar_forecast'] = features['solar_forecast'].ffill().fillna(0)
            print("      Solar Forecast resampled from hourly to 15-min")
        else:
            features['solar_forecast'] = 0

        # Reset index
        features = features.reset_index()
        features = features.dropna(subset=['imbalance_price', 'day_ahead_price'])
        features['price_area'] = area

        features = features.sort_values('time_utc').reset_index(drop=True)

        print(f"\n      Merged data: {len(features):,} records")

        # ============================================
        # CAUSAL FEATURES (As client requested)
        # ============================================

        # 1. Wind Forecast Error (actual - forecast)
        # Negative = wind below forecast → Upward regulation likely
        features['wind_error'] = features['total_wind'] - features['wind_forecast']

        # 2. Solar Forecast Error
        features['solar_error'] = features['solar'] - features['solar_forecast']

        # 3. Load Error (using satisfied_demand as actual)
        features['load_error'] = features['satisfied_demand'] - features['total_load']

        # 4. Interconnector Flow (positive = import, absorbs imbalances)
        features['interconnector_flow'] = features['net_exchange']

        # 5. Intraday Spread (imbalance - day_ahead as proxy for market sentiment)
        features['intraday_spread'] = features['imbalance_price'] - features['day_ahead_price']

        # 6. Activation indicators
        features['afrr_active'] = (features['afrr_activated'] > 0).astype(int)
        features['mfrr_active'] = ((features['mfrr_up'] > 0) | (features['mfrr_down'] > 0)).astype(int)

        # 7. Time features
        features['hour'] = features['time_utc'].dt.hour
        features['day_of_week'] = features['time_utc'].dt.dayofweek
        features['month'] = features['time_utc'].dt.month

        # 8. Price momentum
        features['price_momentum'] = features['day_ahead_price'].diff(4) / 4

        # 9. Imbalance momentum
        features['imbalance_momentum'] = features['imbalance_price'].diff(4) / 4

        # ============================================
        # LAGGED CAUSAL FEATURES (NO LEAKAGE!)
        # ============================================

        causal_cols = [
            'wind_error', 'solar_error', 'load_error',
            'interconnector_flow', 'intraday_spread',
            'afrr_activated', 'mfrr_up', 'mfrr_down',
            'day_ahead_price', 'imbalance_price',
            'afrr_active', 'mfrr_active',
            'price_momentum', 'imbalance_momentum'
        ]

        for lag in [1, 2, 4, 8, 16]:
            for col in causal_cols:
                if col in features.columns:
                    features[f'{col}_lag_{lag}'] = features[col].shift(lag)

        # ============================================
        # CORRECT REGIME LABELING
        # Based on ACTUAL price direction, not quantiles!
        # ============================================

        # Define future imbalance price
        features['future_imbalance'] = features['imbalance_price'].shift(-prediction_horizon)

        # Calculate price change
        features['price_change'] = features['future_imbalance'] - features['imbalance_price']

        # ============================================
        # REGIME DEFINITION BASED ON DIRECTION
        # ============================================

        # Use meaningful thresholds based on price volatility
        # Typical imbalance prices range from -2000 to +2000 DKK/MWh
        # So 50-100 DKK/MWh is a meaningful change

        threshold_up = 50  # Price increase > 50 = Upward Regulation
        threshold_down = -50  # Price decrease < -50 = Downward Regulation

        # Default: Neutral
        features['regime'] = 1

        # Upward Regulation: Price increases significantly
        features.loc[features['price_change'] > threshold_up, 'regime'] = 2

        # Downward Regulation: Price decreases significantly
        features.loc[features['price_change'] < threshold_down, 'regime'] = 0

        # ============================================
        # USE DOMINATING DIRECTION IF AVAILABLE
        # This is the ACTUAL regulation direction from Energinet!
        # ============================================

        if 'dominating_direction' in features.columns:
            # dominating_direction: 1 = Upward, -1 = Downward, 0 = Neutral
            # Map to our regime labels: 0=Downward, 1=Neutral, 2=Upward
            features.loc[features['dominating_direction'] == 1, 'regime'] = 2
            features.loc[features['dominating_direction'] == -1, 'regime'] = 0
            features.loc[features['dominating_direction'] == 0, 'regime'] = 1
            print("      Using dominating_direction for regime labeling")

        # ============================================
        # ALSO USE ACTIVATION DATA AS CONFIRMATION
        # ============================================

        # If AFRR Up is activated → Upward Regulation
        if 'afrr_up_mw' in features.columns:
            features.loc[features['afrr_up_mw'] > 5, 'regime'] = 2

        # If AFRR Down is activated → Downward Regulation
        if 'afrr_down_mw' in features.columns:
            features.loc[features['afrr_down_mw'] > 5, 'regime'] = 0

        # Drop rows with no target
        features = features.dropna(subset=['regime'])

        print(f"\n      Regime Distribution:")
        regime_counts = features['regime'].value_counts().sort_index()
        regime_names = ['Downward Regulation', 'Neutral', 'Upward Regulation']
        for r, count in regime_counts.items():
            pct = count / len(features) * 100
            print(f"      {regime_names[r]}: {count:,} ({pct:.1f}%)")

        print(f"\n  Created {len(features):,} regime prediction records")
        print(f"   Regimes: 0=Downward (price falls), 1=Neutral, 2=Upward (price rises)")

        # Define feature columns (only lagged causal features + time)
        exclude_cols = ['time_utc', 'time_dk', 'price_area', 'id', 'created_at',
                        'satisfied_demand', 'future_imbalance', 'price_change', 'regime',
                        'imbalance_price', 'day_ahead_price',
                        'total_wind', 'wind_forecast', 'solar_forecast',
                        'total_load', 'solar', 'net_exchange',
                        'afrr_activated', 'mfrr_up', 'mfrr_down',
                        'afrr_up_mw', 'afrr_down_mw', 'dominating_direction']

        self.feature_columns = [col for col in features.columns if col not in exclude_cols]
        # Keep only lagged features + time features
        self.feature_columns = [col for col in self.feature_columns if
                                'lag_' in col or col in ['hour', 'day_of_week', 'month']]

        print(f"\n   Using {len(self.feature_columns)} causal features")
        print(f"   Features: Wind error, Solar error, Load error, Interconnector flow, Spreads")

        # Store in database
        self.db.insert_dataframe('ml_features', features)

        return features


# ============================================
# MODEL TRAINER
# ============================================

class ModelTrainer:
    """Trains regime-switching classifiers"""

    def __init__(self, db=None):
        self.db = db
        self.models = {}
        self.scaler = None
        self.imputer = None
        self.features = None
        self.results = {}
        self.area = None
        self.regime_names = ['Downward', 'Neutral', 'Upward']

    def train_models_cv(self, features_df, target_col='regime', n_splits=5, export_dataset=False):
        """Train models with Time-Series Cross-Validation"""
        if features_df.empty:
            raise ValueError("Empty features DataFrame")

        self.area = features_df['price_area'].iloc[0] if 'price_area' in features_df.columns else 'Unknown'

        exclude_cols = ['time_utc', 'time_dk', 'price_area', 'id', 'created_at',
                        'regime', 'future_imbalance', 'price_change']

        self.features = [col for col in features_df.columns if col not in exclude_cols]
        self.features = [col for col in self.features if 'lag_' in col or col in ['hour', 'day_of_week', 'month']]

        if len(self.features) < 2:
            print("  Not enough features")
            return {}

        features_df = features_df.sort_values('time_utc').reset_index(drop=True)

        X = features_df[self.features].values
        y = features_df[target_col].values

        print(f"\n🤖 Training REGIME-SWITCHING models for {self.area}")
        print("-" * 50)
        print(f"   Features: {len(self.features)} (causal: wind/solar/load errors, interconnector, spreads)")
        print(f"   Total samples: {len(X):,}")
        print(f"   CV Splits: {n_splits}")
        print(f"   Target: 0=Downward, 1=Neutral, 2=Upward")

        # Handle NaN values
        self.imputer = SimpleImputer(strategy='median')
        X = self.imputer.fit_transform(X)

        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        results = {}

        # Random Forest Classifier
        print("    🌳 Training Random Forest Classifier...")
        try:
            start_time = time.time()
            rf_scores = []
            rf_accuracies = []

            for train_idx, test_idx in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42,
                                            n_jobs=-1)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                rf_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                rf_accuracies.append(accuracy_score(y_test, y_pred))

            results['RandomForest'] = {
                'cv_mean_f1': np.mean(rf_scores),
                'cv_std_f1': np.std(rf_scores),
                'cv_mean_accuracy': np.mean(rf_accuracies),
                'cv_std_accuracy': np.std(rf_accuracies),
                'training_time': time.time() - start_time,
                'model': rf,
                'cv_scores': rf_scores
            }
        except Exception as e:
            print(f"      RandomForest failed: {e}")

        # XGBoost Classifier
        if XGB_AVAILABLE:
            print("    ⚡ Training XGBoost Classifier...")
            try:
                start_time = time.time()
                xgb_scores = []
                xgb_accuracies = []

                for train_idx, test_idx in tscv.split(X_scaled):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    xgb_model = xgb.XGBClassifier(
                        n_estimators=150, max_depth=8, learning_rate=0.05,
                        random_state=42, n_jobs=-1, verbosity=0
                    )
                    xgb_model.fit(X_train, y_train)
                    y_pred = xgb_model.predict(X_test)

                    xgb_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                    xgb_accuracies.append(accuracy_score(y_test, y_pred))

                results['XGBoost'] = {
                    'cv_mean_f1': np.mean(xgb_scores),
                    'cv_std_f1': np.std(xgb_scores),
                    'cv_mean_accuracy': np.mean(xgb_accuracies),
                    'cv_std_accuracy': np.std(xgb_accuracies),
                    'training_time': time.time() - start_time,
                    'model': xgb_model,
                    'cv_scores': xgb_scores
                }
            except Exception as e:
                print(f"      XGBoost failed: {e}")

        # LightGBM Classifier
        if LGB_AVAILABLE:
            print("    💡 Training LightGBM Classifier...")
            try:
                start_time = time.time()
                lgb_scores = []
                lgb_accuracies = []

                for train_idx, test_idx in tscv.split(X_scaled):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    lgb_model = lgb.LGBMClassifier(
                        n_estimators=150, max_depth=8, learning_rate=0.05,
                        random_state=42, n_jobs=-1, verbose=-1
                    )
                    lgb_model.fit(X_train, y_train)
                    y_pred = lgb_model.predict(X_test)

                    lgb_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                    lgb_accuracies.append(accuracy_score(y_test, y_pred))

                results['LightGBM'] = {
                    'cv_mean_f1': np.mean(lgb_scores),
                    'cv_std_f1': np.std(lgb_scores),
                    'cv_mean_accuracy': np.mean(lgb_accuracies),
                    'cv_std_accuracy': np.std(lgb_accuracies),
                    'training_time': time.time() - start_time,
                    'model': lgb_model,
                    'cv_scores': lgb_scores
                }
            except Exception as e:
                print(f"      LightGBM failed: {e}")

        # Gradient Boosting Classifier (with NaN handling)
        print("    🚀 Training Gradient Boosting Classifier...")
        try:
            start_time = time.time()
            gb_scores = []
            gb_accuracies = []

            for train_idx, test_idx in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                gb = GradientBoostingClassifier(n_estimators=150, max_depth=8, learning_rate=0.05, random_state=42)
                gb.fit(X_train, y_train)
                y_pred = gb.predict(X_test)

                gb_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                gb_accuracies.append(accuracy_score(y_test, y_pred))

            results['GradientBoosting'] = {
                'cv_mean_f1': np.mean(gb_scores),
                'cv_std_f1': np.std(gb_scores),
                'cv_mean_accuracy': np.mean(gb_accuracies),
                'cv_std_accuracy': np.std(gb_accuracies),
                'training_time': time.time() - start_time,
                'model': gb,
                'cv_scores': gb_scores
            }
        except Exception as e:
            print(f"      GradientBoosting failed: {e}")

        self.models = {name: results[name]['model'] for name in results if name in results}
        self.results = results

        if results:
            print("\n" + "=" * 70)
            print("  REGIME-SWITCHING MODEL PERFORMANCE")
            print("=" * 70)

            comparison = []
            for name, metrics in results.items():
                comparison.append({
                    'Model': name,
                    'Accuracy': f"{metrics['cv_mean_accuracy']:.4f}",
                    'F1 Score': f"{metrics['cv_mean_f1']:.4f}",
                    'Time (s)': metrics.get('training_time', 0)
                })

            df_comp = pd.DataFrame(comparison)
            df_comp = df_comp.sort_values('F1 Score', ascending=False)
            print("\n" + df_comp.to_string(index=False))

            best_model = df_comp.iloc[0]['Model']
            best_acc = df_comp.iloc[0]['Accuracy']
            print(f"\n🏆 Best Model: {best_model} (Accuracy = {best_acc})")

            # Get the best model's predictions for classification report
            best_results = results[best_model]
            if 'predictions' not in best_results:
                # Generate predictions for the best model
                best_model_obj = best_results['model']
                # We need to train on full data to get predictions
                X_full_scaled = X_scaled  # Already scaled
                y_pred_full = best_model_obj.predict(X_full_scaled)
                best_results['predictions'] = y_pred_full

            print("\n  Classification Report:")
            print(classification_report(y, best_results.get('predictions', y),
                                        target_names=['Downward', 'Neutral', 'Upward']))

        # Export dataset if requested
        if export_dataset:
            self.export_dataset(features_df)

        self._store_results(results)

        return results

    def _store_results(self, results):
        """Store model results in database"""
        if self.db is None:
            return

        for name, metrics in results.items():
            df = pd.DataFrame([{
                'model_name': name,
                'price_area': self.area,
                'cv_mean_r2': metrics['cv_mean_accuracy'],
                'cv_std_r2': metrics['cv_std_accuracy'],
                'cv_mean_mae': 0,
                'cv_std_mae': 0,
                'training_time': metrics.get('training_time', 0)
            }])
            self.db.insert_dataframe('model_comparison', df)

    def export_dataset(self, features_df):
        """Export the dataset to CSV for analysis"""
        if features_df.empty:
            print("  No data to export")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"regime_dataset_{self.area}_{timestamp}.csv"
        filepath = os.path.join("data/csv", filename)
        os.makedirs("data/csv", exist_ok=True)

        # Create a copy with relevant columns
        export_df = features_df[['time_utc', 'time_dk', 'price_area', 'regime']].copy()
        export_df['regime_name'] = export_df['regime'].map({0: 'Downward', 1: 'Neutral', 2: 'Upward'})
        export_df.to_csv(filepath, index=False)
        print(f"  Regime dataset exported to {filepath}")
        return filepath

    def predict_proba(self, features_df, model_name=None):
        """Get regime probabilities"""
        if model_name is None:
            model_name = list(self.models.keys())[0]

        if model_name not in self.models:
            model_name = list(self.models.keys())[0]

        model = self.models[model_name]

        X = features_df[self.features].values
        X = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X)

        return model.predict_proba(X_scaled)

    def predict(self, features_df, model_name=None):
        """Predict regimes"""
        if model_name is None:
            model_name = list(self.models.keys())[0]

        if model_name not in self.models:
            model_name = list(self.models.keys())[0]

        model = self.models[model_name]

        X = features_df[self.features].values
        X = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X)

        return model.predict(X_scaled)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    from data_retrieval import DatabaseManager

    db = DatabaseManager()
    start_date = datetime(2025, 4, 23)
    end_date = datetime.now()

    print("\n  Database Statistics:")
    stats = db.get_table_stats()
    for table, count in stats.items():
        print(f"   {table}: {count:,} records")

    if stats.get('imbalance_prices', 0) == 0:
        print("\n⚠️ No imbalance data found! Run data_retrieval first.")
    else:
        # Create features
        feature_engineer = FeatureEngineer(db)
        features = feature_engineer.create_features('DK1', start_date, end_date)

        if not features.empty:
            # Train models
            trainer = ModelTrainer(db)
            results = trainer.train_models_cv(features, n_splits=5)
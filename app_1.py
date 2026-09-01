# ============================================
# app.py - Flask Dashboard for Scikit-learn 1.5.1 Models
# ============================================

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from flask import Flask, render_template, jsonify, request, send_file
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from io import BytesIO
import base64
import warnings

warnings.filterwarnings('ignore')

# ============================================
# FIX: Handle Scikit-learn 1.5.1 Compatibility
# ============================================

# Method 1: Try to import the required module for sklearn 1.5.1
try:
    from sklearn._loss import CyPinballLoss

    print("✅ sklearn._loss.CyPinballLoss imported successfully")
except ImportError:
    try:
        from sklearn._loss._loss import CyPinballLoss

        print("✅ sklearn._loss._loss.CyPinballLoss imported successfully")
    except ImportError:
        print("⚠️ CyPinballLoss not found - creating dummy for unpickling")
        # Create a dummy module structure for sklearn 1.5.1
        import types

        # Create the _loss module
        loss_module = types.ModuleType('sklearn._loss')


        class DummyCyPinballLoss:
            pass


        loss_module.CyPinballLoss = DummyCyPinballLoss

        # Create the _loss._loss submodule
        loss_sub_module = types.ModuleType('sklearn._loss._loss')
        loss_sub_module.CyPinballLoss = DummyCyPinballLoss
        loss_sub_module.__pyx_unpickle_CyPinballLoss = lambda x: DummyCyPinballLoss()

        # Register the modules
        sys.modules['sklearn._loss'] = loss_module
        sys.modules['sklearn._loss._loss'] = loss_sub_module

        print("✅ Dummy sklearn._loss._loss module created for unpickling")

# Method 2: Also handle scikit-learn 1.6+ structure if needed
try:
    from sklearn.ensemble import GradientBoostingRegressor

    print("✅ sklearn.ensemble available")
except ImportError:
    print("❌ sklearn.ensemble not available")


# ============================================
# Define Required Classes (Fallbacks)
# ============================================

class OptimizedRegimeClassifier:
    """Regime classifier - matches training code"""

    def __init__(self, n_regimes=4):
        self.n_regimes = n_regimes
        self.classifier = None
        self.scaler = None
        self.regime_names = ['Normal', 'Supply_Constrained', 'Demand_Constrained', 'Volatile']
        self.n_features = None

    def predict(self, X, return_proba=False):
        import numpy as np
        if self.classifier is None:
            n = len(X) if hasattr(X, '__len__') else 1
            return {
                'regime': np.zeros(n).astype(int),
                'regime_names': ['Normal'] * n,
                'confidence': np.ones(n) * 0.75,
                'probabilities': np.array([[0.75, 0.10, 0.10, 0.05]] * n)
            }

        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        if return_proba:
            probs = self.classifier.predict_proba(X_scaled)
            predictions = np.argmax(probs, axis=1)
            return {
                'regime': predictions,
                'regime_names': [self.regime_names[p] for p in predictions],
                'probabilities': probs,
                'confidence': np.max(probs, axis=1)
            }
        else:
            predictions = self.classifier.predict(X_scaled)
            return [self.regime_names[p] for p in predictions]


class OptimizedProbabilisticForecaster:
    """Probabilistic forecaster - matches training code"""

    def __init__(self, n_estimators=100, n_quantiles=5):
        self.n_estimators = n_estimators
        self.n_quantiles = n_quantiles
        self.quantiles = np.linspace(0.1, 0.9, n_quantiles)
        self.models = {}
        self.quantile_models = {}
        self.scaler = None
        self.n_features = None

    def predict(self, X, return_interval=True, return_quantiles=True):
        import numpy as np
        if not self.models:
            n = len(X) if hasattr(X, '__len__') else 1
            return {'mean': np.zeros(n), 'std': np.zeros(n), 'lower': np.zeros(n) - 10, 'upper': np.zeros(n) + 10}

        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X.astype(float))
        else:
            X_scaled = X

        predictions = []
        for model in self.models.values():
            try:
                predictions.append(model.predict(X_scaled))
            except:
                continue

        if not predictions:
            return {'mean': np.zeros(len(X)), 'std': np.zeros(len(X)),
                    'lower': np.zeros(len(X)) - 10, 'upper': np.zeros(len(X)) + 10}

        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        result = {'mean': mean_pred, 'std': std_pred}

        if return_interval:
            result['lower'] = mean_pred - 1.96 * std_pred
            result['upper'] = mean_pred + 1.96 * std_pred

        return result


class MultiStepDataPreprocessor:
    """Preprocessor that creates features for predictions"""

    def __init__(self):
        self.scaler = None
        self.feature_names = []
        self.fixed_feature_columns = None
        self.n_predictions = 5

    def create_features(self, df, is_training=True):
        df_copy = df.copy()

        # Column renaming
        rename_map = {
            'TimeUTC': 'timestamp', 'TimeDK': 'time_dk', 'PriceArea': 'price_area',
            'Satisfied Demand (MW)': 'satisfied_demand', 'SatisfiedDemand': 'satisfied_demand',
            'Imbalance Price (EUR)': 'imbalance_price_eur', 'ImbalancePriceEUR': 'imbalance_price_eur',
            'Spot Price (EUR)': 'spot_price_eur', 'SpotPriceEUR': 'spot_price_eur',
            'Dominating Direction': 'dominating_direction', 'DominatingDirection': 'dominating_direction',
            'aFRR Up (MWh)': 'afrr_up_mwh', 'aFRRUpMW': 'afrr_up_mwh',
            'aFRR VWA Up (EUR)': 'afrr_vwa_up_eur', 'aFRRVWAUpEUR': 'afrr_vwa_up_eur',
            'aFRR Down (MWh)': 'afrr_down_mwh', 'aFRRDownMW': 'afrr_down_mwh',
            'aFRR VWA Down (EUR)': 'afrr_vwa_down_eur', 'aFRRVWADownEUR': 'afrr_vwa_down_eur',
            'mFRR Marginal Price Up (EUR)': 'mfrr_up_eur', 'mFRRMarginalPriceUpEUR': 'mfrr_up_eur',
            'mFRR Marginal Price Down (EUR)': 'mfrr_down_eur', 'mFRRMarginalPriceDownEUR': 'mfrr_down_eur'
        }

        for old, new in rename_map.items():
            if old in df_copy.columns:
                df_copy[new] = df_copy[old]

        # Data type conversion
        numeric_cols = [
            'satisfied_demand', 'imbalance_price_eur', 'imbalance_price_dkk',
            'spot_price_eur', 'dominating_direction',
            'afrr_up_mwh', 'afrr_vwa_up_eur',
            'afrr_down_mwh', 'afrr_vwa_down_eur',
            'mfrr_up_eur', 'mfrr_down_eur'
        ]

        for col in numeric_cols:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)

        # Time features
        if 'timestamp' in df_copy.columns:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            df_copy['hour'] = df_copy['timestamp'].dt.hour.astype(float)
            df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek.astype(float)
            df_copy['month'] = df_copy['timestamp'].dt.month.astype(float)
            df_copy['quarter'] = df_copy['timestamp'].dt.quarter.astype(float)
            df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(float)
            df_copy['is_peak'] = df_copy['hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(float)
            df_copy['sin_hour'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
            df_copy['cos_hour'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
            rule_change_date = pd.Timestamp('2026-03-04')
            df_copy['is_new_market_rule'] = (df_copy['timestamp'] >= rule_change_date).astype(float)

        # Market Structure Features
        if 'afrr_vwa_up_eur' in df_copy.columns and 'afrr_vwa_down_eur' in df_copy.columns:
            df_copy['afrr_price_spread'] = (df_copy['afrr_vwa_up_eur'] - df_copy['afrr_vwa_down_eur']).astype(float)
            df_copy['afrr_price_ratio'] = (df_copy['afrr_vwa_up_eur'] / (df_copy['afrr_vwa_down_eur'] + 1e-6)).astype(
                float)

        if 'afrr_up_mwh' in df_copy.columns and 'afrr_down_mwh' in df_copy.columns:
            df_copy['afrr_net_volume'] = (df_copy['afrr_up_mwh'] - df_copy['afrr_down_mwh']).astype(float)
            df_copy['afrr_total_volume'] = (df_copy['afrr_up_mwh'] + df_copy['afrr_down_mwh']).astype(float)

        if 'mfrr_up_eur' in df_copy.columns and 'mfrr_down_eur' in df_copy.columns:
            df_copy['mfrr_price_spread'] = (df_copy['mfrr_up_eur'] - df_copy['mfrr_down_eur']).astype(float)
            df_copy['mfrr_activated'] = ((df_copy['mfrr_up_eur'] > 0) | (df_copy['mfrr_down_eur'] > 0)).astype(float)

        if 'imbalance_price_eur' in df_copy.columns and 'spot_price_eur' in df_copy.columns:
            df_copy['price_spread'] = (df_copy['imbalance_price_eur'] - df_copy['spot_price_eur']).astype(float)
            df_copy['price_ratio'] = (df_copy['imbalance_price_eur'] / (df_copy['spot_price_eur'] + 1e-6)).astype(float)
            df_copy['price_deviation_pct'] = (
                        df_copy['price_spread'] / (df_copy['spot_price_eur'] + 1e-6) * 100).astype(float)

        if 'dominating_direction' in df_copy.columns:
            df_copy['direction_up'] = (df_copy['dominating_direction'] == 1).astype(float)
            df_copy['direction_down'] = (df_copy['dominating_direction'] == -1).astype(float)
            df_copy['direction_neutral'] = (df_copy['dominating_direction'] == 0).astype(float)

        # Lag Features
        lag_features = ['imbalance_price_eur', 'spot_price_eur', 'satisfied_demand']
        for lag in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48]:
            for feat in lag_features:
                if feat in df_copy.columns and 'price_area' in df_copy.columns:
                    df_copy[f'lag_{lag}_{feat}'] = df_copy.groupby('price_area')[feat].shift(lag).fillna(0).astype(
                        float)

        # Rolling Statistics
        if 'imbalance_price_eur' in df_copy.columns and 'price_area' in df_copy.columns:
            for window in [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48]:
                df_copy[f'roll_mean_{window}'] = df_copy.groupby('price_area')['imbalance_price_eur'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                ).fillna(0).astype(float)
                df_copy[f'roll_std_{window}'] = df_copy.groupby('price_area')['imbalance_price_eur'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                ).fillna(0).astype(float)

        # Fill missing values
        df_copy = df_copy.fillna(0)

        # Drop non-numeric columns
        cols_to_drop = ['timestamp', 'time_dk', 'TimeUTC', 'TimeDK']
        for col in cols_to_drop:
            if col in df_copy.columns:
                df_copy = df_copy.drop(columns=[col])

        # Get numeric columns
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        if 'imbalance_price_eur' in numeric_cols:
            numeric_cols.remove('imbalance_price_eur')

        if is_training:
            self.feature_names = sorted(numeric_cols)
            self.fixed_feature_columns = self.feature_names
        else:
            if self.fixed_feature_columns is not None:
                for col in self.fixed_feature_columns:
                    if col not in df_copy.columns:
                        df_copy[col] = 0
                numeric_cols = self.fixed_feature_columns

        print(f"✅ Created {len(numeric_cols)} numeric features")
        return df_copy, numeric_cols

    def prepare_features(self, df, is_training=True):
        df_engineered, feature_cols = self.create_features(df, is_training)

        for col in feature_cols:
            if col not in df_engineered.columns:
                df_engineered[col] = 0

        X = df_engineered[feature_cols].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)

        return X, None, df_engineered


# ============================================
# Flask App
# ============================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'


# ============================================
# Future Price Predictor Class
# ============================================

class FuturePricePredictor:
    def __init__(self, model_path='models/DK1_model_5future.pkl'):
        self.model_path = model_path
        self.model_data = None
        self.prediction_history = []
        self.load_model()

    def load_model(self):
        """Load the trained model - tries multiple methods"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        print(f"📂 Loading model from {self.model_path}...")

        # Method 1: Direct load with proper sklearn version
        try:
            # Set up the environment for sklearn 1.5.1 compatibility
            import sklearn
            print(f"   Scikit-learn version: {sklearn.__version__}")

            # Try to load
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)

            print(f"✅ Model loaded successfully!")
            print(f"   Training date: {self.model_data.get('train_date', 'Unknown')}")
            print(f"   Features: {len(self.model_data.get('feature_columns', []))}")
            print(f"   Price Area: {self.model_data.get('price_area', 'DK1')}")
            return

        except Exception as e:
            print(f"⚠️ Direct load failed: {e}")

        # Method 2: Try with custom unpickler
        try:
            print("   Trying with custom unpickler...")
            from pickle import Unpickler

            class CustomUnpickler(Unpickler):
                def find_class(self, module, name):
                    # Handle sklearn._loss._loss for sklearn 1.5.1
                    if module == 'sklearn._loss._loss':
                        try:
                            # Try to import from sklearn._loss first (sklearn 1.6+)
                            import sklearn._loss
                            return getattr(sklearn._loss, name)
                        except (ImportError, AttributeError):
                            try:
                                # Try to import from sklearn._loss._loss (sklearn 1.5.1)
                                import sklearn._loss._loss
                                return getattr(sklearn._loss._loss, name)
                            except (ImportError, AttributeError):
                                # Create a dummy class for unpickling
                                class DummyClass:
                                    def __init__(self, *args, **kwargs):
                                        pass

                                return DummyClass
                    return super().find_class(module, name)

            with open(self.model_path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                self.model_data = unpickler.load()

            print(f"✅ Model loaded with custom unpickler!")
            print(f"   Training date: {self.model_data.get('train_date', 'Unknown')}")
            print(f"   Features: {len(self.model_data.get('feature_columns', []))}")
            print(f"   Price Area: {self.model_data.get('price_area', 'DK1')}")
            return

        except Exception as e:
            print(f"⚠️ Custom unpickler failed: {e}")

        # Method 3: Fallback - extract what we can and use fallback models
        print("   Creating fallback model with real data support...")
        self.create_fallback_model()

    def create_fallback_model(self):
        """Create a fallback model structure - uses REAL data for predictions"""
        self.model_data = {
            'models': {},
            'regime_classifier': OptimizedRegimeClassifier(),
            'probabilistic_forecaster': OptimizedProbabilisticForecaster(),
            'preprocessor': MultiStepDataPreprocessor(),
            'feature_columns': [],
            'price_area': 'DK1',
            'n_steps': 5,
            'step_interval_minutes': 15,
            'train_date': datetime.now().isoformat()
        }
        print("   ✅ Fallback model created - will fetch REAL data from API")

    def get_next_interval_time(self, current_time=None):
        """Get the next 15-minute interval boundary - FIXED for UTC"""
        import pytz

        if current_time is None:
            current_time = datetime.now(pytz.UTC)

        # Ensure UTC
        if current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)
        else:
            current_time = current_time.astimezone(pytz.UTC)

        minutes = current_time.minute
        next_minutes = ((minutes // 15) + 1) * 15

        if next_minutes == 60:
            next_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            next_time = current_time.replace(minute=next_minutes, second=0, microsecond=0)

        return next_time

    def get_latest_data(self, hours_back=6, price_area='DK1'):
        """Fetch REAL data from Energidataservice API"""
        print(f"📡 Fetching REAL data from API for {price_area}...")
        url = "https://api.energidataservice.dk/dataset/ImbalancePrice"

        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours_back)

        full_url = f"{url}?limit=500&timezone=Europe/Copenhagen&sort=TimeUTC+DESC&start={start_date.strftime('%Y-%m-%dT%H:%M:%S')}&end={end_date.strftime('%Y-%m-%dT%H:%M:%S')}&PriceArea={price_area}"

        try:
            response = requests.get(full_url, timeout=30)

            if response.status_code == 400:
                full_url = f"{url}?limit=200&timezone=Europe/Copenhagen&sort=TimeUTC+DESC&PriceArea={price_area}"
                response = requests.get(full_url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if data.get('records') and len(data['records']) > 0:
                    df = pd.DataFrame(data['records'])

                    if 'TimeUTC' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['TimeUTC'])

                    for col in ['Imbalance Price (EUR)', 'ImbalancePriceEUR']:
                        if col in df.columns:
                            df['imbalance_price_eur'] = pd.to_numeric(df[col], errors='coerce')
                            break

                    df = df.dropna(subset=['imbalance_price_eur'])

                    if len(df) > 0 and 'timestamp' in df.columns:
                        df = df.sort_values('timestamp')
                        print(f"✅ Fetched {len(df)} REAL records")
                        print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
                        print(f"   Latest price: {df['imbalance_price_eur'].iloc[-1]:.2f} EUR/MWh")
                        return df

            print("⚠️ No data from API")
            return None

        except Exception as e:
            print(f"❌ API Error: {e}")
            return None

    # ============================================
    # FIXED: get_actual_price_at_time - Proper UTC handling
    # ============================================

    # ============================================
    # FIXED: get_actual_price_at_time - Add missing import
    # ============================================

    def get_actual_price_at_time(self, target_time, price_area='DK1'):
        """
        Fetch REAL actual price at a specific time from API
        FIXED: Added missing time import
        """
        import time

        # Ensure target_time is timezone-aware
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)
        else:
            target_time = target_time.astimezone(timezone.utc)

        # API expects UTC time
        start_time = target_time - timedelta(minutes=15)  # Wider window
        end_time = target_time + timedelta(minutes=15)

        # Format for API (ISO format)
        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%S')
        end_str = end_time.strftime('%Y-%m-%dT%H:%M:%S')

        print(f"🔍 Looking for actual price at: {target_time.strftime('%H:%M')} UTC")

        url = "https://api.energidataservice.dk/dataset/ImbalancePrice"

        # Multiple query attempts with different parameters
        query_params = [
            f"limit=50&timezone=Europe/Copenhagen&sort=TimeUTC+ASC&start={start_str}&end={end_str}&PriceArea={price_area}",
            f"limit=50&timezone=UTC&sort=TimeUTC+ASC&start={start_str}&end={end_str}&PriceArea={price_area}",
            f"limit=100&timezone=Europe/Copenhagen&sort=TimeUTC+DESC&PriceArea={price_area}"
        ]

        for params in query_params:
            full_url = f"{url}?{params}"
            try:
                response = requests.get(full_url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    records = data.get('records', [])

                    if records:
                        df = pd.DataFrame(records)

                        # Parse timestamps
                        if 'TimeUTC' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['TimeUTC'])
                            df['time_diff'] = abs(df['timestamp'] - target_time)

                            # Find closest match within 15 minutes
                            min_diff = df['time_diff'].min()
                            if min_diff.total_seconds() <= 900:  # 15 minutes
                                closest = df.loc[df['time_diff'].idxmin()]

                                # Extract price
                                for col in ['Imbalance Price (EUR)', 'ImbalancePriceEUR']:
                                    if col in df.columns:
                                        price = pd.to_numeric(closest[col], errors='coerce')
                                        if price is not None and not pd.isna(price):
                                            print(
                                                f"✅ Found actual price: {price:.2f} EUR at {closest['timestamp'].strftime('%H:%M')} UTC")
                                            return float(price)
                                break
                    else:
                        print(f"   No records found for params: {params[:60]}...")

            except Exception as e:
                print(f"   API error: {e}")
                continue

            # Wait before retry
            time.sleep(0.5)

        print(f"⚠️ No actual price found for {target_time.strftime('%H:%M')} UTC")
        return None

    # ============================================
    # FIXED: predict_future_prices - Proper time handling
    # ============================================

    def predict_future_prices(self):
        """
        Predict 5 future prices using REAL data from API
        FIXED: Proper time zone handling
        """
        import pytz

        if self.model_data is None:
            raise ValueError("Model not loaded")

        # Always fetch REAL data first
        df = self.get_latest_data(price_area=self.model_data.get('price_area', 'DK1'))

        # If no REAL data, try again with more hours
        if df is None or len(df) < 10:
            print("📡 Trying again with more data...")
            df = self.get_latest_data(hours_back=24, price_area=self.model_data.get('price_area', 'DK1'))

        # If still no data, return None
        if df is None or len(df) < 10:
            print("❌ No real data available. Please check your internet connection.")
            return None

        # Prepare features
        preprocessor = self.model_data.get('preprocessor')
        if preprocessor is None:
            preprocessor = MultiStepDataPreprocessor()
            preprocessor.fixed_feature_columns = self.model_data.get('feature_columns', [])

        X, _, _ = preprocessor.prepare_features(df, is_training=False)

        if len(X) == 0:
            return None

        X_latest = X.iloc[-1:].copy()
        X_latest_np = X_latest.values.astype(float)

        expected_features = len(self.model_data.get('feature_columns', []))
        if X_latest_np.shape[1] != expected_features and expected_features > 0:
            if X_latest_np.shape[1] < expected_features:
                pad = np.zeros((1, expected_features - X_latest_np.shape[1]))
                X_latest_np = np.hstack([X_latest_np, pad])
            else:
                X_latest_np = X_latest_np[:, :expected_features]

        future_prices = []
        individual_predictions = {}

        # Try to use trained models if available
        models_available = False
        for step in range(1, 6):
            step_key = f'p{step}'
            step_models = self.model_data.get('models', {}).get(step_key, {})
            if step_models:
                models_available = True
                break

        if models_available:
            print("✅ Using trained models for prediction")
            for step in range(1, 6):
                step_key = f'p{step}'
                minutes_ahead = step * 15
                step_models = self.model_data.get('models', {}).get(step_key, {})

                if not step_models:
                    continue

                preds = []
                for name, model in step_models.items():
                    try:
                        pred = float(model.predict(X_latest_np)[0])
                        pred = np.clip(pred, -50, 300)
                        preds.append(pred)

                        if name not in individual_predictions:
                            individual_predictions[name] = []
                        individual_predictions[name].append(pred)
                    except Exception as e:
                        continue

                if preds:
                    future_prices.append({
                        'step': step,
                        'minutes_ahead': minutes_ahead,
                        'price': np.median(preds),
                        'std': np.std(preds),
                        'min': np.min(preds),
                        'max': np.max(preds)
                    })
        else:
            # If models not available, use simple trend-based prediction
            print("⚠️ Models not available - using trend-based prediction")
            current_price = df['imbalance_price_eur'].iloc[-1]
            recent_changes = df['imbalance_price_eur'].diff().tail(6).mean()

            for step in range(1, 6):
                pred_price = current_price + recent_changes * step
                pred_price = np.clip(pred_price, -50, 300)
                future_prices.append({
                    'step': step,
                    'minutes_ahead': step * 15,
                    'price': pred_price,
                    'std': 15 + step * 5,
                    'min': pred_price - 20,
                    'max': pred_price + 30
                })

        # Get regime
        regime_classifier = self.model_data.get('regime_classifier')
        if regime_classifier is None:
            regime_classifier = OptimizedRegimeClassifier()

        try:
            regime_result = regime_classifier.predict(X_latest_np, return_proba=True)
            regime = regime_result['regime_names'][0]
            regime_confidence = float(regime_result['confidence'][0])
            regime_probs = regime_result['probabilities'][0].tolist()
        except Exception as e:
            regime = 'Normal'
            regime_confidence = 0.75
            regime_probs = [0.75, 0.10, 0.10, 0.05]

        current_price = df['imbalance_price_eur'].iloc[-1] if len(df) > 0 else 80
        current_time = datetime.now(pytz.UTC)  # Use UTC time

        # Get next 15-minute interval in UTC
        next_interval = self.get_next_interval_time(current_time)
        future_times = [next_interval + timedelta(minutes=(i) * 15) for i in range(5)]

        # Try to get actual prices (for validation) - only for times that have passed
        actual_prices = {}
        current_utc = datetime.now(pytz.UTC)

        print("\n📡 Checking for actual prices:")
        for i, future_time in enumerate(future_times):
            # Only check if the time has passed (allow 5 minute buffer)
            if future_time < current_utc - timedelta(minutes=5):
                actual = self.get_actual_price_at_time(future_time, self.model_data.get('price_area', 'DK1'))
                if actual is not None:
                    actual_prices[f'Q{i + 1}'] = actual
                    print(f"   Q{i + 1} ({future_time.strftime('%H:%M')} UTC): {actual:.2f} EUR/MWh")
                else:
                    print(f"   Q{i + 1} ({future_time.strftime('%H:%M')} UTC): Not available")
            else:
                print(f"   Q{i + 1} ({future_time.strftime('%H:%M')} UTC): Future time (not yet available)")

        result = {
            'current_price': current_price,
            'current_time': current_time.isoformat(),
            'next_interval_time': next_interval.isoformat(),
            'regime': regime,
            'regime_confidence': regime_confidence,
            'regime_probabilities': regime_probs,
            'future_prices': future_prices,
            'future_times': [t.isoformat() for t in future_times],
            'individual_predictions': individual_predictions,
            'actual_prices': actual_prices,
            'validation_available': len(actual_prices) > 0
        }

        self.prediction_history.append(result)
        return result


# ============================================
# Initialize Predictor
# ============================================

predictor = None
try:
    print("\n" + "=" * 60)
    print("🚀 Initializing Imbalance Price Predictor")
    print("=" * 60)
    predictor = FuturePricePredictor('models/DK1_model_5future.pkl')
    if predictor.model_data:
        print("✅ Predictor initialized successfully with REAL data support")
    else:
        print("⚠️ Predictor initialized with fallback - will still use REAL data")
except Exception as e:
    print(f"❌ Error initializing predictor: {e}")
    predictor = None


# ============================================
# Flask Routes
# ============================================

@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/predict', methods=['GET'])
def api_predict():
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500

    result = predictor.predict_future_prices()

    if result is None:
        return jsonify({'error': 'Prediction failed - no real data available'}), 500

    return jsonify(result)


# ============================================
# FIXED: api_validate route
# ============================================

@app.route('/api/validate', methods=['GET'])
def api_validate():
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500

    if not predictor.prediction_history:
        return jsonify({'error': 'No predictions available'}), 400

    latest = predictor.prediction_history[-1]

    future_times = []
    for t in latest['future_times']:
        try:
            future_times.append(datetime.fromisoformat(t))
        except:
            future_times.append(datetime.strptime(t.replace('Z', '+00:00'), '%Y-%m-%dT%H:%M:%S%z'))

    actual_prices = {}
    errors = {}

    print("\n📡 Validating predictions against actual prices:")

    for i, future_time in enumerate(future_times):
        try:
            actual = predictor.get_actual_price_at_time(future_time, predictor.model_data.get('price_area', 'DK1'))
            if actual is not None:
                actual_prices[f'Q{i + 1}'] = actual
                print(f"   Q{i + 1}: Actual = {actual:.2f} EUR/MWh")
            else:
                print(f"   Q{i + 1}: Not available")
        except Exception as e:
            print(f"   Q{i + 1}: Error - {e}")
            continue

    latest['actual_prices'] = actual_prices
    latest['validation_available'] = len(actual_prices) > 0

    # Calculate errors for validated predictions
    for q, actual in actual_prices.items():
        q_index = int(q[1:]) - 1
        if q_index < len(latest['future_prices']):
            pred = latest['future_prices'][q_index]['price']
            error = abs(pred - actual)
            errors[q] = {
                'predicted': float(pred),
                'actual': float(actual),
                'error': float(error),
                'error_pct': float((error / (actual + 0.01)) * 100)
            }

    return jsonify({
        'actual_prices': actual_prices,
        'errors': errors,
        'validation_available': len(actual_prices) > 0,
        'validated_count': len(actual_prices)
    })


@app.route('/api/history', methods=['GET'])
def api_history():
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500

    history = []
    for pred in predictor.prediction_history[-10:]:
        history.append({
            'time': pred['current_time'],
            'regime': pred['regime'],
            'regime_confidence': pred['regime_confidence'],
            'future_prices': [p['price'] for p in pred['future_prices']],
            'validation_available': pred['validation_available']
        })

    return jsonify({'history': history})


@app.route('/api/plot', methods=['GET'])
def api_plot():
    if predictor is None or not predictor.prediction_history:
        return jsonify({'error': 'No data available'}), 400

    latest = predictor.prediction_history[-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('IMBALANCE PRICE PREDICTION DASHBOARD',
                 fontsize=18, fontweight='bold', color='#2C3E50')

    # Plot 1: Prediction vs Actual
    ax1 = axes[0, 0]
    future_times = [datetime.fromisoformat(t) for t in latest['future_times']]
    predicted_prices = [p['price'] for p in latest['future_prices']]
    lower_bounds = [p['min'] for p in latest['future_prices']]
    upper_bounds = [p['max'] for p in latest['future_prices']]

    ax1.plot(future_times, predicted_prices, 'o-', color='#2E86AB',
             linewidth=2, markersize=8, label='Predicted Price')
    ax1.fill_between(future_times, lower_bounds, upper_bounds,
                     color='#2E86AB', alpha=0.2, label='Prediction Range')

    # Add actual prices
    actual_prices = latest.get('actual_prices', {})
    for q, actual in actual_prices.items():
        q_index = int(q[1:]) - 1
        if q_index < len(future_times):
            ax1.scatter(future_times[q_index], actual, color='#E74C3C',
                        s=150, zorder=5, marker='s', edgecolor='black', linewidth=2)
            pred = predicted_prices[q_index]
            error = abs(actual - pred)
            color = '#27AE60' if error < 20 else '#F39C12' if error < 50 else '#E74C3C'
            ax1.plot([future_times[q_index], future_times[q_index]],
                     [min(actual, pred), max(actual, pred)],
                     color=color, alpha=0.7, linewidth=2)
            ax1.text(future_times[q_index], pred + 3, f'Err: {error:.1f}',
                     ha='center', va='bottom', fontsize=8)

    ax1.axhline(y=latest['current_price'], color='#95A5A6', linestyle='--',
                alpha=0.5, label=f'Current: {latest["current_price"]:.1f} EUR')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Price (EUR/MWh)', fontsize=11)
    ax1.set_title('Price Prediction with Intervals', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Plot 2: Model Agreement
    ax2 = axes[0, 1]
    individual_preds = latest.get('individual_predictions', {})
    if individual_preds:
        models = list(individual_preds.keys())
        quarters = [f'Q{i + 1}' for i in range(len(latest['future_prices']))]
        data = []
        for model in models:
            data.append(individual_preds[model])

        im = ax2.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=300)
        ax2.set_xticks(range(len(quarters)))
        ax2.set_yticks(range(len(models)))
        ax2.set_xticklabels(quarters, fontsize=9)
        ax2.set_yticklabels(models, fontsize=9)
        ax2.set_xlabel('Quarter', fontsize=11)
        ax2.set_ylabel('Model', fontsize=11)
        ax2.set_title('Model Prediction Agreement', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Price (EUR/MWh)')

        for i in range(len(models)):
            for j in range(len(quarters)):
                val = data[i][j] if j < len(data[i]) else 0
                color = 'white' if val > 150 else 'black'
                ax2.text(j, i, f'{val:.0f}', ha="center", va="center",
                         color=color, fontsize=8)

    # Plot 3: Regime Classification
    ax3 = axes[1, 0]
    regimes = ['Normal', 'Supply_Constrained', 'Demand_Constrained', 'Volatile']
    probs = latest.get('regime_probabilities', [0.5, 0.2, 0.2, 0.1])
    colors = ['#2ECC71', '#E74C3C', '#3498DB', '#F39C12']
    bars = ax3.bar(regimes, probs, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Probability', fontsize=11)
    ax3.set_title(f'Regime Classification: {latest["regime"]} ({latest["regime_confidence"]:.1%} confidence)',
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, prob in zip(bars, probs):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                 f'{prob:.1%}', ha='center', va='bottom', fontsize=9)

    # Plot 4: Error Analysis
    ax4 = axes[1, 1]
    if latest.get('validation_available', False):
        actual_prices = latest.get('actual_prices', {})
        errors = []
        q_names = []

        for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4', 'Q5']):
            if q in actual_prices:
                pred = latest['future_prices'][i]['price']
                actual = actual_prices[q]
                errors.append(abs(pred - actual))
                q_names.append(q)

        if errors:
            colors = ['#27AE60' if e < 20 else '#F39C12' if e < 50 else '#E74C3C' for e in errors]
            bars = ax4.bar(q_names, errors, color=colors, edgecolor='black', linewidth=1.5)

            for bar, err in zip(bars, errors):
                ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                         f'{err:.1f}', ha='center', va='bottom', fontsize=10)

            ax4.set_xlabel('Quarter', fontsize=11)
            ax4.set_ylabel('Absolute Error (EUR/MWh)', fontsize=11)
            ax4.set_title('Prediction Accuracy by Quarter', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.axhline(y=20, color='#27AE60', linestyle='--', alpha=0.5, label='Good (20 EUR)')
            ax4.axhline(y=50, color='#F39C12', linestyle='--', alpha=0.5, label='Acceptable (50 EUR)')
            ax4.legend(fontsize=9)
    else:
        ax4.text(0.5, 0.5, '⏳ Waiting for actual prices...\nCheck back after 15-30 minutes',
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Validation Pending', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return jsonify({'image': plot_data})


# ============================================
# Run Flask App
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Starting Flask Dashboard")
    print("=" * 60)
    print(f"📂 Model path: models/DK1_model_5future.pkl")
    print(f"📊 Using REAL data from Energidataservice API")
    print(f"🌐 Server: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
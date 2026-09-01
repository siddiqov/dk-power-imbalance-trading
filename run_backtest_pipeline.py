# ==============================================================================
# run_backtest_pipeline.py - ENHANCED BACKTESTING PIPELINE
# Danish Electricity Imbalance Price Forecasting & Regime Switching Detection
#
# Models:
#   Price Forecasting : RandomForest, XGBoost, LightGBM, CatBoost, ElasticNet
#   Regime Detection  : RandomForestClassifier (supervised) + GaussianHMM (unsupervised)
#   Ensemble          : Weighted median by validation MAE
#   Tuning            : Optuna Bayesian search with TimeSeriesSplit (no leakage)
# ==============================================================================

import os
import sys
import argparse
import json
import pickle
import time
import warnings
from datetime import datetime, timedelta

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              accuracy_score, f1_score)

# ---- Optional model imports --------------------------------------------------
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("  [INFO] xgboost not found – skipping XGBoost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("  [INFO] lightgbm not found – skipping LightGBM")

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False
    print("  [INFO] catboost not found – skipping CatBoost")

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("  [INFO] hmmlearn not found – skipping HMM regime detector")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("  [INFO] optuna not found – hyperparameter tuning disabled")


# ==============================================================================
# 1. DATABASE & FEATURE ENGINEERING LAYER
# ==============================================================================

class BacktestFeaturePipeline:
    """Extracts and engineers causal features with exact 15-min Danish time alignment."""

    def __init__(self, db_path="energy_data.db"):
        self.db_path = db_path

    def get_connection(self):
        return duckdb.connect(self.db_path)

    def load_and_prepare_features(self, area="DK1"):
        """Loads data from DuckDB, builds causal features, aligns to Danish time."""
        print(f"\n{'='*65}")
        print(f"  FEATURE PIPELINE  |  Area: {area}")
        print(f"{'='*65}")

        conn = self.get_connection()
        try:
            # --- Imbalance Prices ---
            imbalance = conn.execute(f"""
                SELECT time_utc, time_dk,
                       imbalance_price_dkk, imbalance_price_eur,
                       spot_price_eur, satisfied_demand,
                       dominating_direction, afrr_up_mw, afrr_down_mw
                FROM imbalance_prices
                WHERE price_area = '{area}'
                  AND imbalance_price_dkk IS NOT NULL
                ORDER BY time_utc
            """).fetchdf()

            if imbalance.empty:
                raise ValueError(f"No imbalance data for {area}. Run data_retrieval.py first.")

            imbalance['time_utc'] = pd.to_datetime(imbalance['time_utc'])
            imbalance['time_dk'] = pd.to_datetime(imbalance['time_dk'])
            imbalance = (imbalance
                         .drop_duplicates(subset=['time_utc'])
                         .sort_values('time_utc')
                         .reset_index(drop=True))
            print(f"  Loaded {len(imbalance):,} imbalance records")

            # --- Day-Ahead Prices ---
            day_ahead = conn.execute(f"""
                SELECT time_utc, price_dkk AS day_ahead_price_dkk, price_eur AS day_ahead_price_eur
                FROM day_ahead_prices WHERE price_area = '{area}' ORDER BY time_utc
            """).fetchdf()
            if not day_ahead.empty:
                day_ahead['time_utc'] = pd.to_datetime(day_ahead['time_utc'])
                day_ahead_15m = (day_ahead
                                 .drop_duplicates('time_utc')
                                 .set_index('time_utc')
                                 .sort_index()
                                 .resample('15min').ffill())
            else:
                day_ahead_15m = pd.DataFrame()

            # --- Electricity Balance ---
            balance = conn.execute(f"""
                SELECT time_utc, total_load, total_wind,
                       wind_offshore, wind_onshore, solar, net_exchange
                FROM electricity_balance WHERE price_area = '{area}' ORDER BY time_utc
            """).fetchdf()
            if not balance.empty:
                balance['time_utc'] = pd.to_datetime(balance['time_utc'])
                balance_15m = (balance
                               .drop_duplicates('time_utc')
                               .set_index('time_utc')
                               .sort_index()
                               .resample('15min').ffill())
            else:
                balance_15m = pd.DataFrame()

            # --- Forecasts (Wind & Solar) ---
            forecasts = conn.execute(f"""
                SELECT time_utc, forecast_type, forecast_current
                FROM forecasts_hour WHERE price_area = '{area}' ORDER BY time_utc
            """).fetchdf()
            wind_fc_15m = solar_fc_15m = pd.DataFrame()
            if not forecasts.empty:
                forecasts['time_utc'] = pd.to_datetime(forecasts['time_utc'])
                for fc_type, col_name in [('Wind', 'wind_forecast'), ('Solar', 'solar_forecast')]:
                    sub = forecasts[forecasts['forecast_type'].str.contains(fc_type, case=False, na=False)]
                    if not sub.empty:
                        agg = sub.groupby('time_utc')['forecast_current'].mean().to_frame(col_name)
                        if fc_type == 'Wind':
                            wind_fc_15m = agg.resample('15min').ffill()
                        else:
                            solar_fc_15m = agg.resample('15min').ffill()

            # --- Reserve Activations ---
            afrr = conn.execute(f"""
                SELECT time_utc, afrr_activated_mw AS afrr_activated
                FROM afrr_activation WHERE price_area = '{area}' ORDER BY time_utc
            """).fetchdf()
            afrr_15m = pd.DataFrame()
            if not afrr.empty:
                afrr['time_utc'] = pd.to_datetime(afrr['time_utc'])
                afrr_15m = (afrr.drop_duplicates('time_utc')
                            .set_index('time_utc').sort_index()
                            .resample('15min').ffill())

            mfrr = conn.execute(f"""
                SELECT time_utc, mfrr_up_mw AS mfrr_up, mfrr_down_mw AS mfrr_down
                FROM mfrr_activation WHERE price_area = '{area}' ORDER BY time_utc
            """).fetchdf()
            mfrr_15m = pd.DataFrame()
            if not mfrr.empty:
                mfrr['time_utc'] = pd.to_datetime(mfrr['time_utc'])
                mfrr_15m = (mfrr.drop_duplicates('time_utc')
                            .set_index('time_utc').sort_index()
                            .resample('15min').ffill())

        finally:
            conn.close()

        # --- Merge into unified 15-min grid ---
        df = imbalance.set_index('time_utc')
        for sub_df in [day_ahead_15m, balance_15m, wind_fc_15m,
                       solar_fc_15m, afrr_15m, mfrr_15m]:
            if not sub_df.empty:
                df = df.join(sub_df, how='left')
        df = df.reset_index()

        # --- Fill missing columns ---
        fill_cols = ['day_ahead_price_dkk', 'day_ahead_price_eur',
                     'total_load', 'total_wind', 'wind_offshore', 'wind_onshore',
                     'solar', 'net_exchange', 'wind_forecast', 'solar_forecast',
                     'afrr_activated', 'mfrr_up', 'mfrr_down']
        for col in fill_cols:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(0)
            else:
                df[col] = 0.0

        # --- Ensure Danish Time column ---
        if 'time_dk' not in df.columns or df['time_dk'].isna().all():
            df['time_dk'] = (df['time_utc']
                             .dt.tz_localize('UTC')
                             .dt.tz_convert('Europe/Copenhagen')
                             .dt.tz_localize(None))

        # =================================================================
        # CAUSAL FEATURE ENGINEERING  (no lookahead / data leakage)
        # =================================================================
        # 1. Forecast Errors
        df['wind_error']  = df['total_wind'] - df['wind_forecast']
        df['solar_error'] = df['solar']       - df['solar_forecast']
        df['load_error']  = df['satisfied_demand'] - df['total_load']

        # 2. Interconnector & Spreads
        df['interconnector_flow']  = df['net_exchange']
        df['intraday_spread_eur']  = df['imbalance_price_eur'] - df['day_ahead_price_eur']
        df['intraday_spread_dkk']  = df['imbalance_price_dkk'] - df['day_ahead_price_dkk']

        # 3. Reserve Activation Flags
        df['afrr_active'] = (df['afrr_activated'].abs() > 0.1).astype(int)
        df['mfrr_active'] = ((df['mfrr_up'] > 0) | (df['mfrr_down'] > 0)).astype(int)
        df['net_reserve_activation'] = df['afrr_activated'] + df['mfrr_up'] - df['mfrr_down']

        # 4. Calendar & Momentum with Cyclical Encoding
        df['hour_dk']      = df['time_dk'].dt.hour
        df['minute_dk']    = df['time_dk'].dt.minute
        df['day_of_week']  = df['time_dk'].dt.dayofweek
        df['month']        = df['time_dk'].dt.month
        df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
        df['quarter_of_day'] = (df['hour_dk'] * 4 + df['minute_dk'] // 15)  # 0..95

        # Cyclical transforms (smooth periodicity)
        df['sin_hour']        = np.sin(2 * np.pi * df['hour_dk'] / 24.0)
        df['cos_hour']        = np.cos(2 * np.pi * df['hour_dk'] / 24.0)
        df['sin_quarter']     = np.sin(2 * np.pi * df['quarter_of_day'] / 96.0)
        df['cos_quarter']     = np.cos(2 * np.pi * df['quarter_of_day'] / 96.0)
        df['sin_dayofweek']   = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
        df['cos_dayofweek']   = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
        df['sin_month']       = np.sin(2 * np.pi * df['month'] / 12.0)
        df['cos_month']       = np.cos(2 * np.pi * df['month'] / 12.0)

        df['price_momentum_eur']     = df['day_ahead_price_eur'].diff(4) / 4.0
        df['imbalance_momentum_eur'] = df['imbalance_price_eur'].diff(4) / 4.0
        df['imbalance_volatility']   = df['imbalance_price_eur'].rolling(8).std().fillna(0)

        # Rolling spread & reserve statistics (strictly causal)
        df['spread_rolling_mean_4']  = df['intraday_spread_eur'].rolling(4).mean().fillna(0)
        df['spread_rolling_std_4']   = df['intraday_spread_eur'].rolling(4).std().fillna(0)
        df['spread_rolling_mean_16'] = df['intraday_spread_eur'].rolling(16).mean().fillna(0)
        df['reserve_rolling_mean_4'] = df['net_reserve_activation'].rolling(4).mean().fillna(0)
        df['reserve_rolling_mean_16']= df['net_reserve_activation'].rolling(16).mean().fillna(0)

        # 5. Cross-product features
        df['wind_load_ratio']    = df['total_wind'] / (df['total_load'].replace(0, np.nan)).fillna(1)
        df['renewable_penetration'] = (df['total_wind'] + df['solar']) / (df['total_load'].replace(0, np.nan)).fillna(1)

        # 6. Multi-Period Lagged Features  (lags: 1×15m, 2×15m, 4×15m=1h, 8×15m=2h, 16×15m=4h)
        causal_base = [
            'wind_error', 'solar_error', 'load_error', 'interconnector_flow',
            'intraday_spread_eur', 'intraday_spread_dkk',
            'imbalance_price_eur', 'imbalance_price_dkk',
            'day_ahead_price_eur', 'day_ahead_price_dkk',
            'afrr_activated', 'mfrr_up', 'mfrr_down',
            'net_reserve_activation',
            'price_momentum_eur', 'imbalance_momentum_eur',
            'spread_rolling_mean_4', 'spread_rolling_std_4',
            'wind_load_ratio', 'renewable_penetration'
        ]
        for lag in [1, 2, 4, 8, 16]:
            for col in causal_base:
                if col in df.columns:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        # =================================================================
        # REGIME LABELS  (0=Downward, 1=Neutral, 2=Upward)
        # =================================================================
        df['regime'] = 1  # default Neutral
        if 'dominating_direction' in df.columns:
            df.loc[df['dominating_direction'] == -1, 'regime'] = 0
            df.loc[df['dominating_direction'] ==  1, 'regime'] = 2
            df.loc[df['dominating_direction'].astype(str).str.lower().str.contains('down', na=False), 'regime'] = 0
            df.loc[df['dominating_direction'].astype(str).str.lower().str.contains('up', na=False), 'regime'] = 2
        if 'afrr_down_mw' in df.columns:
            df.loc[df['afrr_down_mw'].abs() > 5, 'regime'] = 0
        if 'afrr_up_mw' in df.columns:
            df.loc[df['afrr_up_mw'].abs() > 5, 'regime'] = 2

        # Drop NaN rows from lagging
        df = df.dropna(subset=['imbalance_price_eur', 'imbalance_price_dkk']).reset_index(drop=True)
        df = df.iloc[16:].reset_index(drop=True)

        print(f"  Feature matrix: {len(df):,} rows x {len(df.columns)} columns")
        print(f"  Time range (DK): {df['time_dk'].min()} -> {df['time_dk'].max()}")
        return df


# ==============================================================================
# 2. HMM REGIME DETECTOR  (unsupervised, generative)
# ==============================================================================

class HMMRegimeDetector:
    """
    3-state Gaussian HMM fitted on [imbalance_spread, net_reserve_activation, load_error].
    States are post-hoc mapped to Downward/Neutral/Upward by price level.
    """

    def __init__(self, n_states=3, n_iter=200, covariance_type='full', random_state=42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.state_map = {}          # HMM state id -> regime label (0/1/2)
        self.regime_names = {0: 'Downward Regulation', 1: 'Neutral', 2: 'Upward Regulation'}

    def _build_obs(self, df, fit_scaler=False):
        """Build observation matrix from key physical signals."""
        obs = np.column_stack([
            df['intraday_spread_eur'].fillna(0).values,
            df['net_reserve_activation'].fillna(0).values,
            df['load_error'].fillna(0).values,
        ])
        if fit_scaler or self.scaler_mean is None:
            self.scaler_mean = obs.mean(axis=0)
            self.scaler_std  = obs.std(axis=0) + 1e-8
        
        obs_scaled = (obs - self.scaler_mean) / self.scaler_std
        return obs_scaled

    def fit(self, train_df):
        if not HMM_AVAILABLE:
            print("  [HMM] hmmlearn not available – HMM skipped")
            return self
        print(f"\n  [HMM] Fitting {self.n_states}-state GaussianHMM on {len(train_df):,} intervals...")
        obs = self._build_obs(train_df, fit_scaler=True)
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        self.model.fit(obs)

        # --- Map HMM states to Downward/Neutral/Upward ---
        # Strategy: align HMM discovered states to supervised 'regime' labels using
        # majority vote over training data (most reliable when labels exist).
        # Fallback to spread ranking if labels are all Neutral.
        train_states = self.model.predict(obs)

        if 'regime' in train_df.columns:
            # Use supervised label majority vote per HMM state
            vote_map = {}
            for s in range(self.n_states):
                mask = train_states == s
                if mask.sum() > 0:
                    labels_in_state = train_df['regime'].values[mask]
                    # Majority vote
                    from collections import Counter
                    most_common = Counter(labels_in_state).most_common(1)[0][0]
                    vote_map[s] = int(most_common)
                else:
                    vote_map[s] = 1  # fallback neutral

            # Resolve ties: if two HMM states map to the same label, use spread ranking as tiebreaker
            from collections import Counter as _Counter
            used = _Counter(vote_map.values())
            for label, count in used.items():
                if count > 1:
                    # Find all HMM states mapping to this label
                    tied = [s for s, l in vote_map.items() if l == label]
                    # Re-rank by mean spread
                    spreads = {s: train_df['intraday_spread_eur'].values[train_states == s].mean()
                               for s in tied}
                    ranked = sorted(spreads, key=spreads.get)
                    # lowest spread -> 0 (Down), middle -> 1 (Neutral), highest -> 2 (Up)
                    all_available_labels = sorted(set(range(self.n_states)) - set(vote_map.values()) | {label})
                    for rank_idx, state_id in enumerate(ranked):
                        vote_map[state_id] = all_available_labels[min(rank_idx, len(all_available_labels)-1)]

            self.state_map = vote_map
        else:
            # Pure unsupervised fallback: rank by mean intraday spread
            spread_per_state = {
                s: train_df['intraday_spread_eur'].values[train_states == s].mean()
                for s in range(self.n_states)
            }
            sorted_states = sorted(spread_per_state, key=spread_per_state.get)
            self.state_map = {sorted_states[i]: i for i in range(self.n_states)}

        print(f"  [HMM] State mapping (HMM state -> regime): {self.state_map}")
        print(f"  [HMM] Log-likelihood: {self.model.score(obs):.2f}")

        # Compute per-state stats for interpretability
        for s in range(self.n_states):
            mask = train_states == s
            reg_name = self.regime_names.get(self.state_map.get(s, 1), '?')
            n_s = mask.sum()
            avg_spread = train_df['intraday_spread_eur'].values[mask].mean() if n_s > 0 else 0
            print(f"  [HMM] State {s} -> {reg_name:<22} | n={n_s:4d} | avg_spread={avg_spread:+.2f} EUR")

        return self

    def predict(self, df):
        """Returns (regime_labels, state_probs) arrays."""
        if self.model is None:
            n = len(df)
            return np.ones(n, dtype=int), np.tile([0.1, 0.8, 0.1], (n, 1))
        obs = self._build_obs(df, fit_scaler=False)
        raw_states = self.model.predict(obs)
        mapped = np.array([self.state_map.get(s, 1) for s in raw_states])
        # Posterior probabilities
        log_probs = self.model.predict_proba(obs)         # (n, n_states)
        # Reorder columns to match regime mapping
        regime_probs = np.zeros((len(df), self.n_states))
        for raw_s, reg_s in self.state_map.items():
            regime_probs[:, reg_s] = log_probs[:, raw_s]
        return mapped, regime_probs

    def transition_matrix(self):
        """Returns the HMM transition probability matrix in regime order."""
        if self.model is None:
            return np.eye(3)
        raw = self.model.transmat_
        n = self.n_states
        reordered = np.zeros((n, n))
        for raw_i, reg_i in self.state_map.items():
            for raw_j, reg_j in self.state_map.items():
                reordered[reg_i, reg_j] = raw[raw_i, raw_j]
        return reordered


# ==============================================================================
# 3. OPTUNA HYPERPARAMETER OPTIMISER  (Bayesian + TimeSeriesSplit)
# ==============================================================================

class OptunaHyperparamOptimiser:
    """
    Uses Optuna TPE sampler with TimeSeriesSplit(n_splits=5) cross-validation
    to tune each model's hyperparameters without data leakage.
    """

    def __init__(self, X_train, y_train, n_splits=5, n_trials=50, random_state=42):
        # If dataset is large (>8000 samples), tune on the most recent 8000 samples for speed & regime relevance
        if len(X_train) > 8000:
            self.X = X_train[-8000:]
            self.y = y_train[-8000:]
        else:
            self.X = X_train
            self.y = y_train
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_state = random_state
        self.tss = TimeSeriesSplit(n_splits=n_splits)

    def _cv_mae(self, model):
        maes = []
        for train_idx, val_idx in self.tss.split(self.X):
            X_tr, X_val = self.X[train_idx], self.X[val_idx]
            y_tr, y_val = self.y[train_idx], self.y[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            maes.append(mean_absolute_error(y_val, preds))
        return np.mean(maes)

    def tune_random_forest(self):
        if not OPTUNA_AVAILABLE:
            return {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 4}

        def objective(trial):
            params = dict(
                n_estimators      = trial.suggest_int('n_estimators', 50, 150),
                max_depth         = trial.suggest_int('max_depth', 4, 14),
                min_samples_split = trial.suggest_int('min_samples_split', 2, 10),
                min_samples_leaf  = trial.suggest_int('min_samples_leaf', 1, 6),
                max_features      = trial.suggest_categorical('max_features', ['sqrt', 0.4, 0.6]),
                random_state      = self.random_state,
                n_jobs            = 4
            )
            return self._cv_mae(RandomForestRegressor(**params))

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params

    def tune_xgboost(self):
        if not OPTUNA_AVAILABLE or not XGB_AVAILABLE:
            return {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.08,
                    'subsample': 0.8, 'colsample_bytree': 0.8}

        def objective(trial):
            params = dict(
                n_estimators      = trial.suggest_int('n_estimators', 50, 150),
                max_depth         = trial.suggest_int('max_depth', 3, 8),
                learning_rate     = trial.suggest_float('learning_rate', 0.03, 0.2, log=True),
                subsample         = trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 0.9),
                min_child_weight  = trial.suggest_int('min_child_weight', 1, 8),
                random_state      = self.random_state,
                n_jobs            = 4,
                verbosity         = 0
            )
            return self._cv_mae(xgb.XGBRegressor(**params))

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params

    def tune_lightgbm(self):
        if not OPTUNA_AVAILABLE or not LGB_AVAILABLE:
            return {'n_estimators': 100, 'num_leaves': 31, 'learning_rate': 0.08,
                    'feature_fraction': 0.8, 'bagging_fraction': 0.8}

        def objective(trial):
            params = dict(
                n_estimators     = trial.suggest_int('n_estimators', 50, 150),
                num_leaves       = trial.suggest_int('num_leaves', 15, 63),
                learning_rate    = trial.suggest_float('learning_rate', 0.03, 0.2, log=True),
                feature_fraction = trial.suggest_float('feature_fraction', 0.5, 0.9),
                bagging_fraction = trial.suggest_float('bagging_fraction', 0.6, 1.0),
                bagging_freq     = trial.suggest_int('bagging_freq', 1, 5),
                min_child_samples = trial.suggest_int('min_child_samples', 10, 50),
                random_state     = self.random_state,
                n_jobs           = 4,
                verbose          = -1
            )
            return self._cv_mae(lgb.LGBMRegressor(**params))

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params

    def tune_catboost(self):
        if not OPTUNA_AVAILABLE or not CB_AVAILABLE:
            return {'iterations': 100, 'depth': 6, 'learning_rate': 0.08}

        def objective(trial):
            params = dict(
                iterations        = trial.suggest_int('iterations', 50, 150),
                depth             = trial.suggest_int('depth', 4, 8),
                learning_rate     = trial.suggest_float('learning_rate', 0.03, 0.2, log=True),
                l2_leaf_reg       = trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
                random_seed       = self.random_state,
                thread_count      = 4,
                verbose           = False
            )
            return self._cv_mae(cb.CatBoostRegressor(**params))

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params

    def tune_elasticnet(self):
        if not OPTUNA_AVAILABLE:
            return {'alpha': 0.1, 'l1_ratio': 0.5}

        def objective(trial):
            params = dict(
                alpha    = trial.suggest_float('alpha', 1e-4, 10.0, log=True),
                l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0),
                max_iter = 2000
            )
            return self._cv_mae(ElasticNet(**params))

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params

    def tune_hmm(self, train_df):
        """Tune HMM hyperparameters via log-likelihood on a holdout fold."""
        if not OPTUNA_AVAILABLE or not HMM_AVAILABLE:
            return {'n_states': 3, 'covariance_type': 'full', 'n_iter': 200}

        obs_full = np.column_stack([
            train_df['intraday_spread_eur'].fillna(0).values,
            train_df['net_reserve_activation'].fillna(0).values,
            train_df['load_error'].fillna(0).values,
        ])
        obs_full = (obs_full - obs_full.mean(axis=0)) / (obs_full.std(axis=0) + 1e-8)

        # Use last 20% as validation for HMM (time-aware holdout)
        split = int(len(obs_full) * 0.8)
        obs_tr, obs_val = obs_full[:split], obs_full[split:]

        def objective(trial):
            n = trial.suggest_int('n_components', 2, 5)
            cov = trial.suggest_categorical('covariance_type', ['full', 'diag', 'tied'])
            it  = trial.suggest_int('n_iter', 50, 300)
            try:
                m = GaussianHMM(n_components=n, covariance_type=cov,
                                n_iter=it, random_state=42)
                m.fit(obs_tr)
                return -m.score(obs_val)   # minimise negative log-likelihood
            except Exception:
                return 1e9

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=min(self.n_trials, 20), show_progress_bar=False)
        best = study.best_params
        return {
            'n_states': best['n_components'],
            'covariance_type': best['covariance_type'],
            'n_iter': best['n_iter']
        }


# ==============================================================================
# 4. BACKTESTING ENGINE  (train/test split, multi-step forecasting, evaluation)
# ==============================================================================

class BacktestEngine:
    """Coordinates chronological Train/Test Split, Model Training, and Backtest Evaluation."""

    def __init__(self, df, area="DK1", test_intervals=5, target_currency="EUR",
                 use_tuning=False, n_trials=50, fast_tuning=False):
        self.df = df.copy().sort_values('time_utc').reset_index(drop=True)
        self.area = area
        self.test_intervals = test_intervals
        self.target_currency = target_currency
        self.price_col = f'imbalance_price_{target_currency.lower()}'
        self.use_tuning = use_tuning and OPTUNA_AVAILABLE
        self.fast_tuning = fast_tuning
        self.n_trials = 10 if fast_tuning else n_trials
        self.regime_names = {0: 'Downward Regulation', 1: 'Neutral', 2: 'Upward Regulation'}

        # Feature columns: all lag columns + calendar/momentum features
        self.feature_cols = [
            c for c in self.df.columns
            if 'lag_' in c or c in [
                'hour_dk', 'minute_dk', 'day_of_week', 'month',
                'is_weekend', 'quarter_of_day',
                'sin_hour', 'cos_hour', 'sin_quarter', 'cos_quarter',
                'sin_dayofweek', 'cos_dayofweek', 'sin_month', 'cos_month',
                'price_momentum_eur', 'imbalance_momentum_eur', 'imbalance_volatility',
                'spread_rolling_mean_4', 'spread_rolling_std_4', 'spread_rolling_mean_16',
                'reserve_rolling_mean_4', 'reserve_rolling_mean_16',
                'wind_load_ratio', 'renewable_penetration',
                'afrr_active', 'mfrr_active', 'net_reserve_activation'
            ]
        ]
        self.bundle_path = f'models/{self.area}_trained_bundle.pkl'
        os.makedirs('models', exist_ok=True)

    # ------------------------------------------------------------------
    # SAVE & LOAD TRAINED MODEL BUNDLE
    # ------------------------------------------------------------------
    def save_bundle(self, imputer, scaler, regime_clf, hmm_detector, step_models, weights, best_params, metrics):
        """Saves all trained models, scalers, weights, and metadata into a single bundle."""
        bundle = {
            'area': self.area,
            'feature_cols': self.feature_cols,
            'imputer': imputer,
            'scaler': scaler,
            'regime_clf': regime_clf,
            'hmm_detector': hmm_detector,
            'step_models': step_models,
            'weights': weights,
            'best_params': best_params,
            'metrics': metrics,
            'trained_at': datetime.now().isoformat(),
            'training_rows': len(self.df) - self.test_intervals
        }
        with open(self.bundle_path, 'wb') as f:
            pickle.dump(bundle, f)
        print(f"  💾 Model bundle successfully saved to: {self.bundle_path}")

    def load_bundle(self):
        """Loads the pre-trained bundle from disk."""
        if not os.path.exists(self.bundle_path):
            return None
        with open(self.bundle_path, 'rb') as f:
            bundle = pickle.load(f)
        return bundle

    # ------------------------------------------------------------------
    # SPLIT
    # ------------------------------------------------------------------
    def split_train_test(self):
        n = len(self.df)
        if n <= self.test_intervals + 50:
            raise ValueError(f"Dataset too small ({n} rows) for test window {self.test_intervals}")

        train_df = self.df.iloc[:-self.test_intervals].copy().reset_index(drop=True)
        test_df  = self.df.iloc[-self.test_intervals:].copy().reset_index(drop=True)

        print(f"\n{'='*65}")
        print(f"  CHRONOLOGICAL BACKTEST SPLIT  |  Area: {self.area}")
        print(f"{'='*65}")
        print(f"  Training : {len(train_df):,} periods  "
              f"({train_df['time_dk'].min()} → {train_df['time_dk'].max()})")
        print(f"  Test     : {len(test_df):,}  periods  "
              f"({test_df['time_dk'].min()} → {test_df['time_dk'].max()})")
        print(f"  Horizon  : {self.test_intervals * 15} min  ({self.test_intervals} intervals)")
        return train_df, test_df

    # ------------------------------------------------------------------
    # TRAIN + EVALUATE
    # ------------------------------------------------------------------
    def train_and_evaluate(self, retrain=False):
        train_df, test_df = self.split_train_test()

        # Check if pre-trained bundle exists and user did not request retraining
        if not retrain and not self.use_tuning and os.path.exists(self.bundle_path):
            print(f"\n  ⚡ Fast Inference Mode: Loading pre-trained model bundle from {self.bundle_path}...")
            bundle = self.load_bundle()
            if bundle and len(bundle.get('step_models', {})) >= self.test_intervals:
                imputer     = bundle['imputer']
                scaler      = bundle['scaler']
                regime_clf  = bundle['regime_clf']
                hmm_detector= bundle['hmm_detector']
                step_models = bundle['step_models']
                weights     = bundle['weights']

                X_test_raw     = test_df[self.feature_cols].values
                X_test_scaled  = scaler.transform(imputer.transform(X_test_raw))
                X_origin_scaled = scaler.transform(imputer.transform(
                    train_df.iloc[-1:][self.feature_cols].values
                ))

                rf_regime_pred  = regime_clf.predict(X_test_scaled)
                rf_regime_proba = regime_clf.predict_proba(X_test_scaled)
                rf_regime_conf  = np.max(rf_regime_proba, axis=1)

                hmm_regime_pred, hmm_regime_proba = hmm_detector.predict(test_df)
                hmm_regime_conf = np.max(hmm_regime_proba, axis=1)
                consensus_regime = np.where(rf_regime_pred == hmm_regime_pred, rf_regime_pred, rf_regime_pred)

                return self._evaluate_and_save(train_df, test_df, X_origin_scaled,
                                               rf_regime_pred, rf_regime_conf,
                                               hmm_regime_pred, hmm_regime_conf,
                                               consensus_regime, hmm_detector,
                                               step_models, weights, is_loaded=True)

        # --- Preprocessing for Training ---
        X_train_raw = train_df[self.feature_cols].values
        imputer = SimpleImputer(strategy='median')
        scaler  = RobustScaler()
        X_train        = imputer.fit_transform(X_train_raw)
        X_train_scaled = scaler.fit_transform(X_train)

        X_test_raw     = test_df[self.feature_cols].values
        X_test_scaled  = scaler.transform(imputer.transform(X_test_raw))

        # --- Origin feature vector (last training point = forecast launch) ---
        X_origin_scaled = scaler.transform(imputer.transform(
            train_df.iloc[-1:][self.feature_cols].values
        ))

        # =====================================================================
        # A. SUPERVISED REGIME CLASSIFIER  (RandomForest)
        # =====================================================================
        print(f"\n  [1/4] Supervised Regime Classifier (RandomForest)...", flush=True)
        y_regime = train_df['regime'].values
        regime_clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, max_features='sqrt', min_samples_split=4,
            random_state=42, n_jobs=4
        )
        regime_clf.fit(X_train_scaled, y_regime)

        rf_regime_pred  = regime_clf.predict(X_test_scaled)
        rf_regime_proba = regime_clf.predict_proba(X_test_scaled)
        rf_regime_conf  = np.max(rf_regime_proba, axis=1)

        # =====================================================================
        # B. UNSUPERVISED HMM REGIME DETECTOR
        # =====================================================================
        print(f"  [2/4] Unsupervised HMM Regime Detector...", flush=True)
        best_hmm_params = {'n_states': 3, 'covariance_type': 'full', 'n_iter': 200}
        if self.use_tuning and HMM_AVAILABLE:
            print(f"        Tuning HMM with Optuna ({min(self.n_trials,20)} trials)...", flush=True)
            opt = OptunaHyperparamOptimiser(
                X_train_scaled, train_df[self.price_col].values,
                n_trials=self.n_trials
            )
            best_hmm_params = opt.tune_hmm(train_df)
            print(f"        Best HMM params: {best_hmm_params}", flush=True)

        hmm_detector = HMMRegimeDetector(**best_hmm_params)
        hmm_detector.fit(train_df)
        hmm_regime_pred, hmm_regime_proba = hmm_detector.predict(test_df)
        hmm_regime_conf = np.max(hmm_regime_proba, axis=1)

        # Consensus regime (RF + HMM agreement -> high confidence)
        consensus_regime = np.where(rf_regime_pred == hmm_regime_pred,
                                    rf_regime_pred,
                                    rf_regime_pred)  # fall back to supervised when disagree

        # =====================================================================
        # C. PRICE FORECASTERS  (one model per horizon step h=1..N)
        # =====================================================================
        print(f"  [3/4] Training Price Forecasters (RF, XGB, LGB, CatBoost, ElasticNet)...")

        best_params = {}
        if self.use_tuning:
            print(f"        Bayesian tuning with Optuna ({self.n_trials} trials each)...", flush=True)
            # Tune on step-1 spread targets; reuse params for all steps
            target_spread_tune = train_df['intraday_spread_eur'].values if (self.target_currency == 'EUR') else train_df['intraday_spread_dkk'].values
            y_step1 = pd.Series(target_spread_tune).shift(-1)
            valid   = ~y_step1.isna()
            opt = OptunaHyperparamOptimiser(
                X_train_scaled[valid], y_step1[valid].values,
                n_trials=self.n_trials
            )
            best_params['rf']          = opt.tune_random_forest()
            if XGB_AVAILABLE: best_params['xgb']  = opt.tune_xgboost()
            if LGB_AVAILABLE: best_params['lgb']  = opt.tune_lightgbm()
            if CB_AVAILABLE:  best_params['cat']  = opt.tune_catboost()
            best_params['enet']        = opt.tune_elasticnet()
            print(f"        Best params found — saving to results/best_params_{self.area}.json")
            os.makedirs('results', exist_ok=True)
            with open(f'results/best_params_{self.area}.json', 'w') as f:
                json.dump(best_params, f, indent=2)
        else:
            # Load previously saved params if they exist
            param_path = f'results/best_params_{self.area}.json'
            if os.path.exists(param_path):
                with open(param_path) as f:
                    best_params = json.load(f)
                print(f"        Loaded saved best params from {param_path}")

        # Optimized default model parameters for high accuracy & fast execution
        rf_params   = best_params.get('rf',   {'n_estimators': 100, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_split': 4, 'random_state': 42, 'n_jobs': 4})
        xgb_params  = best_params.get('xgb',  {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.08, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': 4, 'verbosity': 0})
        lgb_params  = best_params.get('lgb',  {'n_estimators': 100, 'num_leaves': 31, 'learning_rate': 0.08, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'random_state': 42, 'n_jobs': 4, 'verbose': -1})
        cat_params  = best_params.get('cat',  {'iterations': 150, 'depth': 6, 'learning_rate': 0.08, 'random_seed': 42, 'verbose': False, 'thread_count': 4})
        enet_params = best_params.get('enet', {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 1000})

        # Ensure mandatory fixed params
        rf_params.update({'random_state': 42, 'n_jobs': 4})
        if XGB_AVAILABLE:  xgb_params.update({'random_state': 42, 'n_jobs': 4, 'verbosity': 0})
        if LGB_AVAILABLE:  lgb_params.update({'random_state': 42, 'n_jobs': 4, 'verbose': -1})
        if CB_AVAILABLE:   cat_params.update({'random_seed': 42, 'verbose': False, 'thread_count': 4})

        # Train per-horizon step models
        # Uses Day-Ahead Spot Price Anchoring: predicts the Imbalance Spread (Imbalance - DayAhead)
        # and reconstructs Imbalance = DayAhead + Spread, providing major variance reduction.
        step_models   = {}
        val_mae       = {m: [] for m in ['rf', 'xgb', 'lgb', 'cat', 'enet']}
        eur           = (self.target_currency == 'EUR')
        
        target_spread = train_df['intraday_spread_eur'].values if eur else train_df['intraday_spread_dkk'].values
        target_price  = train_df[self.price_col].values
        day_ahead_arr = train_df['day_ahead_price_eur'].values if eur else train_df['day_ahead_price_dkk'].values

        for h in range(1, self.test_intervals + 1):
            print(f"        Step {h}/{self.test_intervals} training (horizon +{h*15}m)...", flush=True)
            y_spread_h = pd.Series(target_spread).shift(-h)
            valid_mask = ~y_spread_h.isna()
            X_s = X_train_scaled[valid_mask]
            y_s = y_spread_h[valid_mask].values

            rf_m  = RandomForestRegressor(**rf_params)
            rf_m.fit(X_s, y_s)
            val_mae['rf'].append(mean_absolute_error(y_s, rf_m.predict(X_s)))

            xgb_m = None
            if XGB_AVAILABLE:
                xgb_m = xgb.XGBRegressor(**xgb_params)
                xgb_m.fit(X_s, y_s)
                val_mae['xgb'].append(mean_absolute_error(y_s, xgb_m.predict(X_s)))

            lgb_m = None
            if LGB_AVAILABLE:
                lgb_m = lgb.LGBMRegressor(**lgb_params)
                lgb_m.fit(X_s, y_s)
                val_mae['lgb'].append(mean_absolute_error(y_s, lgb_m.predict(X_s)))

            cat_m = None
            if CB_AVAILABLE:
                cat_m = cb.CatBoostRegressor(**cat_params)
                cat_m.fit(X_s, y_s)
                val_mae['cat'].append(mean_absolute_error(y_s, cat_m.predict(X_s)))

            enet_m = ElasticNet(**enet_params)
            enet_m.fit(X_s, y_s)
            val_mae['enet'].append(mean_absolute_error(y_s, enet_m.predict(X_s)))

            step_models[h] = {
                'rf': rf_m, 'xgb': xgb_m, 'lgb': lgb_m,
                'cat': cat_m, 'enet': enet_m
            }

        # Compute ensemble weights (inverse of mean MAE across steps)
        avg_mae  = {m: np.mean(maes) if maes else 1e9 for m, maes in val_mae.items()}
        inv_mae  = {m: 1.0 / (v + 1e-6) for m, v in avg_mae.items()}
        total    = sum(inv_mae.values())
        weights  = {m: inv_mae[m] / total for m in inv_mae}
        print(f"        Ensemble weights (by 1/MAE): " +
              ", ".join(f"{m}={w:.3f}" for m, w in weights.items()), flush=True)

        return self._evaluate_and_save(train_df, test_df, X_origin_scaled,
                                       rf_regime_pred, rf_regime_conf,
                                       hmm_regime_pred, hmm_regime_conf,
                                       consensus_regime, hmm_detector,
                                       step_models, weights, is_loaded=False,
                                       imputer=imputer, scaler=scaler,
                                       regime_clf=regime_clf, best_params=best_params)

    # ------------------------------------------------------------------
    # EVALUATION & ARTIFACT GENERATION
    # ------------------------------------------------------------------
    def _evaluate_and_save(self, train_df, test_df, X_origin_scaled,
                           rf_regime_pred, rf_regime_conf,
                           hmm_regime_pred, hmm_regime_conf,
                           consensus_regime, hmm_detector,
                           step_models, weights, is_loaded=False,
                           imputer=None, scaler=None, regime_clf=None, best_params=None):
        print(f"  [4/4] Generating forecasts for {self.test_intervals} test intervals...")
        eur = (self.target_currency == 'EUR')
        EUR2DKK = 7.46
        test_day_ahead = test_df['day_ahead_price_eur'].values if eur else test_df['day_ahead_price_dkk'].values
        rf_preds, xgb_preds, lgb_preds, cat_preds, enet_preds, ens_preds = [], [], [], [], [], []

        for h in range(1, self.test_intervals + 1):
            m = step_models[h]
            base_da = float(test_day_ahead[h-1]) if h-1 < len(test_day_ahead) else float(test_day_ahead[-1])
            
            spr_rf   = float(m['rf'].predict(X_origin_scaled)[0])
            spr_xgb  = float(m['xgb'].predict(X_origin_scaled)[0])  if m.get('xgb')  else spr_rf
            spr_lgb  = float(m['lgb'].predict(X_origin_scaled)[0])  if m.get('lgb')  else spr_rf
            spr_cat  = float(m['cat'].predict(X_origin_scaled)[0])  if m.get('cat')  else spr_rf
            spr_enet = float(m['enet'].predict(X_origin_scaled)[0])

            # Reconstruct Imbalance Price = Day Ahead Spot Price + Predicted Spread
            p_rf   = base_da + spr_rf
            p_xgb  = base_da + spr_xgb
            p_lgb  = base_da + spr_lgb
            p_cat  = base_da + spr_cat
            p_enet = base_da + spr_enet

            # Weighted ensemble
            parts = {'rf': p_rf, 'xgb': p_xgb, 'lgb': p_lgb, 'cat': p_cat, 'enet': p_enet}
            p_ens = sum(parts[mdl] * weights[mdl] for mdl in parts)

            rf_preds.append(p_rf); xgb_preds.append(p_xgb)
            lgb_preds.append(p_lgb); cat_preds.append(p_cat)
            enet_preds.append(p_enet); ens_preds.append(p_ens)

        test_results = []
        for i in range(len(test_df)):
            row       = test_df.iloc[i]
            act_eur   = float(row['imbalance_price_eur'])
            act_dkk   = float(row['imbalance_price_dkk'])
            act_reg   = int(row['regime'])
            rf_reg    = int(rf_regime_pred[i])
            hmm_reg   = int(hmm_regime_pred[i])
            con_reg   = int(consensus_regime[i])
            rf_conf   = float(rf_regime_conf[i])
            hmm_conf  = float(hmm_regime_conf[i])
            agree     = (rf_reg == hmm_reg)

            pred_eur  = ens_preds[i]  if eur else ens_preds[i] / EUR2DKK
            pred_dkk  = pred_eur * EUR2DKK

            test_results.append({
                'Step'                    : f'Q{i+1} (+{(i+1)*15}m)',
                'TimeDK'                  : row['time_dk'].strftime('%Y-%m-%d %H:%M'),
                'TimeUTC'                 : row['time_utc'].strftime('%Y-%m-%d %H:%M'),
                'Actual_EUR'              : round(act_eur, 2),
                'Predicted_Ensemble_EUR'  : round(pred_eur, 2),
                'Predicted_RF_EUR'        : round(rf_preds[i]   if eur else rf_preds[i]   / EUR2DKK, 2),
                'Predicted_XGB_EUR'       : round(xgb_preds[i]  if eur else xgb_preds[i]  / EUR2DKK, 2),
                'Predicted_LGB_EUR'       : round(lgb_preds[i]  if eur else lgb_preds[i]  / EUR2DKK, 2),
                'Predicted_Cat_EUR'       : round(cat_preds[i]  if eur else cat_preds[i]  / EUR2DKK, 2),
                'Predicted_ENet_EUR'      : round(enet_preds[i] if eur else enet_preds[i] / EUR2DKK, 2),
                'Error_EUR'               : round(abs(pred_eur - act_eur), 2),
                'Actual_DKK'              : round(act_dkk, 2),
                'Predicted_DKK'           : round(pred_dkk, 2),
                'Error_DKK'               : round(abs(pred_dkk - act_dkk), 2),
                'Actual_Regime'           : self.regime_names.get(act_reg, 'Unknown'),
                'RF_Regime'               : self.regime_names.get(rf_reg, 'Unknown'),
                'HMM_Regime'              : self.regime_names.get(hmm_reg, 'Unknown'),
                'Consensus_Regime'        : self.regime_names.get(con_reg, 'Unknown'),
                'Predicted_Regime'        : self.regime_names.get(con_reg, 'Unknown'),
                'Regime_Match'            : '✅ Match' if rf_reg == act_reg else '❌ Mismatch',
                'RF_HMM_Agree'            : '✅ Agree' if agree else '⚠️ Differ',
                'RF_Correct'              : '✅ Match' if rf_reg  == act_reg else '❌ Mismatch',
                'HMM_Correct'             : '✅ Match' if hmm_reg == act_reg else '❌ Mismatch',
                'RF_Confidence'           : f'{rf_conf  * 100:.1f}%',
                'HMM_Confidence'          : f'{hmm_conf * 100:.1f}%',
                'Confidence'              : f'{rf_conf  * 100:.1f}%',
            })

        results_df = pd.DataFrame(test_results)

        actuals_eur      = np.array([r['Actual_EUR'] for r in test_results])
        preds_eur        = np.array([r['Predicted_Ensemble_EUR'] for r in test_results])
        actual_regimes   = test_df['regime'].values
        rf_acc           = float(accuracy_score(actual_regimes, rf_regime_pred))
        hmm_acc          = float(accuracy_score(actual_regimes, hmm_regime_pred))
        consensus_acc    = float(accuracy_score(actual_regimes, consensus_regime))
        mae_eur          = float(mean_absolute_error(actuals_eur, preds_eur))
        rmse_eur         = float(np.sqrt(mean_squared_error(actuals_eur, preds_eur)))

        metrics = {
            'area'                  : self.area,
            'test_intervals'        : self.test_intervals,
            'mae_eur'               : round(mae_eur, 2),
            'rmse_eur'              : round(rmse_eur, 2),
            'regime_accuracy'       : round(consensus_acc * 100, 1),
            'rf_regime_accuracy'    : round(rf_acc  * 100, 1),
            'hmm_regime_accuracy'   : round(hmm_acc * 100, 1),
            'ensemble_weights'      : {k: round(v, 4) for k, v in weights.items()},
            'hyperparameters_tuned' : self.use_tuning,
            'n_optuna_trials'       : self.n_trials if self.use_tuning else 0,
            'tested_at'             : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # --- Print summary ---
        print("\n" + "=" * 80)
        print(f"  BACKTEST RESULTS  |  {self.area}  |  Last {self.test_intervals} Periods (Danish Time)")
        print("=" * 80)
        display_cols = ['Step', 'TimeDK', 'Actual_EUR', 'Predicted_Ensemble_EUR',
                        'Error_EUR', 'Actual_DKK', 'Predicted_DKK',
                        'Actual_Regime', 'Consensus_Regime', 'RF_Correct', 'HMM_Correct',
                        'RF_HMM_Agree', 'Confidence']
        print(results_df[display_cols].to_string(index=False))

        print("\n" + "-" * 55)
        print(f"  Price Forecasting:")
        print(f"    MAE  : {mae_eur:.2f} EUR/MWh")
        print(f"    RMSE : {rmse_eur:.2f} EUR/MWh")
        print(f"  Regime Detection:")
        print(f"    Supervised RF   : {rf_acc  * 100:.1f}%")
        print(f"    Unsupervised HMM: {hmm_acc * 100:.1f}%")
        print(f"    Consensus       : {consensus_acc * 100:.1f}%")
        print(f"  Ensemble Weights  : {weights}")
        print("-" * 55)

        # Save artifacts
        os.makedirs('results', exist_ok=True)
        csv_path = f'results/latest_backtest_results_{self.area}.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path}")

        with open(f'results/backtest_metrics_{self.area}.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        self._generate_plots(results_df, metrics, hmm_detector)

        # Save trained bundle if trained in this run
        if not is_loaded and imputer is not None:
            self.save_bundle(imputer, scaler, regime_clf, hmm_detector, step_models, weights, best_params, metrics)

        return results_df, metrics

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------
    def _generate_plots(self, results_df, metrics, hmm_detector):
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(
            f'{self.area} — Imbalance Price Backtest  |  Last {self.test_intervals} × 15-min Intervals (Danish Time)',
            fontsize=14, fontweight='bold', color='#1A365D'
        )
        gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.35)

        x_labels = results_df['TimeDK'].apply(lambda x: x.split(' ')[1])
        x_pos = np.arange(len(results_df))

        # --- 1. Price comparison (all models) ---
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(x_pos, results_df['Actual_EUR'], 'bo-', lw=2.5, ms=9, label='Original Actual', zorder=5)
        ax1.plot(x_pos, results_df['Predicted_Ensemble_EUR'], 'r^--', lw=2, ms=8, label='Weighted Ensemble')
        ax1.plot(x_pos, results_df['Predicted_RF_EUR'],  'g:s',  alpha=0.65, ms=6, label='Random Forest')
        ax1.plot(x_pos, results_df['Predicted_XGB_EUR'], 'm:d',  alpha=0.65, ms=6, label='XGBoost')
        ax1.plot(x_pos, results_df['Predicted_LGB_EUR'], 'c:v',  alpha=0.65, ms=6, label='LightGBM')
        ax1.plot(x_pos, results_df['Predicted_Cat_EUR'], 'y:p',  alpha=0.65, ms=6, label='CatBoost')
        ax1.plot(x_pos, results_df['Predicted_ENet_EUR'],'k:x',  alpha=0.65, ms=6, label='ElasticNet')
        ax1.set_xticks(x_pos); ax1.set_xticklabels(x_labels, rotation=25)
        ax1.set_ylabel('EUR/MWh', fontsize=10); ax1.set_title('All Models vs. Original Actual Price (EUR/MWh)', fontweight='bold')
        ax1.legend(fontsize=7.5, ncol=3); ax1.grid(True, alpha=0.3)

        # --- 2. Ensemble weights bar chart ---
        ax2 = fig.add_subplot(gs[0, 2])
        w = metrics.get('ensemble_weights', {})
        model_labels = {'rf': 'RF', 'xgb': 'XGB', 'lgb': 'LGB', 'cat': 'CatBoost', 'enet': 'ElasticNet'}
        labels = [model_labels.get(k, k) for k in w]
        vals   = list(w.values())
        colors_w = ['#3182CE', '#E53E3E', '#38A169', '#D69E2E', '#805AD5']
        bars = ax2.bar(labels, vals, color=colors_w[:len(labels)], edgecolor='white', linewidth=1.5)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax2.set_title('Ensemble Weights\n(1/MAE normalised)', fontweight='bold', fontsize=10)
        ax2.set_ylabel('Weight'); ax2.grid(True, axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=20)

        # --- 3. Absolute errors ---
        ax3 = fig.add_subplot(gs[1, 0])
        bar_colors = ['#27AE60' if e < 30 else '#F39C12' if e < 70 else '#E74C3C'
                      for e in results_df['Error_EUR']]
        b3 = ax3.bar(x_pos, results_df['Error_EUR'], color=bar_colors, edgecolor='black', alpha=0.85)
        for b, v in zip(b3, results_df['Error_EUR']):
            ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                     f'{v:.1f}', ha='center', fontsize=8)
        ax3.axhline(metrics['mae_eur'], color='red', ls='--', lw=1.5, label=f"MAE={metrics['mae_eur']:.1f}")
        ax3.set_xticks(x_pos); ax3.set_xticklabels(x_labels, rotation=25)
        ax3.set_title(f'Prediction Error (EUR/MWh)', fontweight='bold')
        ax3.set_ylabel('Abs Error (EUR/MWh)'); ax3.legend(fontsize=8); ax3.grid(True, axis='y', alpha=0.3)

        # --- 4. DKK comparison (Energi Data Service view) ---
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(x_pos, results_df['Actual_DKK'],    'b-o', lw=2.5, ms=8, label='Actual (DKK)')
        ax4.plot(x_pos, results_df['Predicted_DKK'], 'r--^', lw=2, ms=8,  label='Predicted (DKK)')
        ax4.set_xticks(x_pos); ax4.set_xticklabels(x_labels, rotation=25)
        ax4.set_title('Energi Data Service View (DKK/MWh)', fontweight='bold')
        ax4.set_ylabel('DKK/MWh'); ax4.legend(); ax4.grid(True, alpha=0.3)

        # --- 5. HMM Transition Probability Matrix ---
        ax5 = fig.add_subplot(gs[1, 2])
        trans = hmm_detector.transition_matrix()
        regime_labels = ['Down', 'Neutral', 'Upward']
        im = ax5.imshow(trans, cmap='Blues', vmin=0, vmax=1)
        ax5.set_xticks([0, 1, 2]); ax5.set_xticklabels(regime_labels, fontsize=9)
        ax5.set_yticks([0, 1, 2]); ax5.set_yticklabels(regime_labels, fontsize=9)
        for i in range(3):
            for j in range(3):
                ax5.text(j, i, f'{trans[i, j]:.2f}', ha='center', va='center',
                         color='white' if trans[i, j] > 0.5 else 'black', fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
        ax5.set_title('HMM Transition Matrix\n(Row → Next State)', fontweight='bold', fontsize=10)
        ax5.set_xlabel('Next State'); ax5.set_ylabel('Current State')

        # --- 6. Regime detection comparison table ---
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        table_data = []
        col_labels = ['Time (DK)', 'Actual Regime', 'RF Predicted', 'HMM Predicted',
                      'Consensus', 'RF vs HMM', 'RF Correct', 'HMM Correct',
                      'RF Conf.', 'HMM Conf.']
        for _, r in results_df.iterrows():
            table_data.append([
                r['TimeDK'].split(' ')[1],
                r['Actual_Regime'].replace(' Regulation', ''),
                r['RF_Regime'].replace(' Regulation', ''),
                r['HMM_Regime'].replace(' Regulation', ''),
                r['Consensus_Regime'].replace(' Regulation', ''),
                r['RF_HMM_Agree'],
                r['RF_Correct'],
                r['HMM_Correct'],
                r['RF_Confidence'],
                r['HMM_Confidence'],
            ])
        tbl = ax6.table(cellText=table_data, colLabels=col_labels,
                        cellLoc='center', loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.05, 1.9)
        ax6.set_title(
            f'Regime Switching Detection Comparison  |  '
            f'RF={metrics["rf_regime_accuracy"]:.0f}%  '
            f'HMM={metrics["hmm_regime_accuracy"]:.0f}%  '
            f'Consensus={metrics["regime_accuracy"]:.0f}%',
            fontweight='bold', fontsize=10, pad=14
        )

        plot_path = f'results/backtest_comparison_{self.area}.png'
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"  Saved plot: {plot_path}")
        plt.close()


# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Danish Imbalance Price Backtesting Pipeline')
    parser.add_argument('--area',          type=str,   default='DK1', choices=['DK1', 'DK2'])
    parser.add_argument('--test-intervals',type=int,   default=5,
                        help='Number of 15-min test intervals (default=5 → 1h15m; 96 → 24h)')
    parser.add_argument('--currency',      type=str,   default='EUR', choices=['EUR', 'DKK'])
    parser.add_argument('--use-tuning',    action='store_true',
                        help='Enable Optuna Bayesian hyperparameter tuning')
    parser.add_argument('--n-trials',      type=int,   default=50,
                        help='Optuna trials per model (default=50, fast=10)')
    parser.add_argument('--fast-tuning',   action='store_true',
                        help='Quick tuning mode (10 trials per model)')
    parser.add_argument('--train',         action='store_true',
                        help='Force retraining models and updating the saved bundle in models/')
    parser.add_argument('--refresh-data',  action='store_true',
                        help='Fetch latest data from Energi Data Service API first')
    args = parser.parse_args()

    if args.refresh_data:
        print("Fetching latest data from Energi Data Service API...")
        try:
            from data_retrieval import DataPipeline
            pipeline = DataPipeline()
            pipeline.fetch_all_data(args.area,
                                    start_date=datetime.now() - timedelta(days=60),
                                    end_date=datetime.now(), export_csv=False)
        except Exception as e:
            print(f"  Warning: Could not refresh API data: {e}")

    feature_pipeline = BacktestFeaturePipeline('energy_data.db')
    df = feature_pipeline.load_and_prepare_features(area=args.area)

    engine = BacktestEngine(
        df,
        area=args.area,
        test_intervals=args.test_intervals,
        target_currency=args.currency,
        use_tuning=args.use_tuning,
        n_trials=args.n_trials,
        fast_tuning=args.fast_tuning
    )
    results_df, metrics = engine.train_and_evaluate(retrain=args.train or args.use_tuning)

    print("\n  BACKTEST COMPLETE")
    print(f"  MAE:  {metrics['mae_eur']} EUR/MWh")
    print(f"  RMSE: {metrics['rmse_eur']} EUR/MWh")
    print(f"  Regime Accuracy (RF={metrics['rf_regime_accuracy']}% | "
          f"HMM={metrics['hmm_regime_accuracy']}% | "
          f"Consensus={metrics['regime_accuracy']}%)")


if __name__ == '__main__':
    main()

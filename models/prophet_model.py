# ==============================================================================
# models/prophet_model.py — Facebook Prophet for 5-step Imbalance Price Forecasting
# ==============================================================================

import os
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("  [INFO] prophet not installed – pip install prophet")


def train_prophet_models(df: pd.DataFrame, area: str = 'DK1',
                         price_col: str = 'imbalance_price_eur',
                         save_dir: str = 'models'):
    """
    Train 5 Prophet models, one per forecast horizon (Q1=+15m … Q5=+75m).
    Uses the last 15% as validation hold-out.
    Returns (models_dict, val_mae).
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet not installed.  pip install prophet")

    print("\n" + "=" * 65)
    print("  PROPHET TRAINING PIPELINE")
    print("=" * 65)

    # Prepare data: Prophet needs 'ds' (datetime) and 'y' (target)
    pdf = df[['time_dk', price_col]].copy()
    pdf.columns = ['ds', 'y']
    pdf['ds'] = pd.to_datetime(pdf['ds'])
    pdf = pdf.dropna(subset=['y']).sort_values('ds').reset_index(drop=True)

    # Add regressors if available
    regressor_cols = []
    for col in ['spot_price_eur', 'satisfied_demand', 'intraday_spread_eur']:
        if col in df.columns:
            pdf[col] = df[col].values[:len(pdf)]
            pdf[col] = pdf[col].fillna(pdf[col].median())
            regressor_cols.append(col)

    # Create future targets: y_{t+step} for each horizon
    for step in range(1, 6):
        pdf[f'y_step{step}'] = pdf['y'].shift(-step)
    pdf = pdf.dropna().reset_index(drop=True)

    # Use recent ~3,500 periods (last ~5 weeks) for responsive, fast Prophet fitting
    if len(pdf) > 3500:
        pdf = pdf.iloc[-3500:].reset_index(drop=True)

    # Split: last 15% for validation
    n_val = max(int(len(pdf) * 0.15), 100)
    n_train = len(pdf) - n_val
    train = pdf.iloc[:n_train].copy()
    val = pdf.iloc[n_train:].copy()

    models = {}
    val_maes = []

    for step in range(1, 6):
        print(f"\n  Step {step}/5 (horizon +{step*15}m)...")

        # Prepare training data for this horizon
        train_step = train[['ds', f'y_step{step}'] + regressor_cols].copy()
        train_step = train_step.rename(columns={f'y_step{step}': 'y'})

        m = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        # Add intraday seasonality (96 intervals per day)
        m.add_seasonality(name='intraday', period=1, fourier_order=12)
        # Add Danish holidays
        m.add_country_holidays(country_name='DK')

        for reg in regressor_cols:
            m.add_regressor(reg)

        m.fit(train_step)

        # Validate
        val_step = val[['ds'] + regressor_cols].copy()
        forecast = m.predict(val_step)
        val_preds = forecast['yhat'].values
        val_actuals = val[f'y_step{step}'].values
        mae = np.mean(np.abs(val_preds - val_actuals))
        val_maes.append(mae)
        print(f"    Step {step} Val MAE: {mae:.2f} EUR/MWh")

        models[f'step{step}'] = m

    overall_mae = np.mean(val_maes)
    print(f"\n  Overall Prophet Val MAE: {overall_mae:.2f} EUR/MWh")

    # Save bundle
    bundle = {
        'models': models,
        'regressor_cols': regressor_cols,
        'val_mae': overall_mae,
        'val_maes_per_step': val_maes,
        'area': area,
    }
    save_path = os.path.join(save_dir, f'{area}_prophet.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"  Saved: {save_path}")

    return bundle


def predict_with_prophet(bundle: dict, future_times: list, regressor_values: dict = None):
    """
    Predict 5-step prices using trained Prophet models.
    future_times: list of 5 datetime objects (Q1..Q5 Danish Time).
    regressor_values: dict {col_name: value} for external regressors.
    Returns np.array of shape (5,).
    """
    predictions = []
    for step in range(1, 6):
        model = bundle['models'][f'step{step}']
        future_df = pd.DataFrame({'ds': [future_times[step - 1]]})

        for col in bundle['regressor_cols']:
            if regressor_values and col in regressor_values:
                future_df[col] = regressor_values[col]
            else:
                future_df[col] = 0.0

        forecast = model.predict(future_df)
        predictions.append(forecast['yhat'].values[0])

    return np.array(predictions)


def load_prophet_bundle(area='DK1', save_dir='models'):
    """Load the saved Prophet model bundle."""
    path = os.path.join(save_dir, f'{area}_prophet.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

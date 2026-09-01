# ============================================
# REAL-TIME IMBALANCE PRICE PREDICTION
# Using Trained Model: models/DK1_model_5future.pkl
# ============================================

import requests
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')

# For visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print("=" * 70)
print("🏆 REAL-TIME IMBALANCE PRICE PREDICTION")
print("   Using Trained Model: models/DK1_model_5future.pkl")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)


# ============================================
# DEFINE ALL CUSTOM CLASSES (REQUIRED FOR LOADING)
# ============================================

class OptimizedRegimeClassifier:
    """
    This class was used to train and save the model.
    It must be defined here to load the pickle file.
    """

    def __init__(self, model=None, scaler=None, features=None, metrics=None, area=None):
        self.model = model
        self.scaler = scaler
        self.features = features
        self.metrics = metrics
        self.area = area
        self.prediction_horizon = None
        self.regime_names = ['Downward', 'Neutral', 'Upward']
        self.regime_positions = {
            0: {'signal': -1, 'size': 0.6},
            1: {'signal': 0, 'size': 0.1},
            2: {'signal': 1, 'size': 0.8}
        }

    def predict(self, X):
        """Predict using the underlying model"""
        if self.model is not None:
            if hasattr(self.model, 'predict'):
                return self.model.predict(X)
            elif callable(self.model):
                return self.model(X)
        return np.zeros(len(X))

    def predict_proba(self, X):
        """Get probabilities if available"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None

    def predict_with_confidence(self, X):
        """Predict with confidence intervals"""
        predictions = self.predict(X)
        return predictions, np.ones(len(predictions)) * 0.5


class OptimizedProbabilisticForecaster:
    """
    Alternative class that might be referenced in the pickle
    """

    def __init__(self, model=None, scaler=None, features=None, metrics=None, area=None):
        self.model = model
        self.scaler = scaler
        self.features = features
        self.metrics = metrics
        self.area = area
        self.prediction_horizon = None
        self.regime_names = ['Downward', 'Neutral', 'Upward']
        self.regime_positions = {
            0: {'signal': -1, 'size': 0.6},
            1: {'signal': 0, 'size': 0.1},
            2: {'signal': 1, 'size': 0.8}
        }

    def predict(self, X):
        if self.model is not None:
            if hasattr(self.model, 'predict'):
                return self.model.predict(X)
            elif callable(self.model):
                return self.model(X)
        return np.zeros(len(X))

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None


# ============================================
# STEP 1: LOAD THE TRAINED MODEL
# ============================================

def load_model():
    """Load the trained model from models folder"""

    model_path = 'models/DK1_model_5future.pkl'

    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("   Please make sure the file exists in the models folder.")
        return None

    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        print(f"\n✅ Model loaded successfully!")
        print(f"   Path: {model_path}")

        # Print model info
        if isinstance(model_data, dict):
            print(f"   Type: Dictionary")
            print(f"   Keys: {list(model_data.keys())}")

            if 'model' in model_data:
                print(f"   Model type: {type(model_data['model']).__name__}")
            if 'features' in model_data:
                print(f"   Features: {len(model_data['features'])}")
            if 'metrics' in model_data:
                metrics = model_data['metrics']
                if 'test_r2' in metrics:
                    print(f"   Test R²: {metrics['test_r2']:.4f}")
                if 'test_mae' in metrics:
                    print(f"   Test MAE: {metrics['test_mae']:.2f} DKK/MWh")
            if 'area' in model_data:
                print(f"   Area: {model_data['area']}")
        else:
            print(f"   Type: {type(model_data).__name__}")

        return model_data

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


# ============================================
# STEP 2: FETCH REAL-TIME IMBALANCE PRICES
# ============================================

def fetch_imbalance_prices(area='DK1', n=5, end_time=None):
    """Fetch the last N imbalance prices from Energinet API"""

    if end_time is None:
        end_time = datetime.now()

    # Round down to nearest 15 minutes
    end_time = end_time.replace(minute=(end_time.minute // 15) * 15, second=0, microsecond=0)

    # Calculate start time
    start_time = end_time - timedelta(minutes=n * 15 + 30)

    print(f"\n📊 Fetching last {n} imbalance prices for {area}")
    print(f"   Time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
    print("-" * 50)

    url = "https://api.energidataservice.dk/dataset/ImbalancePrice"

    params = {
        'limit': n + 5,
        'start': start_time.strftime('%Y-%m-%d'),
        'end': end_time.strftime('%Y-%m-%d'),
        'sort': 'TimeUTC DESC'
    }

    try:
        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 429:
            print("   ⏳ Rate limited. Waiting 65 seconds...")
            time.sleep(65)
            return fetch_imbalance_prices(area, n, end_time)

        if response.status_code != 200:
            print(f"   ⚠️ API Error: {response.status_code}")
            return None

        data = response.json()
        records = data.get('records', [])

        if not records:
            print("   ⚠️ No records found")
            return None

        # Parse records
        results = []
        for record in records:
            if record.get('PriceArea') == area:
                try:
                    results.append({
                        'time_utc': pd.to_datetime(record.get('TimeUTC')),
                        'time_dk': pd.to_datetime(record.get('TimeDK')),
                        'price': float(record.get('ImbalancePriceDKK', 0))
                    })
                except:
                    continue

        # Sort by time and get last N
        df = pd.DataFrame(results)
        if df.empty:
            print("   ⚠️ No data for area")
            return None

        df = df.sort_values('time_utc')
        df = df.tail(n)

        print(f"   ✅ Retrieved {len(df)} records:")
        for _, row in df.iterrows():
            print(f"      {row['time_utc'].strftime('%H:%M')} → {row['price']:.2f} DKK/MWh")

        return df

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


# ============================================
# STEP 3: PREDICT USING TRAINED MODEL
# ============================================

def predict_with_model(model_data, actual_data):
    """Use the trained model to predict prices"""

    if model_data is None or actual_data is None:
        return None

    # Extract model components
    if isinstance(model_data, dict):
        model = model_data.get('model')
        scaler = model_data.get('scaler')
        feature_cols = model_data.get('features', [])
    else:
        # If model_data is the model directly
        model = model_data
        scaler = None
        feature_cols = []

    if model is None:
        print("❌ No model found in loaded data")
        return None

    print("\n🔮 Generating predictions...")

    # Get prices
    prices = actual_data['price'].values

    # Create features from the actual data
    features = []
    for i in range(len(prices)):
        lag_features = []
        # Use last 4 prices as features
        for lag in [1, 2, 3, 4, 6, 8]:
            if i - lag >= 0:
                lag_features.append(prices[i - lag])
            else:
                lag_features.append(prices[0] if len(prices) > 0 else 0)
        features.append(lag_features)

    features = np.array(features)

    # Scale features if scaler exists
    if scaler is not None and hasattr(scaler, 'transform'):
        try:
            features_scaled = scaler.transform(features)
        except:
            features_scaled = features
    else:
        features_scaled = features

    # Predict
    try:
        # Check if model has predict method
        if hasattr(model, 'predict'):
            predictions = model.predict(features_scaled)
        else:
            predictions = model(features_scaled)
    except:
        # Try without scaling
        predictions = model.predict(features)

    # Create results
    results = []
    for i, (_, row) in enumerate(actual_data.iterrows()):
        pred = predictions[i] if i < len(predictions) else 0
        results.append({
            'time_utc': row['time_utc'],
            'actual_price': row['price'],
            'predicted_price': pred,
            'error': pred - row['price']
        })

    return results


# ============================================
# STEP 4: VISUALIZE RESULTS
# ============================================

def plot_results(results, area='DK1'):
    """Plot actual vs predicted prices"""

    if not results:
        print("❌ No results to plot")
        return

    df = pd.DataFrame(results)
    df = df.sort_values('time_utc')

    # Calculate metrics
    mae = np.mean(np.abs(df['error']))
    rmse = np.sqrt(np.mean(df['error'] ** 2))

    # Direction accuracy
    if len(df) > 1:
        actual_direction = np.sign(df['actual_price'].diff())
        pred_direction = np.sign(df['predicted_price'].diff())
        dir_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        dir_accuracy = 0

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{area} - Imbalance Price Prediction vs Actual\nModel: models/DK1_model_5future.pkl',
                 fontsize=14, fontweight='bold')

    # 1. Time Series
    ax1 = axes[0, 0]
    ax1.plot(df['time_utc'], df['actual_price'], 'b-o', label='Actual', linewidth=2, markersize=8)
    ax1.plot(df['time_utc'], df['predicted_price'], 'r--s', label='Predicted', linewidth=2, markersize=8)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price (DKK/MWh)')
    ax1.set_title('Price Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # 2. Errors
    ax2 = axes[0, 1]
    colors = ['green' if e >= 0 else 'red' for e in df['error']]
    ax2.bar(df['time_utc'], df['error'], color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Error (DKK/MWh)')
    ax2.set_title(f'Prediction Errors\nMAE: {mae:.2f}, RMSE: {rmse:.2f}')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # 3. Scatter Plot
    ax3 = axes[1, 0]
    ax3.scatter(df['actual_price'], df['predicted_price'], s=100, color='purple', alpha=0.7)
    min_val = min(df['actual_price'].min(), df['predicted_price'].min())
    max_val = max(df['actual_price'].max(), df['predicted_price'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'g--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Actual Price (DKK/MWh)')
    ax3.set_ylabel('Predicted Price (DKK/MWh)')
    ax3.set_title(f'Actual vs Predicted\nDirection Accuracy: {dir_accuracy:.1f}%')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Error Distribution
    ax4 = axes[1, 1]
    ax4.hist(df['error'], bins=10, color='orange', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='black', linestyle='--')
    ax4.axvline(x=df['error'].mean(), color='red', linestyle='--',
                label=f"Mean Error: {df['error'].mean():.2f}")
    ax4.set_xlabel('Error (DKK/MWh)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Error Distribution\nStd: {df["error"].std():.2f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f'results/prediction_comparison_{area}_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {plot_path}")

    plt.show()

    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 PREDICTION PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"   MAE:  {mae:.2f} DKK/MWh")
    print(f"   RMSE: {rmse:.2f} DKK/MWh")
    print(f"   Direction Accuracy: {dir_accuracy:.1f}%")
    print(f"   Mean Error: {df['error'].mean():.2f} DKK/MWh")
    print(f"   Std Error: {df['error'].std():.2f} DKK/MWh")

    print("\n📊 Detailed Results:")
    print("-" * 60)
    print(f"{'Time':<12} {'Actual':<12} {'Predicted':<12} {'Error':<12}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{row['time_utc'].strftime('%H:%M'):<12} "
              f"{row['actual_price']:>8.2f}    "
              f"{row['predicted_price']:>8.2f}    "
              f"{row['error']:>8.2f}")


# ============================================
# STEP 5: SAVE RESULTS TO CSV
# ============================================

def save_results(results, area='DK1'):
    """Save results to CSV"""

    if not results:
        return

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'results/prediction_comparison_{area}_{timestamp}.csv'

    os.makedirs('results', exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")


# ============================================
# STEP 6: MAIN EXECUTION
# ============================================

def main():
    """Main execution function"""

    # Load the trained model from models folder
    print("\n📂 Loading trained model from models folder...")
    model_data = load_model()

    if model_data is None:
        print("\n❌ Could not load model. Please check:")
        print("   1. The file exists at: models/DK1_model_5future.pkl")
        print("   2. The file is a valid pickle file")
        return

    # Fetch real-time prices
    actual_data = fetch_imbalance_prices(area='DK1', n=5)

    if actual_data is None:
        print("❌ Could not fetch actual prices")
        return

    # Generate predictions
    results = predict_with_model(model_data, actual_data)

    if results is None:
        print("❌ Could not generate predictions")
        return

    # Plot results
    plot_results(results, area='DK1')

    # Save results
    save_results(results, area='DK1')

    print("\n" + "=" * 70)
    print("✅ REAL-TIME PREDICTION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
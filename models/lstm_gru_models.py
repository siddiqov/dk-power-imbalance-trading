# ==============================================================================
# models/lstm_gru_models.py — LSTM & GRU for 5-step Imbalance Price Forecasting
# ==============================================================================

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------------------------------------------------------- #
# Fast Vectorized Pre-Allocation
# --------------------------------------------------------------------------- #

def make_sequence_tensors(X_scaled: np.ndarray, y_prices: np.ndarray, lookback: int = 24):
    """
    Fast pre-allocated sliding window builder.
    Returns (X_tensor, y_tensor) with shapes:
      X_tensor: (N, lookback, n_features)
      y_tensor: (N, 5)
    """
    n_samples = len(X_scaled) - lookback - 4
    n_features = X_scaled.shape[1]

    Xs = np.empty((n_samples, lookback, n_features), dtype=np.float32)
    ys = np.empty((n_samples, 5), dtype=np.float32)

    for i in range(n_samples):
        Xs[i] = X_scaled[i: i + lookback]
        ys[i] = y_prices[i + lookback: i + lookback + 5]

    return torch.from_numpy(Xs), torch.from_numpy(ys)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class ImbalanceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 5),   # 5-step forecast
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class ImbalanceGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 5),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_sequence_model(
    X_scaled: np.ndarray,
    y_prices: np.ndarray,
    model_class,
    lookback: int = 24,
    epochs: int = 25,
    batch_size: int = 512,
    lr: float = 3e-3,
    patience: int = 8,
    verbose: bool = True,
):
    """Train an LSTM or GRU model and return (model, val_mae)."""

    n_features = X_scaled.shape[1]
    X_tens, y_tens = make_sequence_tensors(X_scaled, y_prices, lookback=lookback)

    # Time-aware split: last 15% for validation
    n_total = len(X_tens)
    n_val = max(int(n_total * 0.15), 100)
    n_train = n_total - n_val

    train_ds = TensorDataset(X_tens[:n_train], y_tens[:n_train])
    val_ds   = TensorDataset(X_tens[n_train:], y_tens[n_train:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = model_class(input_size=n_features).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=4, factor=0.5)
    criterion = nn.L1Loss()  # MAE loss

    best_val_mae = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # --- Validate ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= n_val
        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:2d}/{epochs:2d} | Train MAE: {train_loss:.2f} | Val MAE: {val_loss:.2f} EUR", flush=True)

        if val_loss < best_val_mae:
            best_val_mae = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val_mae


def predict_with_sequence_model(model, X_recent_scaled: np.ndarray):
    """Predict 5 future steps from the most recent `lookback` intervals."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_recent_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        pred = model(x).cpu().numpy().flatten()
    return pred  # shape (5,)


# --------------------------------------------------------------------------- #
# Full training pipeline
# --------------------------------------------------------------------------- #

def train_all_sequence_models(df: pd.DataFrame, feature_cols: list, price_col: str = 'imbalance_price_eur',
                              save_dir: str = 'models', area: str = 'DK1'):
    """
    Train LSTM and GRU with lookback=24 and lookback=96.
    Returns dict of {name: {'model': model, 'val_mae': float, 'lookback': int, 'scaler': scaler}}.
    """

    print("\n" + "=" * 65, flush=True)
    print("  LSTM / GRU TRAINING PIPELINE", flush=True)
    print("=" * 65, flush=True)

    # Use recent 10,000 periods (~3.5 months) for sequence models
    if len(df) > 10000:
        seq_df = df.iloc[-10000:].reset_index(drop=True)
    else:
        seq_df = df.copy()

    X_raw = seq_df[feature_cols].values.astype(np.float32)
    y_raw = seq_df[price_col].values.astype(np.float32)

    # Handle NaN
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_raw = imputer.fit_transform(X_raw)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)

    results = {}

    for lookback in [24, 96]:
        for model_name, model_class in [('lstm', ImbalanceLSTM), ('gru', ImbalanceGRU)]:
            tag = f"{model_name}_{lookback}"
            print(f"\n  Training {model_name.upper()} (lookback={lookback}, {lookback*15/60:.0f}h window)...", flush=True)

            model, val_mae = train_sequence_model(
                X_scaled, y_raw, model_class,
                lookback=lookback, epochs=25, batch_size=512, lr=3e-3, verbose=True
            )

            save_path = os.path.join(save_dir, f"{area}_{tag}.pt")
            torch.save({
                'model_state': model.state_dict(),
                'input_size': X_scaled.shape[1],
                'lookback': lookback,
                'val_mae': val_mae,
                'model_type': model_name,
            }, save_path)

            print(f"  >> {tag} Val MAE: {val_mae:.2f} EUR/MWh  |  Saved: {save_path}", flush=True)
            results[tag] = {'model': model, 'val_mae': val_mae, 'lookback': lookback}

    # Pick best LSTM and best GRU
    best_lstm = min([k for k in results if 'lstm' in k], key=lambda k: results[k]['val_mae'])
    best_gru  = min([k for k in results if 'gru'  in k], key=lambda k: results[k]['val_mae'])
    print(f"\n  Best LSTM: {best_lstm} (MAE {results[best_lstm]['val_mae']:.2f})", flush=True)
    print(f"  Best GRU:  {best_gru}  (MAE {results[best_gru]['val_mae']:.2f})", flush=True)

    # Save meta
    meta = {
        'best_lstm': best_lstm,
        'best_gru': best_gru,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'imputer': imputer,
        'results': {k: {'val_mae': v['val_mae'], 'lookback': v['lookback']} for k, v in results.items()},
    }
    meta_path = os.path.join(save_dir, f'{area}_seq_meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"  Saved meta: {meta_path}", flush=True)

    return results, meta


def load_best_sequence_model(area='DK1', model_type='lstm', save_dir='models'):
    """Load the best LSTM or GRU model for live inference."""
    meta_path = os.path.join(save_dir, f'{area}_seq_meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    best_key = meta[f'best_{model_type}']
    info = meta['results'][best_key]
    lookback = info['lookback']

    pt_path = os.path.join(save_dir, f'{area}_{best_key}.pt')
    checkpoint = torch.load(pt_path, map_location=DEVICE, weights_only=False)

    model_class = ImbalanceLSTM if model_type == 'lstm' else ImbalanceGRU
    model = model_class(input_size=checkpoint['input_size']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    return model, lookback, meta['scaler'], meta['imputer'], meta['feature_cols']

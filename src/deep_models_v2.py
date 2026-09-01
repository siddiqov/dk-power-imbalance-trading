# ==============================================================================
# src/deep_models_v2.py
# PyTorch Deep Learning Sequence Models for Day-Ahead 96-Quarter Forecasting
#
# Architectures:
# 1. Bidirectional Seq2Seq LSTM / GRU (Multi-Horizon 96-Step Output)
# 2. Temporal Fusion Transformer / Multi-Head Attention Model
#    (Combines past history with KNOWN FUTURE Day-Ahead Spot profile)
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class BiSeq2SeqLSTM(nn.Module):
    """
    Bidirectional LSTM with Multi-Horizon 96-step Projection Head.
    Ingests past 96 quarters -> Projects all 96 future Day-Ahead quarters simultaneously.
    """

    def __init__(self, input_dim=22, hidden_dim=64, num_layers=2, output_horizon=96, dropout=0.2):
        super(BiSeq2SeqLSTM, self).__init__()
        self.output_horizon = output_horizon
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_horizon)
        )

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        lstm_out, (hn, cn) = self.lstm(x)
        # Use last timestep representation (forward + backward concatenated)
        last_out = lstm_out[:, -1, :]  # [batch_size, hidden_dim * 2]
        pred_96q = self.fc_head(last_out)  # [batch_size, 96]
        return pred_96q


class TemporalFusionAttentionModel(nn.Module):
    """
    Temporal Fusion Transformer / Cross-Attention Architecture.
    Explicitly conditions forecasts on KNOWN FUTURE Day-Ahead Spot prices & cyclicals.
    """

    def __init__(self, past_dim=22, future_dim=15, embed_dim=64, num_heads=4, output_horizon=96, dropout=0.2):
        super(TemporalFusionAttentionModel, self).__init__()
        self.output_horizon = output_horizon

        # Past Encoder
        self.past_proj = nn.Linear(past_dim, embed_dim)
        self.past_gru = nn.GRU(embed_dim, embed_dim, batch_first=True, bidirectional=True)

        # Future Known Encoder (Spot price profile + time cyclicals)
        self.future_proj = nn.Linear(future_dim, embed_dim * 2)

        # Cross Attention: Future queries attend to Past context keys/values
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim * 2, num_heads=num_heads, batch_first=True, dropout=dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # 1 output spread per future step
        )

    def forward(self, past_x, future_known_x):
        # past_x: [batch, past_len, past_dim]
        # future_known_x: [batch, 96, future_dim]

        # 1. Encode Past History
        p_emb = torch.relu(self.past_proj(past_x))
        p_out, _ = self.past_gru(p_emb)  # [batch, past_len, embed_dim * 2]

        # 2. Encode Known Future Spot & Cyclicals
        f_emb = torch.relu(self.future_proj(future_known_x))  # [batch, 96, embed_dim * 2]

        # 3. Cross Attention
        attn_out, _ = self.cross_attn(query=f_emb, key=p_out, value=p_out)
        f_combined = f_emb + attn_out

        # 4. Project to 96 Quarters of Spreads
        out = self.feed_forward(f_combined).squeeze(-1)  # [batch, 96]
        return out


class DeepSequenceTrainer:
    """Helper class to train and evaluate PyTorch deep models on tabular datasets."""

    def __init__(self, seq_len=96, horizon=96, epochs=6, lr=0.005, batch_size=64):
        self.seq_len = seq_len
        self.horizon = horizon
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_sequence_windows(self, X_arr, y_spread_arr):
        """Creates sliding windows of (past_x, future_y) with 96 steps each."""
        X_list, y_list = [], []
        # Sample most recent 2,500 steps (~4 weeks) for fast, responsive sequence training on CPU
        if len(X_arr) > 2500:
            X_arr = X_arr[-2500:]
            y_spread_arr = y_spread_arr[-2500:]

        n = len(X_arr)
        total_len = self.seq_len + self.horizon

        # Use stride of 24 (6 hours)
        for i in range(0, n - total_len + 1, 24):
            past_x = X_arr[i : i + self.seq_len]
            fut_y = y_spread_arr[i + self.seq_len : i + total_len]
            X_list.append(past_x)
            y_list.append(fut_y)

        if not X_list:
            return torch.zeros((1, self.seq_len, X_arr.shape[1])), torch.zeros((1, self.horizon))

        return torch.tensor(np.array(X_list), dtype=torch.float32), torch.tensor(np.array(y_list), dtype=torch.float32)

    def train_biseq2seq_lstm(self, X_train, y_train_spread, X_val_recent):
        """Trains BiSeq2SeqLSTM and predicts all 96 quarters for validation day."""
        X_ten, y_ten = self.create_sequence_windows(X_train.values, y_train_spread)
        dataset = TensorDataset(X_ten, y_ten)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = BiSeq2SeqLSTM(input_dim=X_train.shape[1], hidden_dim=64, output_horizon=self.horizon).to(self.device)
        criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)

        model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

        # Predict on most recent 96 steps preceding the validation day
        model.eval()
        with torch.no_grad():
            recent_in = torch.tensor(X_val_recent.values[-self.seq_len:], dtype=torch.float32).unsqueeze(0).to(self.device)
            pred_96 = model(recent_in).cpu().numpy().squeeze(0)

        return model, pred_96

    def train_tft_attention(self, X_train, y_train_spread, X_val_recent, X_val_future_known):
        """Trains TemporalFusionAttentionModel and predicts all 96 quarters for validation day."""
        X_ten, y_ten = self.create_sequence_windows(X_train.values, y_train_spread)

        # Future known inputs: subset of features available Day-Ahead (e.g. Spot + cyclicals)
        fut_dim = min(15, X_train.shape[1])
        fut_ten = X_ten[:, :, :fut_dim]  # Slice future known dimensions

        dataset = TensorDataset(X_ten, fut_ten, y_ten)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = TemporalFusionAttentionModel(past_dim=X_train.shape[1], future_dim=fut_dim, output_horizon=self.horizon).to(self.device)
        criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)

        model.train()
        for epoch in range(self.epochs):
            for b_past, b_fut, b_y in loader:
                b_past, b_fut, b_y = b_past.to(self.device), b_fut.to(self.device), b_y.to(self.device)
                optimizer.zero_grad()
                pred = model(b_past, b_fut)
                loss = criterion(pred, b_y)
                loss.backward()
                optimizer.step()

        # Predict on validation day using known future spot & cyclicals
        model.eval()
        with torch.no_grad():
            past_in = torch.tensor(X_val_recent.values[-self.seq_len:], dtype=torch.float32).unsqueeze(0).to(self.device)
            fut_in = torch.tensor(X_val_future_known.values[:self.horizon, :fut_dim], dtype=torch.float32).unsqueeze(0).to(self.device)
            pred_96 = model(past_in, fut_in).cpu().numpy().squeeze(0)

        return model, pred_96

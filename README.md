# Danish Power Imbalance Trading & Forecasting Engine (DK1 / DK2)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20PyTorch%20%7C%20CatBoost-orange.svg)]()
[![Market](https://img.shields.io/badge/Market-Danish%20Power%20(DK1%20%2F%20DK2)-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

An end-to-end quantitative trading and machine learning platform for predicting electricity imbalance prices and executing algorithmic trading strategies in the Danish power market (**DK1 & DK2 price zones**).

---

## 📌 Project Overview

In deregulated power grids such as the Nordic synchronized area, balance responsible parties (BRPs) face volatile imbalance settlement prices when real-time consumption and production deviate from day-ahead schedules.

This repository implements a multi-horizon price forecasting engine, risk-managed quantitative trading strategies, an automated backtesting framework, and an interactive real-time Flask analytics dashboard.

`	ext
                    ┌──────────────────────────────────────────────┐
                    │ Energinet / Nord Pool Ingestion API          │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Feature Engineering & Market Regimes (DB/CSV)│
                    └──────────────────────┬───────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐             ┌──────────────────┐
│  CatBoost / GBDT │             │   LSTM / GRU     │             │     Prophet      │
│  Multi-Horizon   │             │   Deep Learning  │             │  Decomposition   │
└────────┬─────────┘             └────────┬─────────┘             └────────┬─────────┘
         │                                │                                │
         └────────────────────────────────┼────────────────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Strategy Execution & Backtesting Engine      │
                    │ (Dual-Threshold, Volatility, Regime-Adaptive)│
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Interactive Flask Web Dashboard (UI/Charts)  │
                    └──────────────────────────────────────────────┘
`

---

## 🚀 Key Features

- **Automated Data Retrieval**: Ingests Day-Ahead prices, Imbalance prices, aFRR/mFRR balancing reserves, generation forecasts, and physical exchange flows from Energinet.
- **Multi-Model Forecasting Ensemble**:
  - **CatBoost & LightGBM**: Tabular gradient boosted decision trees with custom multi-horizon lag features.
  - **PyTorch LSTM & GRU**: Recurrent deep learning sequence architectures (24-step & 96-step rolling horizons).
  - **Facebook Prophet**: Seasonal and trend decomposition for baseline daily patterns.
- **Quantitative Trading Strategies**:
  - *Dual-Threshold Spread Arbitrage*
  - *Volatility-Adaptive Dynamic Bands*
  - *Regime-Filtered Mean-Reversion*
  - *Momentum & Imbalance Sign Prediction*
- **Backtesting & Risk Analytics**: Evaluates total P&L, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Win Rate, and Profit Factor across market regimes.
- **Interactive Web UI**: Real-time Flask dashboard visualizing intraday predictions, historical spreads, model parameter metrics, and ledger logs.

---

## 📂 Repository Structure

`	ext
dk-power-imbalance-trading/
├── data/                      # Historical datasets & CSV dumps
│   └── csv/                   # Ingested balancing & price datasets
├── models/                    # Model architecture scripts
│   ├── lstm_gru_models.py     # PyTorch LSTM / GRU architectures
│   └── prophet_model.py       # Prophet wrapper & utilities
├── scripts/
│   └── download_models.py     # Helper to download pre-trained weights from Google Drive
├── templates/
│   └── dashboard.html         # Interactive web dashboard template
├── app.py                     # Flask web server & visualization dashboard
├── data_retrieval.py          # Energinet API ingestion & database builder
├── energy_data.db             # Historical SQLite database (~34 MB)
├── fetch_full_history.py      # Batch historical ingestion utility
├── fetch_prodex_2026.py       # Real-time Prodex/Energinet pipeline
├── model_training.py          # GBDT & multi-model training pipeline
├── predict_future_prices.py   # Multi-horizon inference engine
├── run_backtest_pipeline.py   # Comprehensive strategy backtesting framework
├── trading_strategy.py        # Trading logic & portfolio simulator
├── train_all_models.py        # Master training orchestrator (GBDT + PyTorch + Prophet)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
`

---

## 🛠️ Quick Start

### 1. Clone & Setup Environment

`ash
git clone https://github.com/siddiqov/dk-power-imbalance-trading.git
cd dk-power-imbalance-trading

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
`

### 2. Download Pre-Trained Models (Google Drive)

Pre-trained model binaries (.pkl, .pt) are hosted on [Google Drive (Models Folder)](https://drive.google.com/drive/folders/1xDMe236gDmZ4jUcXXgWoFsdITAxMteWU?usp=sharing).

You can automatically download all trained weights into your local models/ directory:

`ash
python scripts/download_models.py
`

### 3. Run the Web Dashboard

`ash
python app.py
`
Open your browser and navigate to http://127.0.0.1:5000 to view the interactive dashboard.

### 4. Train Models & Run Backtesting

`ash
# Train all models (GBDT, LSTM, GRU, Prophet)
python train_all_models.py

# Run comprehensive strategy backtesting
python run_backtest_pipeline.py

# Generate multi-horizon forecasts
python predict_future_prices.py
`

---

## 📊 Models & Benchmarks

| Model | Architecture | Horizon | Primary Target |
|---|---|---|---|
| **CatBoost Regressor** | Gradient Boosted Decision Trees | Multi-Quarter / 5-Step | Imbalance Price (DK1/DK2) |
| **LightGBM / XGBoost** | Tree Ensemble | 24-Hour Horizon | Price Spread / Volatility |
| **PyTorch LSTM** | 2-Layer Bidirectional LSTM | 24h & 96h Steps | Sequence Residuals |
| **PyTorch GRU** | 2-Layer Gated Recurrent Unit | 24h & 96h Steps | Fast Sequence Forecasting |
| **Prophet** | Additive Decomposition | Trend & Day-of-Week | Baseline Seasonal Components |

---

## 📜 License

This project is licensed under the MIT License.

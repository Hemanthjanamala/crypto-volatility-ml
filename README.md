# Crypto Volatility & Price Forecasting (ARIMA • Prophet • LSTM) + FastAPI

MSc Data Science project focused on forecasting cryptocurrency closing prices and analysing volatility behaviour across calm vs turbulent regimes using a Kaggle OHLCV dataset.

**Student:** Hemanth Janamala  
**University:** University of Hertfordshire  
**GitHub:** [Crypto Volatility](https://github.com/Hemanthjanamala/crypto-volatility-ml.git)

---

## Project Goal

This project builds an end-to-end forecasting pipeline that:

1. **Forecasts next-day / short-horizon crypto closing prices** (RQ1)
2. **Compares models under different volatility regimes** (calm vs turbulent) (RQ2)
3. **Explains what drives predictions using engineered features and diagnostics** (RQ3)

The work is designed for **reproducibility** (Git + structured folders) and **stakeholder demonstration** (saved plots + FastAPI endpoints).

---

## Research Questions

- **RQ1:** Can we forecast next-day cryptocurrency closing prices using OHLCV and engineered features?
- **RQ2:** Which models (ARIMA, Prophet, LSTM) capture volatility patterns best, especially across calm vs turbulent regimes?
- **RQ3:** Which features (lags, volatility, momentum indicators) drive predictions, and can we make these models interpretable for stakeholders?

---

## Dataset

- Source: Kaggle ZIP containing multiple CSV files (typically one per coin).
- Locally combined into: data/raw/crypto_all_combined.csv
- Feature-engineered dataset saved as: data/processed/crypto_features.csv

**Core columns:**
Date, Open, High, Low, Close, Volume, Name, Symbol, SourceFile

---

## Features Engineered (examples)

- Log returns and returns %
- Rolling volatility (7d, 30d)
- Momentum (7d, 30d)
- RSI(14), MACD, EMA indicators
- Bollinger Band width
- Lag features (Close/Volume/Returns lags)
- Calendar features (DayOfWeek, Month, Quarter)

Feature engineering code: src/data/features.py

---

## Models Implemented

### 1) AutoARIMA
- Backtest with historical exogenous regressors
- Expanding walk-forward CV (1-step ahead)
- Produces backtest forecast + confidence intervals and a future forecast plot

### 2) Prophet
- Trend + seasonality + changepoints
- Rolling-origin evaluation design (time-based split + horizon)
- Produces forecast-vs-actual plot + residual plot

### 3) LSTM
- Sequence modelling with fixed lookback window
- Rolling-window / time-based evaluation to avoid leakage
- Produces forecast-vs-actual plot + residual diagnostics

---

## Project Structure

`	ext
crypto-volatility-ml/
│
├── data/
│   ├── raw/                          # Kaggle CSVs or extracted files
│   │   └── crypto_all_combined.csv   # combined output
│   └── processed/
│       ├── crypto_features.csv       # final feature dataset used by models
│       ├── predictions_*.csv         # saved model predictions (backtest/future)
│       └── split_indices.json        # optional: split metadata (if used)
│
├── notebooks/
│   ├── 00_eda.ipynb                  # EDA + finance-specific plots
│   ├── 01_modeling.ipynb             # model training/experiments
│   └── 02_results.ipynb              # comparison dashboard + plots
│
├── reports/
│   ├── figures/                      # ALL saved plots go here
│   └── report.md                     # final write-up (export to PDF)
│
├── src/
│   ├── api/
│   │   ├── main.py                   # FastAPI app entry
│   │   └── routers/
│   │       ├── arima_router.py
│   │       ├── prophet_router.py
│   │       ├── lstm_router.py
│   │       └── meta_router.py        # best model / best coin endpoints
│   │
│   ├── data/
│   │   ├── clean.py                  # cleaning + scaling (train-only where needed)
│   │   ├── features.py               # feature engineering
│   │   └── split.py                  # time-based split helpers (optional)
│   │
│   └── models/
│       ├── arima_forecast.py
│       ├── prophet_forecast.py
│       └── lstm_forecast.py   
├── requirements.txt
└── README.md
``

---

## Setup (Windows / PowerShell)

### 1) Create & activate virtual environment

`powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
`

### 2) Install dependencies

`powershell
pip install -r requirements.txt
`

---

## Build the Processed Dataset

1. Put your Kaggle CSV files inside:
   data/raw/

2. Run your combine + feature pipeline (whichever script/notebook you use):

* Option A: Run notebook 
otebooks/00_eda.ipynb (recommended)
* Option B: Run your pipeline script (if you created one)

Expected output:

* data/raw/crypto_all_combined.csv
* data/processed/crypto_features.csv

---

## Run Forecasting Models (as scripts)

Examples:

`powershell
python src/models/arima_forecast.py
python src/models/prophet_forecast.py
python src/models/lstm_forecast.py
`

All plots are saved to:

eports/figures/

All predictions are saved to:
data/processed/

---

## Run the FastAPI Service

### Start API

From project root:

`powershell
uvicorn src.api.main:app --reload
`

Open Swagger UI:

* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Endpoints (Core)

### Forecast endpoints

* POST /arima/forecast
* POST /prophet/forecast
* POST /lstm/forecast

Request body example:

`json
{
  "coin": "Bitcoin",
  "horizon": 7
}
`

### Meta endpoints (selection)

* GET /meta/best-model?coin=Bitcoin
* GET /meta/best-coin
* GET /meta/leaderboard

Selection logic uses saved evaluation metrics (MAE/RMSE) from model outputs.

---

## Outputs (What gets saved)

### Plots

Saved to 
eports/figures/, for example:

* ARIMA_Bitcoin_backtest.png
* ARIMA_Bitcoin_future.png
* Prophet_Ethereum_forecast.png
* LSTM_Litecoin_forecast.png
* residual plots for each model

### Prediction CSVs

Saved to data/processed/, for example:

* predictions_arima_backtest_Bitcoin.csv
* predictions_arima_future_Bitcoin.csv
* predictions_prophet_Ethereum.csv
* predictions_lstm_Litecoin.csv

---

## Notes on Validation (No Leakage)

This project uses **time-series validation only**:

* no shuffling
* train on past → validate on future
* walk-forward / rolling-origin where appropriate

---

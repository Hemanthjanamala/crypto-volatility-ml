# File: src/data/features.py

import pandas as pd
import numpy as np

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate engineered OHLCV + technical indicators for crypto forecasting.
    Safe for Prophet regressors, LSTM multivariate inputs, and ML models.
    """

    df = df.copy()

    # --- 1) Cleanup ---
    if "index" in df.columns:
        df = df.drop(columns=["index"])

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    if "Date" not in df.columns:
        df = df.reset_index()

    required_cols = ["Name", "Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Name"] = df["Name"].astype(str)

    df = df.sort_values(["Name", "Date"]).reset_index(drop=True)

    # Helper function for grouped transforms
    def by_group(col, func):
        return df.groupby("Name", group_keys=False)[col].transform(func)

    # -----------------------------------------------
    # 2) Log returns + volatility
    # -----------------------------------------------
    df["LogReturn"] = by_group("Close", lambda x: np.log(x).diff())
    df["Return_%"] = by_group("Close", lambda x: x.pct_change())

    df["Volatility_7d"] = by_group("LogReturn", lambda x: x.rolling(7, min_periods=1).std())
    df["Volatility_30d"] = by_group("LogReturn", lambda x: x.rolling(30, min_periods=1).std())

    # -----------------------------------------------
    # 3) Momentum indicators
    # -----------------------------------------------
    df["Momentum_7d"] = by_group("Close", lambda x: x.diff(7))
    df["Momentum_30d"] = by_group("Close", lambda x: x.diff(30))

    # -----------------------------------------------
    # 4) RSI (Relative Strength Index)
    # -----------------------------------------------
    def compute_rsi(price_series, period=14):
        delta = price_series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period, min_periods=1).mean()
        avg_loss = loss.rolling(period, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    df["RSI_14"] = by_group("Close", lambda x: compute_rsi(x))

    # -----------------------------------------------
    # 5) MACD (12, 26) + Signal (9)
    # -----------------------------------------------
    def ema(series, span):
        return series.rolling(span, min_periods=1).mean()

    df["EMA_12"] = by_group("Close", lambda x: x.ewm(span=12, adjust=False).mean())
    df["EMA_26"] = by_group("Close", lambda x: x.ewm(span=26, adjust=False).mean())
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = by_group("MACD", lambda x: x.ewm(span=9, adjust=False).mean())

    # -----------------------------------------------
    # 6) EMA trend indicators (10, 20, 50)
    # -----------------------------------------------
    for span in [10, 20, 50]:
        df[f"EMA_{span}"] = by_group("Close", lambda x: x.ewm(span=span, adjust=False).mean())

    # -----------------------------------------------
    # 7) Bollinger Bands (20-day)
    # -----------------------------------------------
    rolling_mean = by_group("Close", lambda x: x.rolling(20, min_periods=1).mean())
    rolling_std = by_group("Close", lambda x: x.rolling(20, min_periods=1).std())

    df["BB_Upper"] = rolling_mean + 2 * rolling_std
    df["BB_Lower"] = rolling_mean - 2 * rolling_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / rolling_mean

    # -----------------------------------------------
    # 8) Price range and candle shape features
    # -----------------------------------------------
    df["High_Low_%"] = (df["High"] - df["Low"]) / df["Close"]
    df["Close_Open_%"] = (df["Close"] - df["Open"]) / df["Open"]
    df["MarketPressure"] = df["Close"] - df["Open"]

    # -----------------------------------------------
    # 9) Lag features (to prevent leakage)
    # -----------------------------------------------
    for lag in [1, 7, 14, 30]:
        df[f"Close_lag{lag}"] = by_group("Close", lambda x: x.shift(lag))
        df[f"Volume_lag{lag}"] = by_group("Volume", lambda x: x.shift(lag))
        df[f"Return_lag{lag}"] = by_group("Return_%", lambda x: x.shift(lag))

    # -----------------------------------------------
    # 10) Time-based features
    # -----------------------------------------------
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter

    return df

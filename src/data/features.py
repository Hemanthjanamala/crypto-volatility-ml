# File: src/data/features.py

import pandas as pd
import numpy as np

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate engineered features for crypto OHLCV data safely.
    Handles multi-indexed inputs, stray index columns, and ensures alignment.
    """
    df = df.copy()

    # --- 1️ Sanity cleanup ---
    # Drop leftover "index" column from prior resets
    if "index" in df.columns:
        df = df.drop(columns=["index"])

    # Reset MultiIndex if present
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Reset regular index if "Date" isn't a column
    if "Date" not in df.columns:
        df = df.reset_index()

    # Ensure mandatory columns
    required_cols = ["Name", "Date", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- 2️ Standardize formats ---
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Name"] = df["Name"].astype(str)

    # Sort by coin and date
    df = df.sort_values(["Name", "Date"]).reset_index(drop=True)

    # --- 3️ Define helper for safe grouped transform ---
    def by_group(col, func):
        """Apply transform per crypto name safely"""
        return (
            df.groupby("Name", group_keys=False)[col]
            .transform(func)
        )

    # --- 4️ Compute engineered features ---
    # Log returns
    df["LogReturn"] = by_group("Close", lambda x: np.log(x).diff())

    # Rolling volatility
    df["Volatility_7d"] = by_group("LogReturn", lambda x: x.rolling(7, min_periods=1).std())
    df["Volatility_30d"] = by_group("LogReturn", lambda x: x.rolling(30, min_periods=1).std())

    # Momentum (rolling mean)
    df["Momentum_7d"] = by_group("LogReturn", lambda x: x.rolling(7, min_periods=1).mean())
    df["Momentum_30d"] = by_group("LogReturn", lambda x: x.rolling(30, min_periods=1).mean())

    # Lag features
    for lag in [1, 7, 30]:
        df[f"Close_lag{lag}"] = by_group("Close", lambda x: x.shift(lag))
        df[f"Volume_lag{lag}"] = by_group("Volume", lambda x: x.shift(lag))

    # Bollinger Band width
    rolling_mean = by_group("Close", lambda x: x.rolling(20, min_periods=1).mean())
    rolling_std = by_group("Close", lambda x: x.rolling(20, min_periods=1).std())
    df["Bollinger_Width"] = (rolling_std * 2) / rolling_mean

    # Time-based features
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month

    return df




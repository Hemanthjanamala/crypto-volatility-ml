# File: src/data/clean.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_and_scale(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Cleans and scales features safely without dropping all rows.
    Steps:
    1. Drop fully-NaN MarketCap
    2. Fill NaNs per-coin median, then global median
    3. Scale numeric features
    """
    df = df.copy()
    print("Starting cleaning...")

    # --- 1️ Drop MarketCap if useless ---
    if "MarketCap" in df.columns and df["MarketCap"].isna().all():
        df = df.drop(columns=["MarketCap"])
        print("Dropped MarketCap (all NaN)")

    # --- 2️ Fill missing values ---
    # Fill per-coin (grouped median)
    df[feature_cols] = df.groupby("Name")[feature_cols].transform(lambda x: x.fillna(x.median()))
    # Fill any leftover NaNs with global medians
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    # --- 3️ Check missingness ---
    missing_total = df[feature_cols].isna().sum().sum()
    print(f"Remaining missing values after fill: {missing_total}")

    # --- 4️ Prepare X, y ---
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    print("X shape before scaling:", X.shape)
    print("Sample rows:")
    print(X.head(3))

    if X.empty:
        raise ValueError("X is empty after cleaning — likely all rows dropped due to NaNs.")

    # --- 5️ Scale ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)

    # --- 6️ Merge scaled data back ---
    processed = pd.concat([df[["Name", "Date"]], X_scaled, y], axis=1)
    print("Processed shape:", processed.shape)

    return processed, scaler

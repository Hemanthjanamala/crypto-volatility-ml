# src/models/lstm_forecast.py
from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # reduce TensorFlow noise (INFO/WARN)

from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# -----------------------------
# Paths / constants
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "crypto_features.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Close"
LOOKBACK = 60
HISTORY_WINDOW = 300


# -----------------------------
# Helpers
# -----------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_coin(coin: str) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values(["Name", "Date"])
    sub = df[df["Name"].str.lower() == coin.lower()].copy()

    if sub.empty:
        raise ValueError(f"Coin not found in dataset: {coin}")

    sub = sub[["Date", TARGET]].dropna().sort_values("Date").reset_index(drop=True)

    min_needed = LOOKBACK + 50
    if len(sub) < min_needed:
        raise ValueError(f"Not enough rows for {coin}. Need >= {min_needed}, got {len(sub)}")

    return sub


def make_sequences(series_1d: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    series_1d: shape (n,)
    returns:
      X: (n-lookback, lookback, 1)
      y: (n-lookback,)
    """
    X, y = [], []
    for i in range(lookback, len(series_1d)):
        X.append(series_1d[i - lookback:i])
        y.append(series_1d[i])
    X = np.array(X, dtype=np.float32)[..., None]
    y = np.array(y, dtype=np.float32)
    return X, y


def build_model(lookback: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.25),
        LSTM(32),
        Dropout(0.25),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def mc_dropout_predict(model: tf.keras.Model, X: np.ndarray, n: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo dropout: run model in training=True mode multiple times.
    returns mean, std in *scaled* space.
    """
    preds = []
    for _ in range(n):
        preds.append(model(X, training=True).numpy().flatten())
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)


def scaled_std_to_original(std_scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Correct way to convert std from MinMax scaled space back to original scale.
    MinMax: x_scaled = (x - min) / (max - min)
    => std_orig = std_scaled * (max - min)
    """
    scale = float(scaler.data_max_[0] - scaler.data_min_[0])
    return std_scaled * scale


# -----------------------------
# Plots
# -----------------------------
def save_backtest_plot(dates, y_true, y_pred, lo, hi, split_date, coin: str) -> str:
    path = FIG_DIR / f"LSTM_{coin}_backtest.png"
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label="Actual", linewidth=2)
    plt.plot(dates, y_pred, label="Forecast", linewidth=2)
    plt.fill_between(dates, lo, hi, alpha=0.25, label="95% (MC Dropout)")
    plt.axvline(split_date, linestyle="--", color="black", alpha=0.6, label="Train/Test split")
    plt.title(f"LSTM Backtest Forecast — {coin}")
    plt.xlabel("Date")
    plt.ylabel(TARGET)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return str(path)


def save_loss_plot(history: tf.keras.callbacks.History, coin: str) -> str:
    path = FIG_DIR / f"LSTM_{coin}_loss.png"
    plt.figure(figsize=(10, 4))
    plt.plot(history.history.get("loss", []), label="train")
    plt.plot(history.history.get("val_loss", []), label="val")
    plt.title(f"LSTM Loss Curve — {coin}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return str(path)


def save_future_plot(hist_dates, hist_close, future_dates, future_pred, coin: str) -> str:
    path = FIG_DIR / f"LSTM_{coin}_future.png"
    plt.figure(figsize=(12, 6))
    plt.plot(hist_dates, hist_close, label="History", linewidth=2)
    plt.plot(future_dates, future_pred, label="Future Forecast", linewidth=2)
    plt.axvline(hist_dates.iloc[-1], linestyle="--", color="black", alpha=0.6, label="Last observed")
    plt.title(f"LSTM Future Forecast — {coin}")
    plt.xlabel("Date")
    plt.ylabel(TARGET)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return str(path)


# -----------------------------
# Time-series CV (walk-forward)
# -----------------------------
def lstm_walk_forward_cv(
    close_values: np.ndarray,
    folds: int = 3,
    train_size: int = 1500,
    val_size: int = 250,
    epochs: int = 25,
    batch_size: int = 32
) -> Dict:
    """
    Rolling walk-forward CV (no leakage):
    - fit scaler on each training fold
    - train a fresh model per fold
    - evaluate on the next val block
    """
    n = len(close_values)
    min_required = train_size + folds * val_size + LOOKBACK + 10
    if n < min_required:
        # Don't hard fail: return "not enough data" gracefully
        return {
            "cv_type": "rolling_walk_forward",
            "folds": folds,
            "train_size": train_size,
            "val_size": val_size,
            "cv_mae": None,
            "cv_rmse": None,
            "note": f"Not enough data for CV. Need >= {min_required}, got {n}"
        }

    maes, rmses = [], []
    start0 = n - (train_size + folds * val_size)

    for i in range(folds):
        s = start0 + i * val_size
        train = close_values[s: s + train_size]
        val = close_values[s + train_size: s + train_size + val_size]

        scaler = MinMaxScaler()
        train_s = scaler.fit_transform(train.reshape(-1, 1)).flatten()

        Xtr, ytr = make_sequences(train_s, LOOKBACK)

        # Build validation sequences with lookback context
        context = close_values[s + train_size - LOOKBACK: s + train_size + val_size]
        context_s = scaler.transform(context.reshape(-1, 1)).flatten()
        Xv, yv = make_sequences(context_s, LOOKBACK)

        model = build_model(LOOKBACK)
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(
            Xtr, ytr,
            validation_data=(Xv, yv),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[es]
        )

        mean_pred_s, _ = mc_dropout_predict(model, Xv, n=20)

        y_pred = scaler.inverse_transform(mean_pred_s.reshape(-1, 1)).flatten()
        y_true = scaler.inverse_transform(yv.reshape(-1, 1)).flatten()

        maes.append(mean_absolute_error(y_true, y_pred))
        rmses.append(rmse(y_true, y_pred))

    return {
        "cv_type": "rolling_walk_forward",
        "folds": folds,
        "train_size": train_size,
        "val_size": val_size,
        "cv_mae": float(np.mean(maes)),
        "cv_rmse": float(np.mean(rmses))
    }


# -----------------------------
# Main forecasting routine
# -----------------------------
def lstm_forecast_coin(
    coin: str,
    horizon: int = 7,
    test_size: float = 0.2
) -> Dict:
    sub = load_coin(coin)
    n = len(sub)
    split = int((1 - test_size) * n)

    if split <= LOOKBACK:
        raise ValueError(f"Split too early for lookback. split={split}, lookback={LOOKBACK}")

    train = sub.iloc[:split].copy()
    test = sub.iloc[split:].copy()

    # -------- Backtest model (train -> test) --------
    scaler_bt = MinMaxScaler()
    train_s = scaler_bt.fit_transform(train[[TARGET]].values).flatten()

    Xtr, ytr = make_sequences(train_s, LOOKBACK)

    # test sequences with context
    context = sub.iloc[split - LOOKBACK:][[TARGET]].values.flatten()
    context_s = scaler_bt.transform(context.reshape(-1, 1)).flatten()
    Xte, yte = make_sequences(context_s, LOOKBACK)

    model_bt = build_model(LOOKBACK)
    es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

    history = model_bt.fit(
        Xtr, ytr,
        validation_data=(Xte, yte),
        epochs=40,
        batch_size=32,
        verbose=0,
        callbacks=[es]
    )

    mean_pred_s, std_pred_s = mc_dropout_predict(model_bt, Xte, n=40)

    y_pred = scaler_bt.inverse_transform(mean_pred_s.reshape(-1, 1)).flatten()
    y_true = scaler_bt.inverse_transform(yte.reshape(-1, 1)).flatten()

    std_orig = scaled_std_to_original(std_pred_s, scaler_bt)
    lo = y_pred - 1.96 * std_orig
    hi = y_pred + 1.96 * std_orig

    # train metrics (use same model + train sequences)
    mean_train_s, _ = mc_dropout_predict(model_bt, Xtr, n=20)
    y_train_pred = scaler_bt.inverse_transform(mean_train_s.reshape(-1, 1)).flatten()
    y_train_true = scaler_bt.inverse_transform(ytr.reshape(-1, 1)).flatten()

    metrics = {
        "train_mae": float(mean_absolute_error(y_train_true, y_train_pred)),
        "train_rmse": rmse(y_train_true, y_train_pred),
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_rmse": rmse(y_true, y_pred),
    }

    # Backtest CSV (aligned exactly to test dates)
    back_csv = OUT_DIR / f"predictions_lstm_backtest_{coin}.csv"
    pd.DataFrame({
        "Date": test["Date"].values[: len(y_pred)],
        "y_true": y_true[: len(y_pred)],
        "y_pred": y_pred[: len(y_pred)],
        "ci_low": lo[: len(y_pred)],
        "ci_high": hi[: len(y_pred)]
    }).to_csv(back_csv, index=False)

    # Backtest plot (history window + test)
    hist_start = max(0, split - HISTORY_WINDOW)
    hist_dates = sub["Date"].iloc[hist_start:split]
    hist_close = sub[TARGET].iloc[hist_start:split]

    plot_dates = np.concatenate([hist_dates.values, test["Date"].values[: len(y_pred)]])
    plot_true = np.concatenate([hist_close.values, y_true[: len(y_pred)]])
    plot_pred = np.concatenate([np.full(len(hist_close), np.nan), y_pred[: len(y_pred)]])
    plot_lo = np.concatenate([np.full(len(hist_close), np.nan), lo[: len(y_pred)]])
    plot_hi = np.concatenate([np.full(len(hist_close), np.nan), hi[: len(y_pred)]])

    back_plot = save_backtest_plot(
        plot_dates, plot_true, plot_pred, plot_lo, plot_hi,
        split_date=sub["Date"].iloc[split],
        coin=coin
    )

    loss_plot = save_loss_plot(history, coin)

    # -------- CV (best practice for time series) --------
    cv = lstm_walk_forward_cv(sub[TARGET].values, folds=3, train_size=1500, val_size=250)

    # -------- Future forecast: retrain on FULL data (no scaler mismatch) --------
    scaler_full = MinMaxScaler()
    full_s = scaler_full.fit_transform(sub[[TARGET]].values).flatten()
    Xfull, yfull = make_sequences(full_s, LOOKBACK)

    model_full = build_model(LOOKBACK)
    es2 = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model_full.fit(Xfull, yfull, epochs=30, batch_size=32, verbose=0, callbacks=[es2])

    last_seq = full_s[-LOOKBACK:].copy()
    future_scaled: List[float] = []

    for _ in range(horizon):
        x = last_seq.reshape(1, LOOKBACK, 1)
        nxt = float(model_full.predict(x, verbose=0)[0, 0])
        future_scaled.append(nxt)
        last_seq = np.concatenate([last_seq[1:], [nxt]])

    future_pred = scaler_full.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
    last_date = sub["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    future_csv = OUT_DIR / f"predictions_lstm_future_{coin}.csv"
    pd.DataFrame({"Date": future_dates, "y_pred": future_pred}).to_csv(future_csv, index=False)

    future_plot = save_future_plot(
        sub["Date"].iloc[-HISTORY_WINDOW:],
        sub[TARGET].iloc[-HISTORY_WINDOW:],
        future_dates,
        future_pred,
        coin
    )

    return {
        "coin": coin,
        "model": "LSTM_MC_Dropout",
        "metrics": metrics,
        "cross_validation": cv,
        "artifacts": {
            "backtest_plot": back_plot,
            "loss_plot": loss_plot,
            "backtest_predictions_csv": str(back_csv),
            "future_plot": future_plot,
            "future_predictions_csv": str(future_csv)
        }
    }


# -----------------------------
# Script entry
# -----------------------------
if __name__ == "__main__":
    coins = ["Bitcoin", "Ethereum", "Litecoin"]
    rows = []

    for coin in coins:
        try:
            out = lstm_forecast_coin(coin, horizon=7)
            m = out["metrics"]
            cv = out["cross_validation"]

            print(f"\n=== {coin} ===")
            print(f"Train MAE : {m['train_mae']:.6f} | Train RMSE: {m['train_rmse']:.6f}")
            print(f"Test  MAE : {m['test_mae']:.6f} | Test  RMSE: {m['test_rmse']:.6f}")

            if cv.get("cv_mae") is None:
                print(f"CV: skipped ({cv.get('note')})")
            else:
                print(
                    f"CV ({cv['cv_type']}, train_size={cv['train_size']}, val_size={cv['val_size']}) "
                    f"MAE: {cv['cv_mae']:.6f} | RMSE: {cv['cv_rmse']:.6f}"
                )

            rows.append({
                "coin": coin,
                **m,
                "cv_mae": cv.get("cv_mae"),
                "cv_rmse": cv.get("cv_rmse"),
                "cv_train_size": cv.get("train_size"),
                "cv_val_size": cv.get("val_size"),
            })

        except Exception as e:
            print(f"\n!!! {coin} failed: {e}")

    if rows:
        summary_path = OUT_DIR / "metrics_lstm_summary.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"\nSaved summary: {summary_path}")

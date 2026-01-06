# src/models/prophet_forecast.py
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
HISTORY_WINDOW = 300

REGRESSORS: List[str] = [
    "Volatility_7d", "Volatility_30d", "RSI_14", "BB_Width",
    "Close_lag1", "Close_lag7", "Close_lag14", "Close_lag30",
    "Volume_lag1", "Volume_lag7", "Volume_lag14", "Volume_lag30",
    "DayOfWeek", "Month"
]


# -----------------------------
# Helpers
# -----------------------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def add_log_target_safe(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Safer than log(): use log1p() and drop non-positive prices.
    This prevents -inf which often causes downstream numpy/matplotlib crashes.
    """
    df = df.copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df[df[price_col] > 0].copy()  # critical for log transforms
    df["y_log"] = np.log1p(df[price_col].astype(float))  # log(1 + price)
    return df


def load_coin(coin: str) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values(["Name", "Date"])
    sub = df[df["Name"].str.lower() == coin.lower()].copy()

    if sub.empty:
        raise ValueError(f"Coin not found: {coin}")

    needed = ["Date", TARGET] + REGRESSORS
    missing = [c for c in needed if c not in sub.columns]
    if missing:
        raise ValueError(f"Missing columns for Prophet: {missing}")

    sub = sub[needed].copy()

    # Force numerics (important)
    sub[TARGET] = pd.to_numeric(sub[TARGET], errors="coerce")
    for r in REGRESSORS:
        sub[r] = pd.to_numeric(sub[r], errors="coerce")

    sub = sub.dropna().sort_values("Date").reset_index(drop=True)

    min_rows = 600
    if len(sub) < min_rows:
        raise ValueError(f"Not enough rows for {coin}: {len(sub)} (need >= {min_rows})")

    return sub


def _assert_no_sequences(df: pd.DataFrame, cols: List[str], context: str) -> None:
    """
    Hard guard: if ANY cell contains list/tuple/ndarray, Prophet/numpy will crash later.
    """
    bad = []
    for c in cols:
        # sample-based check (fast) + full check fallback if needed
        s = df[c].head(200)
        if s.apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
            bad.append(c)
    if bad:
        raise ValueError(f"[{context}] Found sequence values in columns: {bad}")


# -----------------------------
# Plots
# -----------------------------
def save_prophet_plot(dates, y_true, yhat, lo, hi, split_date, coin: str, name: str, ylabel: str) -> str:
    path = FIG_DIR / f"Prophet_{coin}_{name}.png"
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label="Actual", linewidth=2)
    plt.plot(dates, yhat, label="Forecast", linewidth=2)
    plt.fill_between(dates, lo, hi, alpha=0.25, label="95% Uncertainty")
    plt.axvline(split_date, linestyle="--", color="black", alpha=0.6, label="Train/Test split")
    plt.title(f"Prophet Forecast — {coin}")
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return str(path)


def save_residual_plot(dates, residuals, coin: str) -> str:
    path = FIG_DIR / f"Prophet_{coin}_residuals.png"
    plt.figure(figsize=(12, 4))
    plt.plot(dates, residuals, linewidth=1.5)
    plt.axhline(0, linestyle="--", color="black", alpha=0.6)
    plt.title(f"Prophet Residuals — {coin}")
    plt.xlabel("Date")
    plt.ylabel("Residual (log1p scale)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return str(path)


# -----------------------------
# CV
# -----------------------------
def prophet_rolling_origin_cv(
    model: Prophet,
    horizon_days: int = 30,
    initial: str = "730 days",
    period: str = "120 days",
) -> Dict:
    """
    Rolling-origin CV. Windows multiprocessing can be flaky, so default to parallel=None.
    """
    try:
        cv_df = cross_validation(
            model,
            initial=initial,
            period=period,
            horizon=f"{horizon_days} days",
            parallel=None,
        )
    except Exception:
        # fallback (some setups prefer processes)
        cv_df = cross_validation(
            model,
            initial=initial,
            period=period,
            horizon=f"{horizon_days} days",
            parallel="processes",
        )

    perf = performance_metrics(cv_df)
    return {
        "cv_type": "rolling_origin",
        "initial": initial,
        "period": period,
        "horizon_days": horizon_days,
        "cv_mae": float(perf["mae"].mean()),
        "cv_rmse": float(perf["rmse"].mean()),
        "cv_mape": float(perf["mape"].mean()),
    }


# -----------------------------
# Forecast
# -----------------------------
def prophet_forecast_coin(coin: str, horizon: int = 7, test_size: float = 0.2) -> Dict:
    sub = load_coin(coin)
    sub = add_log_target_safe(sub, price_col=TARGET)

    # Prophet expects: ds, y
    dfp = sub.rename(columns={"Date": "ds"}).copy()
    dfp["ds"] = pd.to_datetime(dfp["ds"], errors="coerce")
    dfp["y"] = pd.to_numeric(dfp["y_log"], errors="coerce")

    # Force all regressors float (avoid int corner issues)
    for r in REGRESSORS:
        dfp[r] = pd.to_numeric(dfp[r], errors="coerce").astype(float)

    dfp = dfp.dropna(subset=["ds", "y"] + REGRESSORS).reset_index(drop=True)

    # Defensive check: no list/array values in regressors
    _assert_no_sequences(dfp, ["y"] + REGRESSORS, context=f"{coin}/dfp")

    n = len(dfp)
    split = int((1 - test_size) * n)
    if split < 200:
        raise ValueError(f"Train split too small ({split} rows). Increase data or reduce test_size.")

    train = dfp.iloc[:split].copy()
    test = dfp.iloc[split:].copy()

    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    for r in REGRESSORS:
        m.add_regressor(r)

    m.fit(train[["ds", "y"] + REGRESSORS])

    # ---- Train/Test predictions (log1p space) ----
    fc_train = m.predict(train[["ds"] + REGRESSORS])
    fc_test = m.predict(test[["ds"] + REGRESSORS])

    # Convert back to price scale
    y_train_true = np.expm1(train["y"].to_numpy(dtype=float))
    y_train_pred = np.expm1(fc_train["yhat"].to_numpy(dtype=float))

    y_test_true = np.expm1(test["y"].to_numpy(dtype=float))
    y_test_pred = np.expm1(fc_test["yhat"].to_numpy(dtype=float))

    metrics = {
        "train_mae": float(mean_absolute_error(y_train_true, y_train_pred)),
        "train_rmse": rmse(y_train_true, y_train_pred),
        "test_mae": float(mean_absolute_error(y_test_true, y_test_pred)),
        "test_rmse": rmse(y_test_true, y_test_pred),
    }

    # ---- Save backtest CSV (price scale) ----
    back_csv = OUT_DIR / f"predictions_prophet_backtest_{coin}.csv"
    pd.DataFrame({
        "Date": test["ds"].to_numpy(),
        "y_true": y_test_true,
        "y_pred": y_test_pred,
        "lo": np.expm1(fc_test["yhat_lower"].to_numpy(dtype=float)),
        "hi": np.expm1(fc_test["yhat_upper"].to_numpy(dtype=float)),
        "y_true_log1p": test["y"].to_numpy(dtype=float),
        "y_pred_log1p": fc_test["yhat"].to_numpy(dtype=float),
    }).to_csv(back_csv, index=False)

    # ---- Backtest plot window (price scale) ----
    plot_start = max(0, split - HISTORY_WINDOW)
    plot_df = dfp.iloc[plot_start:].copy()
    plot_fc = m.predict(plot_df[["ds"] + REGRESSORS])

    y_true_plot = np.expm1(plot_df["y"].to_numpy(dtype=float))

    # Build forecast arrays safely (avoid np.where + datetime issues)
    yhat_all = np.expm1(plot_fc["yhat"].to_numpy(dtype=float))
    lo_all = np.expm1(plot_fc["yhat_lower"].to_numpy(dtype=float))
    hi_all = np.expm1(plot_fc["yhat_upper"].to_numpy(dtype=float))

    yhat_plot = np.full(len(plot_df), np.nan, dtype=float)
    lo_plot = np.full(len(plot_df), np.nan, dtype=float)
    hi_plot = np.full(len(plot_df), np.nan, dtype=float)

    split_in_plot = split - plot_start
    if split_in_plot < 0:
        split_in_plot = 0

    yhat_plot[split_in_plot:] = yhat_all[split_in_plot:]
    lo_plot[split_in_plot:] = lo_all[split_in_plot:]
    hi_plot[split_in_plot:] = hi_all[split_in_plot:]

    back_plot = save_prophet_plot(
        dates=plot_df["ds"].to_numpy(),
        y_true=y_true_plot,
        yhat=yhat_plot,
        lo=lo_plot,
        hi=hi_plot,
        split_date=dfp["ds"].iloc[split],
        coin=coin,
        name="backtest",
        ylabel="Close",
    )

    resid_plot = save_residual_plot(
        dates=test["ds"].to_numpy(),
        residuals=(test["y"].to_numpy(dtype=float) - fc_test["yhat"].to_numpy(dtype=float)),
        coin=coin,
    )

    # ---- Rolling-origin CV (log space) ----
    cv = prophet_rolling_origin_cv(m, horizon_days=30)

    # ---- Future forecast (hold regressors constant) ----
    last_row = dfp.iloc[-1]
    future_dates = pd.date_range(dfp["ds"].iloc[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = pd.DataFrame({"ds": future_dates})
    for r in REGRESSORS:
        future[r] = float(last_row[r])

    future_fc = m.predict(future)

    future_csv = OUT_DIR / f"predictions_prophet_future_{coin}.csv"
    pd.DataFrame({
        "Date": future_fc["ds"].to_numpy(),
        "y_pred": np.expm1(future_fc["yhat"].to_numpy(dtype=float)),
        "lo": np.expm1(future_fc["yhat_lower"].to_numpy(dtype=float)),
        "hi": np.expm1(future_fc["yhat_upper"].to_numpy(dtype=float)),
        "y_pred_log1p": future_fc["yhat"].to_numpy(dtype=float),
    }).to_csv(future_csv, index=False)

    # Future plot
    hist = dfp.tail(HISTORY_WINDOW)
    path_future = FIG_DIR / f"Prophet_{coin}_future.png"

    plt.figure(figsize=(12, 6))
    plt.plot(hist["ds"], np.expm1(hist["y"]), label="History", linewidth=2)
    plt.plot(future_fc["ds"], np.expm1(future_fc["yhat"]), label="Future Forecast", linewidth=2)
    plt.fill_between(
        future_fc["ds"].to_numpy(),
        np.expm1(future_fc["yhat_lower"].to_numpy(dtype=float)),
        np.expm1(future_fc["yhat_upper"].to_numpy(dtype=float)),
        alpha=0.25,
        label="95% Uncertainty",
    )
    plt.axvline(hist["ds"].iloc[-1], linestyle="--", color="black", alpha=0.6, label="Last observed")
    plt.title(f"Prophet Future Forecast — {coin}")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_future, dpi=300)
    plt.close()

    return {
        "coin": coin,
        "model": "Prophet_Log1pClose_Regressors",
        "cross_validation": cv,
        "metrics": metrics,
        "artifacts": {
            "backtest_plot": back_plot,
            "residual_plot": resid_plot,
            "backtest_predictions_csv": str(back_csv),
            "future_plot": str(path_future),
            "future_predictions_csv": str(future_csv),
            "future_regressors_note": "Future regressors held constant at last observed value (no external data).",
            "target_note": "Model trained on log1p(Close); predictions converted back using expm1().",
        },
    }


# -----------------------------
# Script entry
# -----------------------------
if __name__ == "__main__":
    coins = ["Bitcoin", "Ethereum", "Litecoin"]
    rows = []

    for coin in coins:
        try:
            out = prophet_forecast_coin(coin, horizon=7)
            m = out["metrics"]
            cv = out["cross_validation"]

            print(f"\n=== {coin} ===")
            print(f"Train MAE : {m['train_mae']:.6f} | Train RMSE: {m['train_rmse']:.6f}")
            print(f"Test  MAE : {m['test_mae']:.6f} | Test  RMSE: {m['test_rmse']:.6f}")
            print(
                f"CV ({cv['cv_type']}, initial={cv['initial']}, period={cv['period']}, horizon={cv['horizon_days']}d) "
                f"MAE: {cv['cv_mae']:.6f} | RMSE: {cv['cv_rmse']:.6f} | MAPE: {cv['cv_mape']:.6f}"
            )

            rows.append({
                "coin": coin,
                "train_mae": m["train_mae"],
                "train_rmse": m["train_rmse"],
                "test_mae": m["test_mae"],
                "test_rmse": m["test_rmse"],
                "cv_mae": cv["cv_mae"],
                "cv_rmse": cv["cv_rmse"],
                "cv_mape": cv["cv_mape"],
                "cv_initial": cv["initial"],
                "cv_period": cv["period"],
                "cv_horizon_days": cv["horizon_days"],
            })

        except Exception as e:
            print(f"\n!!! {coin} failed: {e}")

    if rows:
        summary_path = OUT_DIR / "metrics_prophet_summary.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"\nSaved summary: {summary_path}")

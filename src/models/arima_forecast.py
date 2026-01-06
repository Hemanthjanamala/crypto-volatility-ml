# src/models/arima_forecast.py
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------- Paths / Config --------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "crypto_features.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Close"
HISTORY_WINDOW = 300

EXOG_BACKTEST = [
    "Volatility_7d", "Volatility_30d", "RSI_14", "BB_Width",
    "Close_lag1", "Close_lag7", "Close_lag14", "Close_lag30",
    "Volume_lag1", "Volume_lag7", "Volume_lag14", "Volume_lag30",
    "DayOfWeek", "Month"
]

AUTO_ARIMA_KWARGS = dict(
    seasonal=False,
    start_p=0, start_q=0, min_p=0, min_q=0,
    max_p=5, max_q=5,
    max_order=10,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
)

# -------------------- Utils --------------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_coin(coin: str) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values(["Name", "Date"])
    sub = df[df["Name"].str.lower() == coin.lower()].copy()
    if sub.empty:
        raise ValueError(f"Coin not found in Name column: {coin}")

    required = ["Date", TARGET] + EXOG_BACKTEST
    missing = [c for c in required if c not in sub.columns]
    if missing:
        raise ValueError(f"Missing required columns for ARIMA: {missing}")

    sub = sub[required].dropna().sort_values("Date").reset_index(drop=True)
    if len(sub) < 600:
        raise ValueError(f"Not enough rows for {coin}: {len(sub)}")

    return sub


def save_plot(path: Path, title: str, x, y_lines: list[tuple], vline=None, fill_ci=None, xlabel="Date", ylabel="Close"):
    """
    y_lines: list of (y, label, linewidth)
    fill_ci: (ci_low, ci_high, label)
    """
    plt.figure(figsize=(12, 6))
    for y, label, lw in y_lines:
        plt.plot(x, y, label=label, linewidth=lw)

    if fill_ci is not None:
        ci_low, ci_high, label = fill_ci
        plt.fill_between(x, ci_low, ci_high, alpha=0.25, label=label)

    if vline is not None:
        plt.axvline(vline, linestyle="--", color="black", alpha=0.6, label="Train/Test split")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return str(path)


# -------------------- CV --------------------
def arima_expanding_walkforward_cv(sub: pd.DataFrame, min_train: int = 900, max_steps: int = 250) -> dict:
    y = sub[TARGET].values
    X = sub[EXOG_BACKTEST].values
    n = len(sub)

    start = max(min_train, int(0.6 * n))
    end = min(n - 1, start + max_steps)

    model = auto_arima(y[:start], exogenous=X[:start], **AUTO_ARIMA_KWARGS)

    preds, trues = [], []
    for t in range(start, end):
        preds.append(model.predict(1, exogenous=X[t:t+1])[0])
        trues.append(y[t])
        model.update(y[t:t+1], exogenous=X[t:t+1])

    return {
        "cv_type": "expanding_walk_forward",
        "steps": len(trues),
        "cv_mae": float(mean_absolute_error(trues, preds)),
        "cv_rmse": rmse(trues, preds),
    }


# -------------------- Main forecast function --------------------
def arima_forecast_coin(coin: str, horizon: int = 7, test_size: float = 0.2) -> dict:
    sub = load_coin(coin)
    n = len(sub)
    split = int((1 - test_size) * n)

    train = sub.iloc[:split]
    test = sub.iloc[split:]

    y_train = train[TARGET].values
    X_train = train[EXOG_BACKTEST].values
    y_test = test[TARGET].values
    X_test = test[EXOG_BACKTEST].values

    # -------- Backtest with exogenous --------
    model = auto_arima(y_train, exogenous=X_train, **AUTO_ARIMA_KWARGS)
    p, d, q = model.order

    train_pred = model.predict_in_sample(exogenous=X_train)
    train_mae = float(mean_absolute_error(y_train, train_pred))
    train_rmse = rmse(y_train, train_pred)

    fcst, ci = model.predict(len(test), exogenous=X_test, return_conf_int=True, alpha=0.05)
    test_mae = float(mean_absolute_error(y_test, fcst))
    test_rmse = rmse(y_test, fcst)

    # Save backtest predictions
    backtest_csv = OUT_DIR / f"predictions_arima_backtest_{coin}.csv"
    pd.DataFrame({
        "Date": test["Date"].values,
        "y_true": y_test,
        "y_pred": fcst,
        "ci_low": ci[:, 0],
        "ci_high": ci[:, 1],
    }).to_csv(backtest_csv, index=False)

    # Backtest plot window
    hist_start = max(0, split - HISTORY_WINDOW)
    plot_df = sub.iloc[hist_start:].copy()
    plot_dates = plot_df["Date"].values
    plot_true = plot_df[TARGET].values

    plot_pred = np.full_like(plot_true, np.nan, dtype=float)
    plot_ci_low = np.full_like(plot_true, np.nan, dtype=float)
    plot_ci_high = np.full_like(plot_true, np.nan, dtype=float)

    test_start_in_plot = split - hist_start
    plot_pred[test_start_in_plot:] = fcst
    plot_ci_low[test_start_in_plot:] = ci[:, 0]
    plot_ci_high[test_start_in_plot:] = ci[:, 1]

    backtest_plot = save_plot(
        path=FIG_DIR / f"ARIMA_{coin}_backtest.png",
        title=f"ARIMA Backtest Forecast — {coin}",
        x=plot_dates,
        y_lines=[(plot_true, "Actual", 2), (plot_pred, "Forecast", 2)],
        vline=sub["Date"].iloc[split],
        fill_ci=(plot_ci_low, plot_ci_high, "95% CI"),
        ylabel="Close",
    )

    residual_plot = save_plot(
        path=FIG_DIR / f"ARIMA_{coin}_residuals.png",
        title=f"ARIMA Residuals — {coin}",
        x=test["Date"].values,
        y_lines=[(y_test - fcst, "Residual", 1.5)],
        vline=None,
        fill_ci=None,
        ylabel="Residual",
    )

    # -------- CV --------
    cv = arima_expanding_walkforward_cv(sub)

    # -------- Future forecast (univariate) --------
    close_series = sub[TARGET].values
    uni = auto_arima(close_series, **AUTO_ARIMA_KWARGS)
    up, ud, uq = uni.order

    future_fcst = uni.predict(horizon)
    last_date = sub["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    future_csv = OUT_DIR / f"predictions_arima_future_{coin}.csv"
    pd.DataFrame({"Date": future_dates, "y_pred": future_fcst}).to_csv(future_csv, index=False)

    hist_dates = sub["Date"].iloc[-HISTORY_WINDOW:]
    hist_close = sub[TARGET].iloc[-HISTORY_WINDOW:]
    future_plot = save_plot(
        path=FIG_DIR / f"ARIMA_{coin}_future.png",
        title=f"ARIMA Future Forecast — {coin}",
        x=list(hist_dates) + list(future_dates),
        y_lines=[
            (list(hist_close) + [np.nan] * len(future_dates), "History", 2),
            ([np.nan] * len(hist_close) + list(future_fcst), "Future Forecast", 2),
        ],
        vline=hist_dates.iloc[-1],
        fill_ci=None,
        ylabel="Close",
    )

    return {
        "coin": coin,
        "model": "AutoARIMA",
        "order_backtest_exog": {"p": int(p), "d": int(d), "q": int(q)},
        "order_future_univariate": {"p": int(up), "d": int(ud), "q": int(uq)},
        "cross_validation": cv,
        "metrics": {
            "train_mae": train_mae, "train_rmse": train_rmse,
            "test_mae": test_mae, "test_rmse": test_rmse,
        },
        "artifacts": {
            "backtest_plot": backtest_plot,
            "residual_plot": residual_plot,
            "backtest_predictions_csv": str(backtest_csv),
            "future_plot": future_plot,
            "future_predictions_csv": str(future_csv),
        },
    }


# -------------------- Run for multiple coins --------------------
if __name__ == "__main__":
    coins = ["Bitcoin", "Ethereum", "Litecoin"]
    rows = []

    for coin in coins:
        try:
            out = arima_forecast_coin(coin, horizon=7)
            m = out["metrics"]
            cv = out["cross_validation"]

            print(f"\n=== {coin} ===")
            print(f"Train MAE : {m['train_mae']:.6f} | Train RMSE: {m['train_rmse']:.6f}")
            print(f"Test  MAE : {m['test_mae']:.6f} | Test  RMSE: {m['test_rmse']:.6f}")
            print(f"CV steps={cv['steps']} | CV MAE: {cv['cv_mae']:.6f} | CV RMSE: {cv['cv_rmse']:.6f}")

            rows.append({
                "coin": coin,
                **m,
                "cv_mae": cv["cv_mae"],
                "cv_rmse": cv["cv_rmse"],
                "cv_steps": cv["steps"],
                "order_backtest": out["order_backtest_exog"],
                "order_future": out["order_future_univariate"],
            })

        except Exception as e:
            print(f"\n!!! {coin} failed: {e}")

    if rows:
        summary_path = OUT_DIR / "metrics_arima_summary.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"\nSaved summary: {summary_path}")

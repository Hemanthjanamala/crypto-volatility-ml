from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "crypto_features.csv"

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def available_coins_from_dataset() -> list[str]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, usecols=["Name"])
    coins = sorted(df["Name"].dropna().unique().tolist())
    return coins

def _pred_file_name(model: str, coin: str) -> str:
    """
    Expected backtest files:
      data/processed/predictions_arima_backtest_Bitcoin.csv
      data/processed/predictions_prophet_backtest_Bitcoin.csv
      data/processed/predictions_lstm_backtest_Bitcoin.csv
    """
    return f"predictions_{model}_backtest_{coin}.csv"

def load_backtest_predictions(model: str, coin: str) -> pd.DataFrame:
    path = PROCESSED_DIR / _pred_file_name(model, coin)
    if not path.exists():
        raise FileNotFoundError(f"Missing backtest predictions: {path}")
    df = pd.read_csv(path)

    required = {"Date", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{path.name} must contain columns: {sorted(required)}. "
            f"Found: {list(df.columns)}"
        )
    df = df.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)
    return df

def score_backtest(df: pd.DataFrame) -> dict:
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "n": int(len(df)),
    }

def best_model_for_coin(coin: str, metric: str = "rmse") -> dict:
    metric = metric.lower().strip()
    if metric not in {"rmse", "mae"}:
        raise ValueError("metric must be 'rmse' or 'mae'")

    candidates = {}
    for model in ["arima", "prophet", "lstm"]:
        try:
            df = load_backtest_predictions(model, coin)
            candidates[model] = score_backtest(df)
        except FileNotFoundError:
            continue

    if not candidates:
        raise FileNotFoundError(f"No backtest files found for coin='{coin}'")

    best_model = min(candidates.items(), key=lambda kv: kv[1][metric])[0]

    return {
        "coin": coin,
        "metric": metric,
        "best_model": best_model,
        "scores": candidates,
        "note": "Best model selected by lowest error on backtest predictions (future holdout).",
    }

def best_coin_by_accuracy(model: str = "arima", metric: str = "rmse") -> dict:
    model = model.lower().strip()
    metric = metric.lower().strip()
    if model not in {"arima", "prophet", "lstm"}:
        raise ValueError("model must be one of: arima, prophet, lstm")
    if metric not in {"rmse", "mae"}:
        raise ValueError("metric must be 'rmse' or 'mae'")

    coins = available_coins_from_dataset()

    rows = []
    for coin in coins:
        try:
            df = load_backtest_predictions(model, coin)
            s = score_backtest(df)
            rows.append({"coin": coin, **s})
        except FileNotFoundError:
            # coin not backtested for this model
            continue

    if not rows:
        raise FileNotFoundError(
            f"No backtest files found for model='{model}'. "
            f"Expected files like: predictions_{model}_backtest_<Coin>.csv"
        )

    table = pd.DataFrame(rows).sort_values(metric).reset_index(drop=True)
    best_coin = str(table.iloc[0]["coin"])

    return {
        "model": model,
        "metric": metric,
        "best_coin": best_coin,
        "ranking": table.to_dict(orient="records"),
        "note": "This ranks coins by forecast accuracy (lower error is better). Not investment advice.",
    }

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "crypto_features.csv"

def load_features() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values(["Name", "Date"])
    return df

def best_coin_risk_return(window: int = 30, annualize: bool = False) -> dict:
    """
    Historical risk-adjusted ranking from your dataset only.
    score = mean(Return) / std(Return) over last `window` days per coin.
    Not a guarantee, not investment advice — just a descriptive ranking.
    """
    if window < 7:
        raise ValueError("window should be >= 7 days")

    df = load_features()

    if "Return_%" not in df.columns:
        df["Return_%"] = df.groupby("Name")["Close"].pct_change()

    rows = []
    for coin, g in df.groupby("Name"):
        r = g["Return_%"].dropna()
        if len(r) < window + 5:
            continue

        tail = r.tail(window)
        mu = float(tail.mean())
        sig = float(tail.std(ddof=1))

        if sig <= 0 or np.isnan(sig):
            continue

        score = mu / sig
        if annualize:
            score = score * np.sqrt(365)

        rows.append({
            "coin": coin,
            "window_days": window,
            "mean_return": mu,
            "volatility": sig,
            "risk_adjusted_score": float(score),
        })

    if not rows:
        raise ValueError("Not enough data to compute risk/return ranking.")

    table = pd.DataFrame(rows).sort_values("risk_adjusted_score", ascending=False).reset_index(drop=True)

    return {
        "method": "mean_return_over_volatility",
        "annualized": annualize,
        "top_coin": str(table.iloc[0]["coin"]),
        "ranking": table.to_dict(orient="records"),
        "disclaimer": "This is a historical ranking from your dataset. It is NOT investment advice.",
    }

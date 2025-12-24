from fastapi import APIRouter, HTTPException

from src.services.selection import (
    available_coins_from_dataset,
    best_model_for_coin,
    best_coin_by_accuracy,
)
from src.services.scoring import best_coin_risk_return

router = APIRouter()

@router.get("/available_coins")
def available_coins():
    try:
        return {"coins": available_coins_from_dataset()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/best_model")
def api_best_model(coin: str, metric: str = "rmse"):
    try:
        return best_model_for_coin(coin=coin, metric=metric)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/best_coin_by_accuracy")
def api_best_coin_by_accuracy(model: str = "arima", metric: str = "rmse"):
    try:
        return best_coin_by_accuracy(model=model, metric=metric)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/best_coin_risk_return")
def api_best_coin_risk_return(window: int = 30, annualize: bool = False):
    try:
        return best_coin_risk_return(window=window, annualize=annualize)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/recommend")
def recommend(mode: str = "accuracy", metric: str = "rmse", model: str = "arima", window: int = 30):
    """
    mode = 'accuracy' -> returns best coin for a chosen model by backtest error
    mode = 'risk_return' -> returns top coin by historical risk-adjusted score
    """
    try:
        mode = mode.lower().strip()

        if mode == "accuracy":
            rec = best_coin_by_accuracy(model=model, metric=metric)
            return {
                "mode": "accuracy",
                "recommended_coin": rec["best_coin"],
                "based_on": f"lowest {metric} in backtest for model={model}",
                "details": rec,
                "disclaimer": "Ranking by forecast accuracy; not investment advice.",
            }

        if mode == "risk_return":
            rec = best_coin_risk_return(window=window, annualize=False)
            return {
                "mode": "risk_return",
                "recommended_coin": rec["top_coin"],
                "based_on": f"highest mean_return/volatility over last {window} days",
                "details": rec,
                "disclaimer": rec["disclaimer"],
            }

        raise ValueError("mode must be one of: accuracy, risk_return")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

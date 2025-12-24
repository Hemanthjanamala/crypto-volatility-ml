from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.models.arima_forecast import arima_forecast_coin
from src.models.prophet_forecast import prophet_forecast_coin
from src.models.lstm_forecast import lstm_forecast_coin

router = APIRouter()

class NextDayRequest(BaseModel):
    coin: str = Field(..., examples=["Bitcoin"])
    model: str = Field(..., examples=["arima"])  # arima | prophet | lstm

@router.post("/next_day")
def predict_next_day(req: NextDayRequest):
    try:
        m = req.model.lower().strip()
        if m == "arima":
            return arima_forecast_coin(coin=req.coin, horizon=1)
        if m == "prophet":
            return prophet_forecast_coin(coin=req.coin, horizon=1)
        if m == "lstm":
            return lstm_forecast_coin(coin=req.coin, horizon=1)
        raise ValueError("model must be one of: arima, prophet, lstm")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

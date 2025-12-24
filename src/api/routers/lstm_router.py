from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from src.models.lstm_forecast import lstm_forecast_coin

router = APIRouter()

class ForecastRequest(BaseModel):
    coin: str = Field(..., examples=["Bitcoin"])
    horizon: int = Field(7, ge=1, le=60)

@router.post("/forecast")
def lstm_forecast(request: ForecastRequest):
    try:
        return lstm_forecast_coin(coin=request.coin, horizon=request.horizon)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

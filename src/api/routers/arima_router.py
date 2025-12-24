from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from src.models.arima_forecast import arima_forecast_coin

router = APIRouter()

class ForecastRequest(BaseModel):
    coin: str = Field(..., examples=["Bitcoin"])
    horizon: int = Field(7, ge=1, le=60)

@router.post("/forecast")
def arima_forecast(request: ForecastRequest):
    try:
        return arima_forecast_coin(coin=request.coin, horizon=request.horizon)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

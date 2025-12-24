from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from src.models.prophet_forecast import prophet_forecast_coin

router = APIRouter()

class ForecastRequest(BaseModel):
    coin: str = Field(..., examples=["Bitcoin"])
    horizon: int = Field(7, ge=1, le=60)

@router.post("/forecast")
def prophet_forecast(request: ForecastRequest):
    try:
        return prophet_forecast_coin(coin=request.coin, horizon=request.horizon)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from fastapi import FastAPI

from src.api.routers.arima_router import router as arima_router
from src.api.routers.prophet_router import router as prophet_router
from src.api.routers.lstm_router import router as lstm_router
from src.api.routers.meta_router import router as meta_router
from src.api.routers.predict_router import router as predict_router

app = FastAPI(
    title="Crypto Volatility Forecasting API",
    version="1.0.0",
)

@app.get("/")
def root():
    return {"message": "API running"}

@app.get("/health")
def health():
    return {"ok": True}

app.include_router(arima_router, prefix="/arima", tags=["ARIMA"])
app.include_router(prophet_router, prefix="/prophet", tags=["Prophet"])
app.include_router(lstm_router, prefix="/lstm", tags=["LSTM"])
app.include_router(meta_router, prefix="/meta", tags=["Meta"])
app.include_router(predict_router, prefix="/predict", tags=["Predict"])

from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .forecast_service import ForecastRecord
from .model_cache import ModelCache
from .live_data import LiveDataFetcher

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Initialize model cache with automatic daily retraining for multiple cryptos
model_cache = ModelCache(data_dir=str(DATA_DIR), close_col="close", window=30)

# Initialize live data fetcher
live_fetcher = LiveDataFetcher()

app = FastAPI(title="Hybrid ARIMA–LSTM–XGBoost API - Live Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/forecasts", response_model=List[ForecastRecord])
async def get_forecasts():
    """
    Get cryptocurrency forecasts using cached trained models
    Returns cached data immediately if available
    """
    # Return cached data if available
    if model_cache.get_cached_overview():
        from .forecast_service import ForecastRecord
        return [ForecastRecord(**record) for record in model_cache.get_cached_overview()]
    
    # No cache available - user needs to run train_all.py first
    return []


@app.get("/api/live-prices")
async def get_live_prices():
    """Get current live cryptocurrency prices from CoinGecko API"""
    prices = live_fetcher.get_live_prices()
    return {
        "timestamp": model_cache._last_trained.isoformat() if model_cache._last_trained else None,
        "prices": prices
    }


@app.get("/api/metrics")
async def get_metrics():
    """Get model performance metrics"""
    import pandas as pd
    try:
        metrics_path = Path(__file__).resolve().parents[1] / "outputs" / "metrics.csv"
        df = pd.read_csv(metrics_path, index_col=0)
        return {
            "models": df.to_dict('index')
        }
    except Exception as e:
        return {"error": str(e), "models": {}}


@app.get("/api/status")
async def get_status():
    """Get system status including training schedule and data range"""
    import pandas as pd
    
    # Get data range from BTC CSV (representative sample)
    data_range = None
    try:
        btc_path = DATA_DIR / "BTC.csv"
        df = pd.read_csv(btc_path)
        df['date'] = pd.to_datetime(df['date'])
        data_range = {
            "start": df['date'].min().strftime('%Y-%m-%d'),
            "end": df['date'].max().strftime('%Y-%m-%d'),
            "days": len(df)
        }
    except Exception as e:
        print(f"Error reading data range: {e}")
    
    return {
        "status": "operational",
        "training": model_cache.get_training_status(),
        "data_source": "live_api",
        "update_frequency": "24_hours",
        "data_range": data_range
    }


@app.post("/api/retrain")
async def force_retrain():
    """Force immediate model retraining (admin endpoint)"""
    model_cache.force_retrain()
    return {"message": "Retraining initiated", "status": model_cache.get_training_status()}



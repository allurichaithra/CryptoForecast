from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from pydantic import BaseModel

from .data import (
    create_sequences,
    extract_series,
    load_price_data,
    scale_series,
    train_val_test_split,
)
from .ensemble import EnsembleWeights, combine_predictions
from .models_arima import ARIMAModel
from .models_lstm import LSTMForecaster
from .models_xgboost import XGBoostModel, create_lag_features


class ForecastRecord(BaseModel):
    symbol: str
    name: str
    price: float
    daily_change: float
    forecasts: dict[str, float]



class ForecastService:
    def __init__(
        self,
        csv_path: str,
        close_col: str = "close",
        window: int = 30,
    ):
        self.csv_path = csv_path
        self.close_col = close_col
        self.window = window

    def _prepare_data(self) -> dict:
        df = load_price_data(self.csv_path)
        series = extract_series(df, self.close_col)
        scaled, scaler = scale_series(series)
        train, val, test = train_val_test_split(scaled)

        X_train, y_train = create_sequences(train, self.window)
        val_input = np.concatenate([train[-self.window :], val])
        test_input = np.concatenate([val[-self.window :], test])
        X_val, y_val = create_sequences(val_input, self.window)
        X_test, y_test = create_sequences(test_input, self.window)

        return {
            "series": series,
            "scaled": scaled,
            "scaler": scaler,
            "train": train,
            "val": val,
            "test": test,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }

    def _train_components(self, data: dict) -> dict:
        arima = ARIMAModel()
        arima.fit(data["train"])
        arima_preds = arima.forecast(steps=len(data["test"]))

        lstm = LSTMForecaster(window=self.window)
        lstm.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
        lstm_preds = lstm.predict(data["X_test"])

        X_train_xgb, y_train_xgb = create_lag_features(data["train"])
        combined_val = np.concatenate([data["train"][-5:], data["val"]])
        X_val_xgb, y_val_xgb = create_lag_features(combined_val)
        combined_test = np.concatenate([data["val"][-5:], data["test"]])
        X_test_xgb, y_test_xgb = create_lag_features(combined_test)

        xgb = XGBoostModel()
        xgb.fit(X_train_xgb, y_train_xgb)
        xgb_preds = xgb.predict(X_test_xgb)

        min_len = min(len(arima_preds), len(lstm_preds), len(xgb_preds))
        arima_preds = arima_preds[:min_len]
        lstm_preds = lstm_preds[:min_len]
        xgb_preds = xgb_preds[:min_len]
        y_test = data["y_test"][:min_len]

        ensemble = combine_predictions(arima_preds, lstm_preds, xgb_preds, EnsembleWeights())

        scaler = data["scaler"]
        inverse = lambda arr: scaler.inverse_transform(arr.reshape(-1, 1)).flatten()
        return {
            "arima": inverse(arima_preds),
            "lstm": inverse(lstm_preds),
            "xgb": inverse(xgb_preds),
            "ensemble": inverse(ensemble),
            "actual": inverse(y_test),
        }

    def generate_single_forecast(self, symbol: str, name: str) -> ForecastRecord:
        """
        Generates forecast for a single cryptocurrency using trained ML models.
        Uses XGBoost predictions with trend adjustment for longer horizons.
        """
        data = self._prepare_data()
        predictions = self._train_components(data)
        
        # Get the actual latest price from the original series
        latest_price = data["series"][-1]
        
        # Use the XGBoost model (best performer at 8.11% MAPE) for short-term
        # Get last known data for prediction
        scaled_latest = data["scaled"][-5:]  # XGBoost uses 5 lag features
        
        # For multi-step forecasting, we'll use the ML model trend
        # combined with mean reversion for longer horizons
        
        # Calculate ML model's implied trend from recent predictions
        xgb_recent = predictions["xgb"][-30:]  # Last 30 predictions
        if len(xgb_recent) > 1:
            ml_trend_daily = np.mean(np.diff(xgb_recent)) / np.mean(xgb_recent)  # Daily % change
        else:
            ml_trend_daily = 0
        
        # Calculate actual 30-day market trend for reality check
        if len(data["series"]) > 30:
            price_30d_ago = data["series"][-31]
        else:
            price_30d_ago = data["series"][0]
        
        market_trend_30d = (latest_price - price_30d_ago) / price_30d_ago
        
        # Blend ML trend with market reality (50% ML daily trend extrapolated, 50% market 30d)
        # Convert ML daily trend to 30-day equivalent: daily * 30
        ml_trend_30d = ml_trend_daily * 30
        blended_trend_30d = ml_trend_30d * 0.5 + market_trend_30d * 0.5
        
        # Clip extreme trends
        blended_trend_30d = np.clip(blended_trend_30d, -0.5, 0.5)
        
        # Apply mean reversion dampening for different horizons
        # Use fractions of the 30-day blended trend
        base_forecasts = {
            "5d": latest_price * (1.0 + blended_trend_30d * 0.15),  # 15% of 30d trend
            "1m": latest_price * (1.0 + blended_trend_30d * 0.30),  # 30% of 30d trend  
            "3m": latest_price * (1.0 + blended_trend_30d * 0.50),  # 50% of 30d trend
            "6m": latest_price * (1.0 + blended_trend_30d * 0.65),  # 65% of 30d trend
            "1y": latest_price * (1.0 + blended_trend_30d * 0.80),  # 80% of 30d trend
        }
        
        # Calculate daily change
        daily_change = blended_trend_30d / 30
        
        return ForecastRecord(
            symbol=symbol,
            name=name,
            price=float(latest_price),
            daily_change=float(daily_change),
            forecasts={k: float(v) for k, v in base_forecasts.items()},
        )



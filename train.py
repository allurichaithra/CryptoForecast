"""
Utility script to train the Hybrid ARIMA–LSTM–XGBoost ensemble and
save evaluation artifacts (metrics + plots).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from backend.data import (
    create_sequences,
    extract_series,
    load_price_data,
    scale_series,
    train_val_test_split,
)
from backend.ensemble import EnsembleWeights, combine_predictions
from backend.models_arima import ARIMAModel
from backend.models_lstm import LSTMForecaster
from backend.models_xgboost import XGBoostModel, create_lag_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid crypto forecaster.")
    parser.add_argument("--csv_path", type=str, default="data/sample_prices.csv")
    parser.add_argument("--close_col", type=str, default="close")
    parser.add_argument("--window", type=int, default=5)
    return parser.parse_args()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def main() -> None:
    args = parse_args()
    df = load_price_data(args.csv_path)
    series = extract_series(df, args.close_col)
    scaled, scaler = scale_series(series)
    train, val, test = train_val_test_split(scaled)

    X_train, y_train = create_sequences(train, args.window)
    val_input = np.concatenate([train[-args.window :], val])
    test_input = np.concatenate([val[-args.window :], test])
    X_val, y_val = create_sequences(val_input, args.window)
    X_test, y_test = create_sequences(test_input, args.window)

    arima = ARIMAModel()
    arima.fit(train)
    arima_preds = arima.forecast(steps=len(test))[: len(y_test)]

    lstm = LSTMForecaster(window=args.window)
    lstm.fit(X_train, y_train, X_val, y_val, epochs=10)
    lstm_preds = lstm.predict(X_test)[: len(y_test)]

    X_train_xgb, y_train_xgb = create_lag_features(train)
    combined_test = np.concatenate([val[-5:], test])
    X_test_xgb, _ = create_lag_features(combined_test)
    xgb = XGBoostModel()
    xgb.fit(X_train_xgb, y_train_xgb)
    xgb_preds = xgb.predict(X_test_xgb)[: len(y_test)]

    min_len = min(len(arima_preds), len(lstm_preds), len(xgb_preds), len(y_test))
    arima_preds = arima_preds[:min_len]
    lstm_preds = lstm_preds[:min_len]
    xgb_preds = xgb_preds[:min_len]
    y_test = y_test[:min_len]

    ensemble_preds = combine_predictions(arima_preds, lstm_preds, xgb_preds, EnsembleWeights())

    inverse = lambda arr: scaler.inverse_transform(arr.reshape(-1, 1)).flatten()
    arima_real = inverse(arima_preds)
    lstm_real = inverse(lstm_preds)
    xgb_real = inverse(xgb_preds)
    ensemble_real = inverse(ensemble_preds)
    y_true = inverse(y_test)

    results = {
        "ARIMA": metrics(y_true, arima_real),
        "LSTM": metrics(y_true, lstm_real),
        "XGBoost": metrics(y_true, xgb_real),
        "Hybrid Ensemble": metrics(y_true, ensemble_real),
    }

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(out_dir / "metrics.csv")

    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual", color="white")
    plt.plot(ensemble_real, label="Hybrid", color="cyan")
    plt.legend()
    plt.title("Hybrid Ensemble Forecast")
    plt.xlabel("Time steps")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(out_dir / "forecast_plot.png", dpi=200)
    plt.close()

    print("Metrics saved to outputs/metrics.csv and plot saved to outputs/forecast_plot.png")


if __name__ == "__main__":
    main()



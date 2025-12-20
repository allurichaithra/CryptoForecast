from dataclasses import dataclass

import numpy as np


@dataclass
class EnsembleWeights:
    arima: float = 0.10   # Reduced from 0.30 - ARIMA has 33% MAPE (poor for crypto)
    lstm: float = 0.40    # Increased from 0.35 - LSTM has 13% MAPE (good)
    xgb: float = 0.50     # Increased from 0.35 - XGBoost has 8% MAPE (best)

    def normalized(self) -> tuple[float, float, float]:
        total = self.arima + self.lstm + self.xgb
        return self.arima / total, self.lstm / total, self.xgb / total


def combine_predictions(
    arima_preds: np.ndarray,
    lstm_preds: np.ndarray,
    xgb_preds: np.ndarray,
    weights: EnsembleWeights | None = None,
) -> np.ndarray:
    weights = weights or EnsembleWeights()
    w_arima, w_lstm, w_xgb = weights.normalized()
    return w_arima * arima_preds + w_lstm * lstm_preds + w_xgb * xgb_preds



from typing import Optional

import numpy as np
from xgboost import XGBRegressor


class XGBoostModel:
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.model: Optional[XGBRegressor] = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def create_lag_features(series: np.ndarray, max_lag: int = 5) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(max_lag, len(series)):
        X.append(series[i - max_lag : i])
        y.append(series[i])
    return np.array(X), np.array(y)



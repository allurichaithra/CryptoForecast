from typing import Tuple

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    def __init__(self, order: Tuple[int, int, int] = (3, 1, 0)):
        self.order = order
        self._fitted = None

    def fit(self, series: np.ndarray) -> None:
        model = ARIMA(series, order=self.order)
        self._fitted = model.fit()

    def forecast(self, steps: int = 1) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("ARIMA model must be fit first.")
        return self._fitted.forecast(steps=steps)



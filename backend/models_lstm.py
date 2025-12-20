from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(window: int, units: int = 64, lr: float = 0.001) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(window, 1)),
            layers.LSTM(units),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


class LSTMForecaster:
    def __init__(self, window: int, units: int = 64, lr: float = 0.001):
        self.window = window
        self.units = units
        self.lr = lr
        self.model: Optional[tf.keras.Model] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 15,
        batch_size: int = 32,
    ) -> None:
        self.model = build_model(self.window, self.units, self.lr)
        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LSTM model has not been trained.")
        preds = self.model.predict(X, verbose=0)
        return preds.flatten()



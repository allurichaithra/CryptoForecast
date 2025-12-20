from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_price_data(csv_path: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV and ensure it is sorted chronologically by date.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(path)
    
    # Sort by date column if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    return df


def extract_series(df: pd.DataFrame, close_col: str) -> np.ndarray:
    if close_col not in df.columns:
        raise ValueError(f"Column '{close_col}' not in DataFrame.")
    return df[close_col].astype(float).values


def scale_series(series: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    return scaled, scaler


def train_val_test_split(
    series: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(series)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return series[:train_end], series[train_end:val_end], series[val_end:]


def create_sequences(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window : i])
        y.append(series[i])
    X = np.array(X)
    X = np.expand_dims(X, -1)
    y = np.array(y)
    return X, y



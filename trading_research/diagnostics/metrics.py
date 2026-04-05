"""Core model metric utilities."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob >= 0.5).astype(float) == y_true))


def logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = np.clip(y_prob, 1e-8, 1 - 1e-8)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def train_test_deviance_gap(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
) -> float:
    """Positive values imply train fit is better than test fit."""
    train_mse = float(np.mean((y_train_true - y_train_pred) ** 2)) if len(y_train_true) else 0.0
    test_mse = float(np.mean((y_test_true - y_test_pred) ** 2)) if len(y_test_true) else 0.0
    return float(test_mse - train_mse)


def prediction_disagreement(df: np.ndarray | "pd.DataFrame") -> float:
    """Average stddev across model prediction columns."""
    arr = np.asarray(df, dtype=float)
    if arr.ndim == 1:
        return 0.0
    return float(np.nanmean(np.nanstd(arr, axis=1)))

"""Classification model implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


@dataclass
class LogisticClassifier:
    learning_rate: float = 0.05
    steps: int = 400
    l2: float = 1e-3
    fit_intercept: bool = True
    coef_: np.ndarray | None = None

    def _design(self, x: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return x
        ones = np.ones((x.shape[0], 1), dtype=float)
        return np.concatenate([ones, x], axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticClassifier":
        x_design = self._design(x)
        coef = np.zeros(x_design.shape[1], dtype=float)
        for _ in range(self.steps):
            probs = _sigmoid(x_design @ coef)
            grad = (x_design.T @ (probs - y)) / len(y)
            reg = self.l2 * coef
            if self.fit_intercept:
                reg[0] = 0.0
            coef -= self.learning_rate * (grad + reg)
        self.coef_ = coef
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fit")
        return _sigmoid(self._design(x) @ self.coef_)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(float)


def classify_from_score(score: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert raw/continuous scores into binary predictions."""
    return (score >= threshold).astype(float)

"""Regression model implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgeRegressor:
    alpha: float = 1.0
    fit_intercept: bool = True
    coef_: np.ndarray | None = None

    def _design(self, x: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return x
        ones = np.ones((x.shape[0], 1), dtype=float)
        return np.concatenate([ones, x], axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        design = self._design(x)
        ident = np.eye(design.shape[1], dtype=float)
        if self.fit_intercept:
            ident[0, 0] = 0.0
        lhs = design.T @ design + self.alpha * ident
        rhs = design.T @ y
        try:
            self.coef_ = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            # Fallback for singular systems (e.g., duplicated/constant columns).
            self.coef_ = np.linalg.pinv(lhs) @ rhs
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fit")
        return self._design(x) @ self.coef_

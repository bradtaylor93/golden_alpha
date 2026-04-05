"""Advanced regression models for non-linear research baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _rbf_kernel(x_a: np.ndarray, x_b: np.ndarray, gamma: float) -> np.ndarray:
    a_sq = np.sum(x_a**2, axis=1, keepdims=True)
    b_sq = np.sum(x_b**2, axis=1, keepdims=True).T
    sq_dist = a_sq + b_sq - 2.0 * (x_a @ x_b.T)
    return np.exp(-gamma * np.clip(sq_dist, 0.0, None))


@dataclass
class KernelRidgeRegressor:
    """Kernel ridge with RBF kernel."""

    alpha: float = 1.0
    gamma: float = 1.0
    x_train_: np.ndarray | None = None
    dual_coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2:
            raise ValueError("KernelRidgeRegressor expects 2D features")
        k = _rbf_kernel(x, x, gamma=self.gamma)
        n = k.shape[0]
        lhs = k + self.alpha * np.eye(n, dtype=float)
        self.dual_coef_ = np.linalg.solve(lhs, y)
        self.x_train_ = x
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_train_ is None or self.dual_coef_ is None:
            raise RuntimeError("Model is not fit")
        x = np.asarray(x, dtype=float)
        k = _rbf_kernel(x, self.x_train_, gamma=self.gamma)
        return k @ self.dual_coef_


@dataclass
class LoessRegressor:
    """Lightweight LOESS-style local linear smoother.

    This implementation uses k-nearest neighbors with tricube distance
    weighting and weighted least-squares local fits.
    """

    frac: float = 0.25
    ridge: float = 1e-4
    x_train_: np.ndarray | None = None
    y_train_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LoessRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2:
            raise ValueError("LoessRegressor expects 2D features")
        self.x_train_ = x
        self.y_train_ = y
        return self

    def _tricube(self, z: np.ndarray) -> np.ndarray:
        w = (1.0 - np.clip(np.abs(z), 0.0, 1.0) ** 3) ** 3
        w[np.abs(z) >= 1.0] = 0.0
        return w

    def _local_predict(self, x0: np.ndarray) -> float:
        assert self.x_train_ is not None and self.y_train_ is not None
        x_train = self.x_train_
        y_train = self.y_train_
        n = x_train.shape[0]
        k = max(3, int(np.ceil(self.frac * n)))

        d = np.linalg.norm(x_train - x0[None, :], axis=1)
        nn_idx = np.argpartition(d, k - 1)[:k]
        d_nn = d[nn_idx]
        x_nn = x_train[nn_idx]
        y_nn = y_train[nn_idx]

        scale = float(np.max(d_nn)) if np.max(d_nn) > 0 else 1.0
        w = self._tricube(d_nn / scale)
        if float(np.sum(w)) <= 0:
            return float(np.mean(y_nn))

        # Local linear regression around x0.
        x_center = x_nn - x0[None, :]
        x_design = np.concatenate([np.ones((k, 1), dtype=float), x_center], axis=1)
        w_diag = np.diag(w)
        lhs = x_design.T @ w_diag @ x_design + self.ridge * np.eye(x_design.shape[1], dtype=float)
        rhs = x_design.T @ w_diag @ y_nn
        beta = np.linalg.solve(lhs, rhs)
        return float(beta[0])  # prediction at centered point (x0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_train_ is None or self.y_train_ is None:
            raise RuntimeError("Model is not fit")
        x = np.asarray(x, dtype=float)
        preds = np.array([self._local_predict(row) for row in x], dtype=float)
        return preds


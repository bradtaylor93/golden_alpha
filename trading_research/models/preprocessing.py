"""Train-fitted preprocessing utilities for TransformNode support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0.0] = 1.0
        self.std_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fit")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


@dataclass
class ClipByQuantile:
    """Fit-dependent clipping transform as an example TransformNode primitive."""

    low_q: float = 0.01
    high_q: float = 0.99
    lo_: np.ndarray | None = None
    hi_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "ClipByQuantile":
        self.lo_ = np.quantile(x, self.low_q, axis=0)
        self.hi_ = np.quantile(x, self.high_q, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.lo_ is None or self.hi_ is None:
            raise RuntimeError("Transform not fit.")
        return np.clip(x, self.lo_, self.hi_)

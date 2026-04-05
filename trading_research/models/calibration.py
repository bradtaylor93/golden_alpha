"""Calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IdentityCalibrator:
    """Placeholder calibrator that preserves the incoming score."""

    def fit(self, scores: np.ndarray, targets: np.ndarray) -> "IdentityCalibrator":
        _ = (scores, targets)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return scores

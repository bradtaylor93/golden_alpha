"""Calibration diagnostics helpers."""

from __future__ import annotations

import numpy as np


def calibration_gap(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Difference between mean prediction and mean realized value."""
    if len(y_true) == 0:
        return 0.0
    return float(abs(float(np.mean(y_pred)) - float(np.mean(y_true))))


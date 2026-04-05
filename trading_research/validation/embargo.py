"""Embargo helpers used by time-series validation."""

from __future__ import annotations

import numpy as np


def trim_train_with_embargo(
    train_positions: np.ndarray,
    test_start_position: int,
    embargo: int,
) -> np.ndarray:
    if embargo <= 0:
        return train_positions
    cutoff = test_start_position - embargo
    return train_positions[train_positions < cutoff]

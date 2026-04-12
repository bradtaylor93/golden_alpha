"""Coefficient stability diagnostics."""

from __future__ import annotations

import pandas as pd


def coefficient_stability(weight_dicts: list[dict[str, float]]) -> float:
    """Return average stddev across aligned coefficients."""
    if not weight_dicts:
        return 0.0
    df = pd.DataFrame(weight_dicts).fillna(0.0)
    if df.empty:
        return 0.0
    return float(df.std(axis=0, ddof=0).mean())

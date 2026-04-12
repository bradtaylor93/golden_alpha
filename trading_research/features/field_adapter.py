"""Adapter feature family for externally provided proprietary fields."""

from __future__ import annotations

from typing import Callable

import pandas as pd


class FieldAdapterFeatureFamily:
    """Thin adapter around an externally injected feature builder."""

    name = "field_adapter"
    version = "1"

    def __init__(
        self,
        adapter_fn: Callable[[pd.DataFrame, dict[str, object]], pd.DataFrame] | None = None,
    ) -> None:
        self._adapter_fn = adapter_fn

    def build(self, bars: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
        if self._adapter_fn is None:
            # Scaffold output to keep recipe wiring functional in open-source setups.
            out = bars.loc[:, ["timestamp", "asset"]].copy()
            out["field_adapter_stub"] = 0.0
            return out
        return self._adapter_fn(bars, params)

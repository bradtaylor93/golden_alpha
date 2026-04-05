"""Canonical data schemas used by the engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BarSchema:
    timestamp_col: str = "timestamp"
    asset_col: str = "asset"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (
            self.timestamp_col,
            self.asset_col,
            self.open_col,
            self.high_col,
            self.low_col,
            self.close_col,
            self.volume_col,
        )


def normalize_bars_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw bars into canonical timestamp/asset sorted table."""
    schema = BarSchema()
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Bars missing required columns: {missing}")
    out = df.loc[:, schema.required_columns].copy()
    out[schema.timestamp_col] = pd.to_datetime(out[schema.timestamp_col], utc=True)
    out = out.sort_values([schema.asset_col, schema.timestamp_col]).reset_index(drop=True)
    return out

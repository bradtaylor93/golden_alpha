"""Market data loading utilities.

The package intentionally starts with local CSV input so results are auditable
and vendor-neutral.  A data vendor can be layered in later by writing the same
long-form OHLCV schema that :func:`load_ohlcv_csv` returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class DataQualityReport:
    """Simple diagnostics for an OHLCV panel."""

    rows: int
    symbols: int
    start: pd.Timestamp
    end: pd.Timestamp
    missing_values: dict[str, int]
    duplicate_symbol_dates: int


def load_ohlcv_csv(path: str | Path, symbols: Iterable[str] | None = None) -> pd.DataFrame:
    """Load daily OHLCV bars from CSV and normalize the schema.

    The expected file is long-form with one row per date and symbol.
    """

    frame = pd.read_csv(path)
    lower_names = {column: column.lower() for column in frame.columns}
    frame = frame.rename(columns=lower_names)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    numeric_columns = ["open", "high", "low", "close", "volume"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")

    if symbols is not None:
        selected = {symbol.upper() for symbol in symbols}
        frame = frame[frame["symbol"].isin(selected)]

    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    duplicates = frame.duplicated(["symbol", "date"], keep=False)
    if duplicates.any():
        dupes = frame.loc[duplicates, ["symbol", "date"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate symbol/date bars found, examples: {dupes}")

    return frame


def quality_report(frame: pd.DataFrame) -> DataQualityReport:
    """Return lightweight quality checks for loaded bars."""

    if frame.empty:
        raise ValueError("Cannot summarize an empty OHLCV frame")
    duplicate_count = int(frame.duplicated(["symbol", "date"]).sum())
    return DataQualityReport(
        rows=int(len(frame)),
        symbols=int(frame["symbol"].nunique()),
        start=pd.Timestamp(frame["date"].min()),
        end=pd.Timestamp(frame["date"].max()),
        missing_values={column: int(frame[column].isna().sum()) for column in frame.columns},
        duplicate_symbol_dates=duplicate_count,
    )


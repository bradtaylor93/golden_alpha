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


def download_yahoo_ohlcv(
    symbols: Iterable[str],
    start: str,
    end: str | None = None,
    chunk_size: int = 25,
) -> pd.DataFrame:
    """Download adjusted daily OHLCV bars from Yahoo Finance.

    Yahoo data is convenient for research replication, but it is not
    point-in-time survivorship-free constituent data. Treat the output as a
    first real-data pass, not as production-grade institutional data.
    """

    import yfinance as yf

    normalized_symbols = [symbol.upper() for symbol in symbols]
    chunks: list[pd.DataFrame] = []
    for idx in range(0, len(normalized_symbols), chunk_size):
        chunk = normalized_symbols[idx : idx + chunk_size]
        raw = yf.download(
            tickers=chunk,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        chunks.extend(_normalize_yahoo_download(raw, chunk))

    if not chunks:
        raise ValueError("Yahoo Finance returned no bars for the requested symbols")

    frame = pd.concat(chunks, ignore_index=True)
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def _normalize_yahoo_download(raw: pd.DataFrame, symbols: list[str]) -> list[pd.DataFrame]:
    """Convert yfinance's wide output into the package's long OHLCV schema."""

    frames: list[pd.DataFrame] = []
    if raw.empty:
        return frames

    if isinstance(raw.columns, pd.MultiIndex):
        ticker_level = 0 if raw.columns.names[0] in {"Ticker", "Symbols"} else 1
        available = set(raw.columns.get_level_values(ticker_level))
        for symbol in symbols:
            if symbol not in available:
                continue
            try:
                symbol_frame = raw[symbol] if ticker_level == 0 else raw.xs(symbol, axis=1, level=ticker_level)
            except KeyError:
                continue
            normalized = _single_yahoo_frame(symbol_frame, symbol)
            if not normalized.empty:
                frames.append(normalized)
        return frames

    if len(symbols) == 1:
        normalized = _single_yahoo_frame(raw, symbols[0])
        if not normalized.empty:
            frames.append(normalized)
    return frames


def _single_yahoo_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rename = {column: str(column).lower() for column in frame.columns}
    out = frame.rename(columns=rename).reset_index()
    date_column = "date" if "date" in out.columns else out.columns[0]
    out = out.rename(columns={date_column: "date"})
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = sorted(set(required) - set(out.columns))
    if missing:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    out = out.loc[:, required].copy()
    out["symbol"] = symbol
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out.loc[:, REQUIRED_COLUMNS]


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


"""Data loading and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vol_shape_regimes.config import DataConfig


@dataclass(frozen=True)
class PriceData:
    frame: pd.DataFrame
    meta: dict[str, Any]


def _load_yahoo(cfg: DataConfig) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("yfinance is required for data.source='yahoo'") from exc

    df = yf.download(
        cfg.asset,
        start=cfg.start,
        end=cfg.end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No Yahoo data returned for {cfg.asset} [{cfg.start}, {cfg.end}]")
    # yfinance may return MultiIndex columns (Price, Ticker) for single tickers.
    if isinstance(df.columns, pd.MultiIndex):
        if cfg.asset in df.columns.get_level_values(-1):
            try:
                df = df.xs(cfg.asset, axis=1, level=-1)
            except Exception:
                df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
        else:
            df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]

    out = df.reset_index().rename(
        columns={
            "Date": "date",
            "Datetime": "date",
            "Close": "close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
        }
    )
    if "Volume" in out.columns:
        out = out.rename(columns={"Volume": "volume"})
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            out[col] = np.nan
    out["date"] = pd.to_datetime(out["date"], utc=False)
    return out[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def _load_csv(cfg: DataConfig) -> pd.DataFrame:
    if not cfg.csv_path:
        raise ValueError("data.csv_path must be set when source='csv'")
    raw = pd.read_csv(cfg.csv_path)
    if cfg.date_col not in raw.columns or cfg.close_col not in raw.columns:
        raise ValueError(f"CSV must contain {cfg.date_col=} and {cfg.close_col=}")
    out = raw.copy()
    out = out.rename(columns={cfg.date_col: "date", cfg.close_col: "close"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("CSV contains invalid date values.")
    for col in ["open", "high", "low", "volume"]:
        if col not in out.columns:
            out[col] = np.nan
    return out[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def _load_synthetic(cfg: DataConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state)
    n = int(cfg.synthetic_rows)
    dates = pd.date_range("2010-01-01", periods=n, freq="B")

    # Regime-varying volatility process to make clustering meaningful in tests.
    regime = np.zeros(n, dtype=int)
    for i in range(1, n):
        if rng.uniform() < 0.03:
            regime[i] = (regime[i - 1] + 1) % 3
        else:
            regime[i] = regime[i - 1]
    vol = np.select(
        [regime == 0, regime == 1, regime == 2],
        [0.006, 0.012, 0.022],
        default=0.01,
    )
    rets = rng.normal(0.0002, vol, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0.0, vol * 0.15))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, vol * 0.20)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, vol * 0.20)))
    volume = rng.integers(1_000_000, 5_000_000, size=n)
    out = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return out


def load_price_data(cfg: DataConfig) -> PriceData:
    """Load single-asset daily data into canonical schema."""
    source = cfg.source.lower()
    if source == "yahoo":
        frame = _load_yahoo(cfg)
    elif source == "csv":
        frame = _load_csv(cfg)
    elif source == "synthetic":
        frame = _load_synthetic(cfg)
    else:
        raise ValueError(f"Unsupported data source: {cfg.source}")

    frame = frame.dropna(subset=["date", "close"]).copy()
    frame = frame[frame["close"] > 0].sort_values("date").reset_index(drop=True)
    if len(frame) < 200:
        raise ValueError("Insufficient rows after preprocessing; need at least 200 observations.")
    return PriceData(
        frame=frame,
        meta={"source": cfg.source, "asset": cfg.asset, "rows": int(len(frame))},
    )

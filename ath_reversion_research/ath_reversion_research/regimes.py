"""Market-regime labels used for conditional performance reports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeConfig:
    """Parameters for broad market regime classification."""

    benchmark_symbol: str = "SPY"
    trend_lookback: int = 200
    volatility_lookback: int = 63
    high_vol_quantile: float = 0.70


def classify_market_regimes(bars: pd.DataFrame, config: RegimeConfig | None = None) -> pd.DataFrame:
    """Return daily bull/bear and volatility regime labels.

    If the preferred benchmark is unavailable, the function uses an equal
    weighted close index built from the supplied universe.
    """

    cfg = config or RegimeConfig()
    closes = bars.pivot(index="date", columns="symbol", values="close").sort_index()
    if cfg.benchmark_symbol in closes:
        benchmark = closes[cfg.benchmark_symbol]
    else:
        benchmark = closes.mean(axis=1)

    returns = benchmark.pct_change()
    trend = benchmark.rolling(cfg.trend_lookback, min_periods=max(20, cfg.trend_lookback // 4)).mean()
    vol = returns.rolling(cfg.volatility_lookback, min_periods=max(20, cfg.volatility_lookback // 3)).std()
    vol_threshold = vol.rolling(cfg.trend_lookback, min_periods=cfg.volatility_lookback).quantile(
        cfg.high_vol_quantile
    )

    direction = np.where(benchmark >= trend, "bull", "bear")
    volatility = np.where(vol >= vol_threshold, "high_vol", "normal_vol")
    label = pd.Series(direction, index=closes.index).str.cat(pd.Series(volatility, index=closes.index), sep="_")
    label = label.where(trend.notna(), "unclassified")
    return pd.DataFrame(
        {
            "date": closes.index,
            "benchmark_close": benchmark.to_numpy(),
            "benchmark_return": returns.to_numpy(),
            "trend": trend.to_numpy(),
            "volatility": vol.to_numpy(),
            "regime": label.to_numpy(),
        }
    ).reset_index(drop=True)

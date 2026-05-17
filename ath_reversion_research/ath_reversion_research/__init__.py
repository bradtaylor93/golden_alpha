"""ATH dip and exponential reversion research backtests."""

from .backtester import BacktestConfig, WalkForwardBacktester
from .data import download_yahoo_ohlcv, load_ohlcv_csv, quality_report
from .strategies import (
    ATHDipRecoveryConfig,
    ATHDipRecoveryStrategy,
    ExponentialReversionConfig,
    ExponentialReversionShortStrategy,
    MovingAverageTouchConfig,
    MovingAverageTouchStrategy,
)
from .universes import symbols_for

__all__ = [
    "ATHDipRecoveryConfig",
    "ATHDipRecoveryStrategy",
    "BacktestConfig",
    "download_yahoo_ohlcv",
    "ExponentialReversionConfig",
    "ExponentialReversionShortStrategy",
    "load_ohlcv_csv",
    "MovingAverageTouchConfig",
    "MovingAverageTouchStrategy",
    "quality_report",
    "symbols_for",
    "WalkForwardBacktester",
]

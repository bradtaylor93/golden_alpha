"""ATH dip and exponential reversion research backtests."""

from .backtester import BacktestConfig, WalkForwardBacktester
from .strategies import (
    ATHDipRecoveryConfig,
    ATHDipRecoveryStrategy,
    ExponentialReversionConfig,
    ExponentialReversionShortStrategy,
)

__all__ = [
    "ATHDipRecoveryConfig",
    "ATHDipRecoveryStrategy",
    "BacktestConfig",
    "ExponentialReversionConfig",
    "ExponentialReversionShortStrategy",
    "WalkForwardBacktester",
]

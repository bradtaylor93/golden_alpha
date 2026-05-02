from __future__ import annotations

import numpy as np
import pandas as pd

from ath_reversion_research import (
    ATHDipRecoveryConfig,
    ATHDipRecoveryStrategy,
    BacktestConfig,
    ExponentialReversionConfig,
    ExponentialReversionShortStrategy,
    MovingAverageTouchConfig,
    MovingAverageTouchStrategy,
    WalkForwardBacktester,
)


def _synthetic_bars(days: int = 360) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=days)
    rows = []
    symbols = ["SPY", "AAA", "BBB"]
    rng = np.random.default_rng(7)
    for symbol_idx, symbol in enumerate(symbols):
        base = 100.0 + symbol_idx * 20.0
        trend = np.linspace(0.0, 0.75 + symbol_idx * 0.15, days)
        cycle = 0.06 * np.sin(np.arange(days) / (12.0 + symbol_idx))
        shock = np.zeros(days)
        if symbol == "AAA":
            shock[120:150] -= np.linspace(0.0, 0.14, 30)
            shock[150:190] -= np.linspace(0.14, 0.01, 40)
        if symbol == "BBB":
            shock[190:220] += np.linspace(0.0, 0.45, 30)
            shock[220:235] += np.linspace(0.45, 0.15, 15)
        noise = rng.normal(0.0, 0.005, days).cumsum()
        close = base * np.exp(trend + cycle + shock + noise)
        for date, price in zip(dates, close):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": price * 0.998,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_backtester_returns_required_reports() -> None:
    bars = _synthetic_bars()
    backtester = WalkForwardBacktester(
        bars,
        BacktestConfig(train_days=120, test_days=60, step_days=60, transaction_cost_bps=5.0),
    )
    results = backtester.run(
        [
            ATHDipRecoveryStrategy(
                ATHDipRecoveryConfig(
                    ath_lookback=80,
                    dip_pct=0.04,
                    recovery_trigger_pct=0.01,
                    max_hold_days=30,
                )
            ),
            ExponentialReversionShortStrategy(
                ExponentialReversionConfig(
                    trend_lookback=60,
                    stretch_zscore=1.3,
                    confirmation_days=5,
                    confirmation_drop_pct=0.01,
                )
            ),
        ]
    )

    assert {"positions", "returns", "folds", "summary", "regime_summary"} == set(results)
    assert not results["summary"].empty
    assert set(results["summary"]["strategy"]) == {
        "ath_dip_recovery",
        "exponential_reversion_short",
    }
    assert {"annual_return", "annual_std", "sharpe", "average_daily_turnover"}.issubset(
        results["summary"].columns
    )
    assert not results["regime_summary"].empty


def test_strategies_generate_nonzero_positions_on_synthetic_patterns() -> None:
    bars = _synthetic_bars()
    ath_positions = ATHDipRecoveryStrategy(
        ATHDipRecoveryConfig(ath_lookback=80, dip_pct=0.04, recovery_trigger_pct=0.01)
    ).generate_positions(bars)
    short_positions = ExponentialReversionShortStrategy(
        ExponentialReversionConfig(trend_lookback=60, stretch_zscore=1.3, confirmation_drop_pct=0.01)
    ).generate_positions(bars)

    assert ath_positions["position"].abs().sum() > 0
    assert short_positions["position"].abs().sum() > 0


def test_moving_average_touch_strategy_generates_positions() -> None:
    bars = _synthetic_bars(days=420)
    strategy = MovingAverageTouchStrategy(
        MovingAverageTouchConfig(
            moving_average_days=120,
            touch_band_pct=0.03,
            require_prior_above_days=10,
            require_bounce_confirmation=True,
            require_ma_rising=True,
            max_hold_days=20,
            name="ma_touch_test",
        )
    )

    positions = strategy.generate_positions(bars)

    assert strategy.name == "ma_touch_test"
    assert positions["position"].abs().sum() > 0
    assert {"enter", "hold", "exit", "flat", "insufficient_history"}.intersection(
        set(positions["signal"])
    )

"""Run 200-day moving-average touch research on real Yahoo Finance data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ath_reversion_research import (
    BacktestConfig,
    MovingAverageTouchConfig,
    MovingAverageTouchStrategy,
    WalkForwardBacktester,
    download_yahoo_ohlcv,
    quality_report,
    symbols_for,
)


def main() -> int:
    output = Path("ath_reversion_research/reports/ma200_real_data")
    output.mkdir(parents=True, exist_ok=True)

    symbols = symbols_for(["broad_stock_sample"])
    bars = download_yahoo_ohlcv(symbols, start="2010-01-01", end=None)
    pd.DataFrame([quality_report(bars).__dict__]).to_csv(output / "data_quality.csv", index=False)

    strategies = [
        MovingAverageTouchStrategy(
            MovingAverageTouchConfig(
                name="ma200_touch_baseline",
                touch_band_pct=0.005,
                require_bounce_confirmation=False,
                require_ma_rising=False,
                require_above_long_ma=False,
                max_hold_days=40,
                stop_loss_pct=0.07,
                take_profit_pct=0.10,
            )
        ),
        MovingAverageTouchStrategy(
            MovingAverageTouchConfig(
                name="ma200_touch_bounce_confirmed",
                touch_band_pct=0.0075,
                require_bounce_confirmation=True,
                require_ma_rising=False,
                require_above_long_ma=True,
                max_hold_days=30,
                stop_loss_pct=0.06,
                take_profit_pct=0.09,
            )
        ),
        MovingAverageTouchStrategy(
            MovingAverageTouchConfig(
                name="ma200_touch_rising_ma",
                touch_band_pct=0.0075,
                require_bounce_confirmation=True,
                require_ma_rising=True,
                require_above_long_ma=True,
                ma_slope_lookback=20,
                max_hold_days=30,
                stop_loss_pct=0.06,
                take_profit_pct=0.09,
            )
        ),
        MovingAverageTouchStrategy(
            MovingAverageTouchConfig(
                name="ma200_touch_wider_stop",
                touch_band_pct=0.01,
                require_bounce_confirmation=True,
                require_ma_rising=True,
                require_above_long_ma=True,
                ma_slope_lookback=40,
                max_hold_days=50,
                stop_loss_pct=0.10,
                take_profit_pct=0.14,
            )
        ),
    ]
    config = BacktestConfig(
        train_days=756,
        test_days=126,
        step_days=126,
        transaction_cost_bps=10.0,
        benchmark_symbol="SPY",
        max_gross_exposure=1.0,
    )
    backtester = WalkForwardBacktester(bars, config)
    results = backtester.run(strategies)
    backtester.write_report(results, output)
    print(results["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

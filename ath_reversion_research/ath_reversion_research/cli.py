"""Command-line entry point for the ATH/reversion research suite."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .backtester import BacktestConfig, WalkForwardBacktester
from .data import load_ohlcv_csv, quality_report
from .strategies import (
    ATHDipRecoveryConfig,
    ATHDipRecoveryStrategy,
    ExponentialReversionConfig,
    ExponentialReversionShortStrategy,
)
from .universes import symbols_for


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run walk-forward OOS ATH dip and exponential reversion strategy tests."
    )
    parser.add_argument("--data", required=True, help="Path to long-form OHLCV CSV data.")
    parser.add_argument("--output", default="ath_reversion_report", help="Directory for CSV report outputs.")
    parser.add_argument(
        "--universes",
        nargs="*",
        default=None,
        help="Universe names to filter to. Defaults to high and lower market-cap samples.",
    )
    parser.add_argument("--train-days", type=int, default=756)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=126)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--benchmark", default="SPY")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_symbols = symbols_for(args.universes)
    bars = load_ohlcv_csv(args.data, selected_symbols)
    report = quality_report(bars)

    config = BacktestConfig(
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        transaction_cost_bps=args.cost_bps,
        benchmark_symbol=args.benchmark,
    )
    strategies = [
        ATHDipRecoveryStrategy(ATHDipRecoveryConfig()),
        ExponentialReversionShortStrategy(ExponentialReversionConfig()),
    ]
    results = WalkForwardBacktester(bars, config).run(strategies)

    out = Path(args.output)
    WalkForwardBacktester(bars, config).write_report(results, out)
    pd.DataFrame([asdict(report)]).to_csv(out / "data_quality.csv", index=False)
    print(f"Wrote report CSVs to {out}")
    print(results["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

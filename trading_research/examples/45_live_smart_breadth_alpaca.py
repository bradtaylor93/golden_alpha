"""Live execution entrypoint for smart-breadth strategy via Alpaca.

Dry-run by default. Use --mode apply to submit/replace orders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_research.live.alpaca import AlpacaBroker, AlpacaCredentials, submit_target_orders
from trading_research.live.smart_breadth_live import (
    LiveSignalConfig,
    build_target_weights,
    default_universe_170,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run live smart-breadth rebalance against Alpaca.")
    p.add_argument("--mode", choices=("dry-run", "apply"), default="dry-run")
    p.add_argument("--paper", action="store_true", help="Use Alpaca paper endpoint.")
    p.add_argument("--strategy-name", type=str, default="smart_breadth_quality_3x")
    p.add_argument("--universe-size", type=int, default=170, help="Number of symbols from default universe.")
    p.add_argument("--history-period", type=str, default="1y", help="Yahoo history period for signal features.")
    p.add_argument("--gross-target", type=float, default=3.0, help="Target gross exposure.")
    p.add_argument("--max-name-weight", type=float, default=0.12, help="Max abs weight per asset.")
    p.add_argument("--min-notional", type=float, default=200.0)
    p.add_argument("--max-notional-per-order", type=float, default=50_000.0)
    p.add_argument("--allow-cancel-open-orders", action="store_true")
    p.add_argument("--report-path", type=str, default="", help="Optional JSON path to save rebalance report.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    universe = list(default_universe_170()[: max(1, int(args.universe_size))])
    cfg = LiveSignalConfig(
        strategy_name=str(args.strategy_name),
        history_period=str(args.history_period),
        interval="1d",
        gross_target=float(args.gross_target),
        max_abs_weight=float(args.max_name_weight),
    )
    target, signal_ts = build_target_weights(
        universe=tuple(universe),
        cfg=build_target_weights.__globals__["strat"].Config(
            gross_target=float(args.gross_target),
            max_abs_weight_per_asset=float(args.max_name_weight),
        ),
        live_cfg=cfg,
    )

    broker: AlpacaBroker | None = None
    equity = float("nan")
    buying_power = float("nan")
    positions = {}
    missing_creds = False
    try:
        creds = AlpacaCredentials.from_env(paper=bool(args.paper))
        broker = AlpacaBroker(creds)
        account = broker.get_account()
        equity = float(account["equity"])
        buying_power = float(account.get("buying_power", account["equity"]))
        positions = broker.get_positions()
    except RuntimeError:
        if args.mode == "apply":
            raise
        missing_creds = True
        # Dry-run should still be usable for signal/weight inspection.
        equity = 100_000.0
        buying_power = 100_000.0
    rebalance = {
        "signal_timestamp": str(signal_ts),
        "n_symbols_target": len(target),
        "target_abs_weight_sum": float(sum(abs(float(v)) for v in target.values)),
        "target_long_weight_sum": float(target[target > 0.0].sum()),
        "target_short_weight_sum_abs": float(abs(target[target < 0.0].sum())),
        "existing_positions": [
            {"symbol": k, "qty": float(v.signed_qty), "market_value": float(v.market_value)}
            for k, v in sorted(positions.items())
        ],
    }

    print(f"mode={args.mode} paper={args.paper}")
    if missing_creds:
        print("warning: missing Alpaca credentials; running signal-only dry run with mock equity=100000")
    print(f"equity={equity:.2f} buying_power={buying_power:.2f} symbols={len(target)}")
    print(f"sum_target_abs_weight={sum(abs(float(v)) for v in target.values):.4f}")
    print(json.dumps(rebalance, indent=2))

    if args.report_path:
        out = Path(args.report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rebalance, indent=2), encoding="utf-8")
        print(f"report_saved={out.resolve()}")

    if args.mode == "apply":
        if broker is None:
            raise RuntimeError("Broker not initialized.")
        orders = submit_target_orders(
            broker=broker,
            target_weights=target,
            equity=equity,
            cancel_open_orders=bool(args.allow_cancel_open_orders),
            dry_run=False,
            max_notional_per_order=float(args.max_notional_per_order),
            min_notional_to_trade=float(args.min_notional),
        )
        print(f"submitted_orders={len(orders)}")
        for o in orders:
            print(json.dumps(o.__dict__, indent=2))
    else:
        print("dry_run_complete no orders submitted")


if __name__ == "__main__":
    main()

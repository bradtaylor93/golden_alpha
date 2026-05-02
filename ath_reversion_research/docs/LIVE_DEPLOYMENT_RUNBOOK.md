# Live Deployment Runbook

This runbook turns the validated state-overlay research into a daily target
weight file and broker-neutral order preview.  It does not place trades.

## Strategy selected for deployment

The default live config implements the validated state overlay:

- stock sleeve: 55% 126-day relative strength, 25% 12-1 momentum,
  10% ATH dip recovery
- asset sleeve: 10% 252-day absolute momentum across broad ETFs
- state overlay: scale up/down using prior-day 20-day advance breadth and
  prior-day SPY 21-day realized volatility
- target volatility: 40%
- gross exposure cap: 1.75
- single-name cap: 8%

The research report is:

`reports/state_overlay/STATE_OVERLAY_RESULTS.md`

## Daily workflow

Run after the market close, once adjusted daily data is available.

```bash
ath-live-deploy generate-targets \
  --config ath_reversion_research/configs/live_state_overlay.json \
  --output ath_reversion_research/reports/live/latest
```

Outputs:

- `target_weights.csv`: next-session target weights and dollar targets
- `risk_report.json`: gross/net exposure, state scale, volatility scale, active
  positions, and warnings

To preview orders from current holdings:

```bash
ath-live-deploy preview-orders \
  --targets ath_reversion_research/reports/live/latest/target_weights.csv \
  --positions ath_reversion_research/examples/current_positions_sample.csv \
  --portfolio-value 100000 \
  --output ath_reversion_research/reports/live/latest/orders_preview.csv
```

Replace the sample positions file with your broker export containing:

```text
symbol,quantity,price
```

## Pre-trade checklist

1. Confirm `risk_report.json` has no warnings.
2. Confirm gross exposure and max single-name weight match your mandate.
3. Remove symbols you cannot trade or borrow.
4. Apply broker-specific lot-size and liquidity constraints.
5. Use limit/VWAP-aware execution.  The research assumes next-bar execution,
   not arbitrary intraday fills.
6. Start in paper trading.  Do not route automatically until live reconciliation
   and slippage tracking have been reviewed.

## Live caveats

- The research still uses Yahoo adjusted data and current symbol lists.
- Real deployment should use point-in-time constituents, delisting-aware prices,
  corporate-action validation, borrow checks, and robust slippage models.
- 2022-2026 was a validation window used during iteration, not an untouched
  final holdout.
- This is a signal generator, not investment advice or an execution system.

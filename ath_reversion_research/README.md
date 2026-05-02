# ATH Dip and Exponential Reversion Research

Self-contained Python folder for testing two daily equity strategies:

1. **ATH dip recovery long**: find stocks that reached a recent all-time high,
   dipped, then started recovering; buy and sell on recovery, stop, target, or
   timeout.
2. **Exponential movement reversion short**: fit a rolling linear trend in log
   price, flag exponential upside stretches, and short only after a confirmed
   drop.
3. **200-day moving-average touch long**: buy stocks that pull back to their
   200-day moving average, with optional bounce and trend-quality filters.

The package runs expanding walk-forward out-of-sample tests with transaction
costs, annualized return/std/Sharpe, average return and turnover statistics, and
performance broken out by broad market regimes.

## Install

```bash
cd ath_reversion_research
python3 -m pip install -e ".[test]"
```

## Data format

Either download adjusted daily OHLCV data from Yahoo Finance or provide one CSV
in long-form daily OHLCV format:

```text
date,symbol,open,high,low,close,volume
2020-01-02,SPY,320.0,322.0,319.0,321.0,60000000
2020-01-02,AAPL,74.0,75.0,73.5,74.8,100000000
```

Static research universes are defined in
`ath_reversion_research/universes.py`:

- liquid ETF and sector ETF sample,
- high market-cap real-name companies,
- lower market-cap sample intended to be validated against point-in-time market
  caps before use in production research.

## Run

Download the built-in universe samples and run the OOS research:

```bash
ath-reversion-backtest \
  --download-yahoo \
  --start 2010-01-01 \
  --output reports/ath_reversion_real_data \
  --universes high_market_cap_cross_asset lower_market_cap_under_10bn_sample \
  --train-days 756 \
  --test-days 126 \
  --step-days 126 \
  --cost-bps 10
```

Run against an existing local CSV:

```bash
ath-reversion-backtest \
  --data /path/to/daily_ohlcv.csv \
  --output reports/ath_reversion \
  --universes high_market_cap_cross_asset lower_market_cap_under_10bn_sample \
  --train-days 756 \
  --test-days 126 \
  --step-days 126 \
  --cost-bps 10
```

Outputs:

- `positions.csv`: daily target positions and signal labels.
- `returns.csv`: walk-forward OOS daily net returns after costs.
- `folds.csv`: train/test date ranges.
- `summary.csv`: Sharpe, return, std, average stats, turnover, drawdown.
- `regime_summary.csv`: same statistics by market regime.

See `reports/MA200_REAL_DATA_RESULTS.md` for a real-data 200MA study across a
99-symbol broad stock sample, including baseline and improved variants.

## Methodology notes

- Signals are generated causally from each asset's price history.
- Positions are shifted by `execution_lag_days` before returns are applied.
- Transaction costs are charged on absolute executed-position changes.
- Portfolio gross exposure is capped and active positions are scaled equally.
- Walk-forward folds use expanding training windows and non-overlapping or
  stepped OOS windows.
- Regimes are classified using SPY when present, otherwise an equal-weighted
  close index from the supplied universe.

This is a research harness, not investment advice. Production usage should add
point-in-time constituent membership, delisting-aware prices, corporate-action
validation, borrow/short constraints, liquidity filters, and richer slippage.

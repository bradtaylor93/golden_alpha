# Real Data Results

These results were run on Friday, May 1, 2026 with Yahoo Finance adjusted daily
OHLCV data from 2010-01-01 through 2026-04-30. The walk-forward setup used a
756-trading-day expanding training window, 126-trading-day OOS test windows,
126-trading-day steps, one-day execution lag, and 10 bps transaction costs.

Yahoo data is not point-in-time survivorship-free institutional data. Treat
these results as a first real-data research pass. Production research should add
point-in-time constituents, delisting-aware prices, borrow costs/availability,
liquidity filters, and more detailed slippage.

## High market-cap cross-asset universe

Data quality:

- Symbols downloaded: 48 / 48
- Rows: 192,155
- Date range: 2010-01-04 to 2026-04-30
- Duplicate symbol/date rows: 0

| Strategy | Total return | Annual return | Annual std | Sharpe | Max drawdown | Hit rate | Avg active positions | Folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ATH dip recovery | 617.64% | 15.98% | 16.08% | 0.99 | -34.53% | 55.43% | 21.19 | 27 |
| Exponential reversion short | -83.34% | -12.61% | 19.19% | -0.66 | -83.03% | 35.91% | 2.28 | 27 |

Regime highlights:

- ATH dip recovery was strongest in bull regimes:
  - bull normal vol: 23.07% annual return, 11.58% annual std, 1.99 Sharpe.
  - bull high vol: 40.92% annual return, 16.11% annual std, 2.54 Sharpe.
- ATH dip recovery lost money in bear regimes:
  - bear high vol: -18.87% annual return, -0.60 Sharpe.
  - bear normal vol: -43.75% annual return, -2.13 Sharpe.
- Exponential reversion short only helped in bear regimes in this run:
  - bear normal vol: 72.48% annual return, 3.53 Sharpe.
  - bull normal vol: -20.41% annual return, -1.20 Sharpe.

## Lower market-cap sample

Data quality:

- Symbols downloaded: 23 / 25
- Missing from Yahoo download: RDFN, ZI
- Rows: 49,038
- Date range: 2010-01-04 to 2026-04-30
- Duplicate symbol/date rows: 0

| Strategy | Total return | Annual return | Annual std | Sharpe | Max drawdown | Hit rate | Avg active positions | Folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ATH dip recovery | 108.86% | 5.70% | 40.70% | 0.14 | -82.22% | 52.30% | 10.36 | 27 |
| Exponential reversion short | -92.77% | -17.93% | 40.73% | -0.44 | -95.00% | 22.60% | 0.78 | 27 |

Regime highlights:

- ATH dip recovery was sharply regime dependent:
  - bull normal vol: 71.03% annual return, 35.53% annual std, 2.00 Sharpe.
  - bull high vol: 55.93% annual return, 39.77% annual std, 1.41 Sharpe.
  - bear normal vol: -52.17% annual return, -1.41 Sharpe.
  - bear high vol: -61.06% annual return, -0.92 Sharpe.
- Exponential reversion short was positive in bear regimes but not enough to
  offset losses in bull regimes:
  - bear normal vol: 15.72% annual return, 0.34 Sharpe.
  - bear high vol: 10.80% annual return, 0.29 Sharpe.
  - bull normal vol: -31.36% annual return, -0.78 Sharpe.
  - bull high vol: -31.24% annual return, -0.90 Sharpe.

## Takeaways

- The ATH dip recovery idea worked materially better in the high market-cap
  universe than in the lower-market-cap sample under this first parameter set.
- The lower-market-cap version had large drawdowns and high volatility; the raw
  signal appears too pro-cyclical without regime/risk filters.
- The exponential reversion short behaves like a bear-market hedge in slices but
  was negative overall because bull-market losses dominated. A practical version
  should likely gate shorts by broader market trend/regime and add borrow and
  squeeze-risk constraints.

Compact CSV outputs are stored in `reports/real_data_summaries/`. Full daily
positions and returns were generated locally under the ignored report folders.

# 200-Day Moving Average Touch Real-Data Results

This run tests buying stocks when they touch the 200-day moving average, then
tries stricter variants intended to reduce false positives and improve risk.

## Data and evaluation

- Data source: Yahoo Finance adjusted daily OHLCV.
- Date range: 2010-01-04 through 2026-05-01.
- Universe: `broad_stock_sample`.
- Downloaded symbols: 98 of 99. Yahoo did not return usable data for `SQ`.
- Rows: 335,492.
- Walk-forward: expanding 756 train days, 126 OOS test days, 126 step days.
- OOS folds: 27.
- Costs: 10 bps on absolute executed-position changes.
- Gross exposure cap: 1.0.

## Strategy variants

| Strategy | Description |
| --- | --- |
| `ma200_touch_baseline` | Buy when daily range touches the 200MA after being above it on at least 80% of the prior 20 sessions. |
| `ma200_touch_bounce_confirmed` | Require touch plus close back above 200MA on a positive day; 6% stop, 9% target, 30-day max hold. |
| `ma200_touch_rising_ma` | Bounce-confirmed plus rising 200MA over 20 sessions. |
| `ma200_touch_wider_stop` | Bounce-confirmed, rising 200MA over 40 sessions, 10% stop, 14% target, 50-day max hold. |

## Overall OOS performance

| Strategy | Total Return | Annual Return | Annual Std | Sharpe | Max DD | Hit Rate | Avg Active Positions | Avg Daily Turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ma200_touch_baseline` | 142.70% | 6.90% | 21.19% | 0.33 | -52.61% | 51.69% | 6.78 | 19.03% |
| `ma200_touch_bounce_confirmed` | 127.30% | 6.37% | 20.94% | 0.30 | -38.13% | 51.12% | 4.83 | 18.67% |
| `ma200_touch_rising_ma` | 69.09% | 4.03% | 20.88% | 0.19 | -36.13% | 50.94% | 4.40 | 18.53% |
| `ma200_touch_wider_stop` | 269.10% | 10.32% | 19.75% | 0.52 | -36.70% | 53.30% | 6.50 | 12.17% |

The best improvement was `ma200_touch_wider_stop`: it lifted Sharpe from 0.33
to 0.52, improved annual return from 6.90% to 10.32%, reduced annualized
volatility, cut max drawdown from -52.61% to -36.70%, and reduced turnover.

## Regime breakdown

| Strategy | Regime | Annual Return | Annual Std | Sharpe | Max DD |
| --- | --- | ---: | ---: | ---: | ---: |
| `ma200_touch_baseline` | bear high vol | -20.06% | 38.93% | -0.52 | -39.19% |
| `ma200_touch_baseline` | bear normal vol | -54.82% | 26.58% | -2.06 | -41.36% |
| `ma200_touch_baseline` | bull high vol | 20.14% | 22.05% | 0.91 | -19.23% |
| `ma200_touch_baseline` | bull normal vol | 15.79% | 15.90% | 0.99 | -24.03% |
| `ma200_touch_wider_stop` | bear high vol | -18.20% | 35.95% | -0.51 | -39.75% |
| `ma200_touch_wider_stop` | bear normal vol | -55.94% | 23.29% | -2.40 | -42.52% |
| `ma200_touch_wider_stop` | bull high vol | 21.04% | 19.16% | 1.10 | -15.11% |
| `ma200_touch_wider_stop` | bull normal vol | 20.92% | 15.56% | 1.34 | -18.07% |

## Interpretation

Buying 200MA touches is strongly regime-dependent. The strategy is profitable
in bull regimes but consistently loses in bear regimes. The wider-stop variant
improves the standalone strategy, but the next obvious improvement is to gate
entries with a market-regime filter: only trade when the benchmark is in a bull
regime or when the stock has stronger relative strength versus SPY.

Detailed CSVs are committed in this folder and in
`reports/real_data_summaries/`.

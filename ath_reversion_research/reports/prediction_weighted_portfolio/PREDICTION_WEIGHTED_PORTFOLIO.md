# Prediction-Weighted Portfolio Backtest

Uses purged out-of-sample large-cap 12m financial+price predictions to form portfolios.

## Leakage controls

- Predictions are from the purged S&P annual improvement study.
- Weights are based only on prediction ranks at each trade date.
- Daily portfolio returns use one-day-lagged target weights.
- Realized forward returns are not used for sizing.

## Dataset

- Prediction events used: 192.
- Symbols: 192.
- Trade dates: 2025-01-29 00:00:00 to 2025-05-01 00:00:00.
- Metrics start after first active portfolio date: 2025-01-29.

## Portfolio summary

| observations | total_return | annual_return | annual_std | sharpe | average_daily_return | average_daily_turnover | max_drawdown | hit_rate | portfolio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 326 | 64.53% | 46.95% | 32.38% | 145.00% | 0.17% | 1.94% | -36.11% | 56.13% | top_quintile_equal |
| 326 | 56.58% | 41.42% | 32.69% | 126.72% | 0.16% | 1.94% | -35.49% | 57.06% | top_quintile_rank_weight |
| 326 | 37.88% | 28.18% | 34.41% | 81.89% | 0.12% | 1.94% | -35.85% | 56.13% | top_decile_rank_weight |
| 326 | 34.36% | 25.64% | 34.12% | 75.15% | 0.11% | 1.94% | -35.82% | 56.13% | top_decile_equal |
| 326 | 23.39% | 17.64% | 15.80% | 111.70% | 0.07% | 1.64% | -14.36% | 53.68% | top_bottom_quintile_long_short |

## Yearly summary

| observations | total_return | annual_return | annual_std | sharpe | average_daily_return | average_daily_turnover | max_drawdown | hit_rate | year | portfolio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 233 | 38.53% | 42.26% | 34.27% | 123.30% | 0.16% | 0.00% | -36.11% | 58.37% | 2025 | top_quintile_equal |
| 233 | 39.76% | 43.63% | 34.84% | 125.23% | 0.17% | 0.00% | -35.49% | 60.52% | 2025 | top_quintile_rank_weight |
| 233 | 35.93% | 39.38% | 35.97% | 109.49% | 0.16% | 0.00% | -35.82% | 59.66% | 2025 | top_decile_equal |
| 233 | 39.05% | 42.83% | 36.20% | 118.31% | 0.17% | 0.00% | -35.85% | 58.80% | 2025 | top_decile_rank_weight |
| 233 | 22.41% | 24.45% | 16.13% | 151.55% | 0.09% | 0.00% | -13.70% | 56.65% | 2025 | top_bottom_quintile_long_short |
| 93 | 18.77% | 59.39% | 27.23% | 218.16% | 0.20% | 0.00% | -10.04% | 50.54% | 2026 | top_quintile_equal |
| 93 | 12.03% | 36.04% | 26.71% | 134.94% | 0.14% | 0.00% | -11.08% | 48.39% | 2026 | top_quintile_rank_weight |
| 93 | -1.16% | -3.11% | 29.11% | -10.70% | 0.00% | 0.00% | -16.27% | 47.31% | 2026 | top_decile_equal |
| 93 | -0.84% | -2.26% | 29.56% | -7.64% | 0.01% | 0.00% | -15.98% | 49.46% | 2026 | top_decile_rank_weight |
| 93 | 0.80% | 2.18% | 14.96% | 14.56% | 0.01% | 0.00% | -7.82% | 46.24% | 2026 | top_bottom_quintile_long_short |

## Interpretation

- This is the more realistic deployment translation of the rank signal: it tests capital-weighted daily P&L, not just event forward returns.
- The sample is still short and based on current S&P 500 membership, so survivorship bias remains.
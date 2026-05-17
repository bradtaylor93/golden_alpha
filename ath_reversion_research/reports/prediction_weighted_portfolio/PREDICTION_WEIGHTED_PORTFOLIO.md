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
| 326 | 66.52% | 48.32% | 35.49% | 1.36 | 0.18% | 5.15% | -35.00% | 56.13% | top_quintile_equal_vol_target_35_brake |
| 326 | 64.53% | 46.95% | 32.38% | 1.45 | 0.17% | 1.94% | -36.11% | 56.13% | top_quintile_equal |
| 326 | 62.78% | 45.73% | 40.16% | 1.14 | 0.18% | 3.68% | -42.99% | 56.13% | top_quintile_equal_vol_target_35 |
| 326 | 56.58% | 41.42% | 32.69% | 1.27 | 0.16% | 1.94% | -35.49% | 57.06% | top_quintile_rank_weight |
| 326 | 48.64% | 35.85% | 30.81% | 1.16 | 0.14% | 3.30% | -35.24% | 56.13% | top_quintile_equal_vol_target_25 |
| 326 | 37.88% | 28.18% | 34.41% | 0.82 | 0.12% | 1.94% | -35.85% | 56.13% | top_decile_rank_weight |
| 326 | 34.36% | 25.64% | 34.12% | 0.75 | 0.11% | 1.94% | -35.82% | 56.13% | top_decile_equal |
| 326 | 23.39% | 17.64% | 15.80% | 1.12 | 0.07% | 1.64% | -14.36% | 53.68% | top_bottom_quintile_long_short |

## Yearly summary

| observations | total_return | annual_return | annual_std | sharpe | average_daily_return | average_daily_turnover | max_drawdown | hit_rate | year | portfolio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 233 | 38.53% | 42.26% | 34.27% | 1.23 | 0.16% | 0.00% | -36.11% | 58.37% | 2025 | top_quintile_equal |
| 233 | 39.76% | 43.63% | 34.84% | 1.25 | 0.17% | 0.00% | -35.49% | 60.52% | 2025 | top_quintile_rank_weight |
| 233 | 35.93% | 39.38% | 35.97% | 1.09 | 0.16% | 0.00% | -35.82% | 59.66% | 2025 | top_decile_equal |
| 233 | 39.05% | 42.83% | 36.20% | 1.18 | 0.17% | 0.00% | -35.85% | 58.80% | 2025 | top_decile_rank_weight |
| 233 | 22.41% | 24.45% | 16.13% | 1.52 | 0.09% | 0.00% | -13.70% | 56.65% | 2025 | top_bottom_quintile_long_short |
| 233 | 28.04% | 30.65% | 32.55% | 0.94 | 0.13% | 0.00% | -35.24% | 58.37% | 2025 | top_quintile_equal_vol_target_25 |
| 233 | 33.03% | 36.16% | 41.58% | 0.87 | 0.16% | 0.00% | -42.99% | 58.37% | 2025 | top_quintile_equal_vol_target_35 |
| 233 | 35.72% | 39.14% | 35.43% | 1.10 | 0.16% | 0.00% | -35.00% | 58.37% | 2025 | top_quintile_equal_vol_target_35_brake |
| 93 | 18.77% | 59.39% | 27.23% | 2.18 | 0.20% | 0.00% | -10.04% | 50.54% | 2026 | top_quintile_equal |
| 93 | 12.03% | 36.04% | 26.71% | 1.35 | 0.14% | 0.00% | -11.08% | 48.39% | 2026 | top_quintile_rank_weight |
| 93 | -1.16% | -3.11% | 29.11% | -0.11 | 0.00% | 0.00% | -16.27% | 47.31% | 2026 | top_decile_equal |
| 93 | -0.84% | -2.26% | 29.56% | -0.08 | 0.01% | 0.00% | -15.98% | 49.46% | 2026 | top_decile_rank_weight |
| 93 | 0.80% | 2.18% | 14.96% | 0.15 | 0.01% | 0.00% | -7.82% | 46.24% | 2026 | top_bottom_quintile_long_short |
| 93 | 16.09% | 49.82% | 26.10% | 1.91 | 0.17% | 0.00% | -10.03% | 50.54% | 2026 | top_quintile_equal_vol_target_25 |
| 93 | 22.37% | 72.79% | 36.54% | 1.99 | 0.24% | 0.00% | -13.88% | 50.54% | 2026 | top_quintile_equal_vol_target_35 |
| 93 | 22.70% | 74.07% | 35.83% | 2.07 | 0.25% | 0.00% | -13.88% | 50.54% | 2026 | top_quintile_equal_vol_target_35_brake |

## Interpretation

- This is the more realistic deployment translation of the rank signal: it tests capital-weighted daily P&L, not just event forward returns.
- Equal-weighting the top prediction quintile was better than rank weighting or top-decile concentration, suggesting prediction ranks are useful but prediction magnitudes are not well calibrated for aggressive sizing.
- The 35% volatility target plus drawdown brake produced the highest annualized return and slightly lower drawdown, but the unscaled top-quintile equal-weight portfolio retained the better Sharpe.
- Sector-neutral prototypes reduced return too much and are not included as recommended variants.
- The sample is still short and based on current S&P 500 membership, so survivorship bias remains.
# Prediction-Deviation Portfolio Backtest

Tests whether high-ranked annual prediction stocks that lag during the holding year are better buys.

## Leakage controls

- Annual predictions are purged out-of-sample predictions from the S&P annual improvement study.
- Intra-year deviance uses only price movement observed from signal date through current close.
- Targets are executed with one-day lag.
- Realized forward returns are not used for sizing.

## Dataset

- Prediction events used: 192.
- Symbols: 192.
- Metrics start after first active portfolio date: 2025-01-29.

## Portfolio summary

| observations | total_return | annual_return | annual_std | sharpe | average_daily_return | average_daily_turnover | max_drawdown | hit_rate | portfolio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 326 | 91.27% | 65.09% | 34.74% | 1.87 | 0.22% | 9.14% | -36.24% | 59.20% | base_plus_deviation_brake_vt35 |
| 326 | 81.03% | 58.21% | 32.38% | 1.80 | 0.20% | 8.73% | -34.04% | 59.20% | base_plus_deviation_brake_vt30 |
| 326 | 64.57% | 46.97% | 29.91% | 1.57 | 0.17% | 6.69% | -34.39% | 59.20% | base_plus_deviation_overlay |
| 326 | 64.53% | 46.95% | 32.38% | 1.45 | 0.17% | 1.94% | -36.11% | 56.13% | baseline_top_quintile_equal |
| 326 | 62.08% | 45.25% | 31.31% | 1.45 | 0.17% | 11.45% | -33.79% | 56.44% | prediction_x_deviation_weighted |
| 326 | 60.23% | 43.97% | 31.29% | 1.41 | 0.16% | 9.02% | -35.85% | 55.21% | deviation_pullback_equal |
| 326 | 58.51% | 42.77% | 30.38% | 1.41 | 0.16% | 10.59% | -34.08% | 56.44% | deviation_weighted |
| 326 | -17.21% | -13.58% | 35.79% | -0.38 | -0.03% | 36.47% | -43.11% | 51.53% | recent_dip_high_rank |

## Yearly summary

| observations | total_return | annual_return | annual_std | sharpe | average_daily_return | average_daily_turnover | max_drawdown | hit_rate | year | portfolio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 233 | 38.53% | 42.26% | 34.27% | 1.23 | 0.16% | 0.00% | -36.11% | 58.37% | 2025 | baseline_top_quintile_equal |
| 233 | 26.68% | 29.14% | 34.53% | 0.84 | 0.13% | 0.00% | -35.85% | 54.94% | 2025 | deviation_pullback_equal |
| 233 | 29.52% | 32.28% | 33.77% | 0.96 | 0.13% | 0.00% | -34.08% | 57.51% | 2025 | deviation_weighted |
| 233 | 36.56% | 40.07% | 34.58% | 1.16 | 0.16% | 0.00% | -33.79% | 56.22% | 2025 | prediction_x_deviation_weighted |
| 233 | -13.77% | -14.80% | 39.09% | -0.38 | -0.03% | 0.00% | -43.11% | 54.94% | 2025 | recent_dip_high_rank |
| 233 | 38.18% | 41.88% | 32.92% | 1.27 | 0.16% | 0.00% | -34.39% | 60.94% | 2025 | base_plus_deviation_overlay |
| 233 | 47.04% | 51.73% | 35.54% | 1.46 | 0.19% | 0.00% | -36.24% | 60.94% | 2025 | base_plus_deviation_brake_vt35 |
| 233 | 44.83% | 49.28% | 33.57% | 1.47 | 0.18% | 0.00% | -34.04% | 60.94% | 2025 | base_plus_deviation_brake_vt30 |
| 93 | 18.77% | 59.39% | 27.23% | 2.18 | 0.20% | 0.00% | -10.04% | 50.54% | 2026 | baseline_top_quintile_equal |
| 93 | 26.49% | 89.04% | 21.14% | 4.21 | 0.26% | 0.00% | -5.95% | 55.91% | 2026 | deviation_pullback_equal |
| 93 | 22.38% | 72.85% | 19.54% | 3.73 | 0.22% | 0.00% | -5.19% | 53.76% | 2026 | deviation_weighted |
| 93 | 18.69% | 59.08% | 21.15% | 2.79 | 0.19% | 0.00% | -5.15% | 56.99% | 2026 | prediction_x_deviation_weighted |
| 93 | -3.99% | -10.46% | 25.94% | -0.40 | -0.03% | 0.00% | -16.54% | 43.01% | 2026 | recent_dip_high_rank |
| 93 | 19.09% | 60.56% | 20.67% | 2.93 | 0.20% | 0.00% | -5.98% | 54.84% | 2026 | base_plus_deviation_overlay |
| 93 | 30.08% | 103.93% | 32.79% | 3.17 | 0.30% | 0.00% | -9.57% | 54.84% | 2026 | base_plus_deviation_brake_vt35 |
| 93 | 24.99% | 83.01% | 29.33% | 2.83 | 0.26% | 0.00% | -9.11% | 54.84% | 2026 | base_plus_deviation_brake_vt30 |

## Interpretation

- The best deviance variant is a blend: keep half the static top-quintile basket and allocate half to high-ranked names lagging their cohort.
- That blend did not increase raw return versus the static top-quintile basket, but it improved Sharpe, hit rate, volatility, and max drawdown.
- Adding a causal drawdown brake and volatility target to the deviance blend produced the strongest return/Sharpe tradeoff in this short sample.
- Pure recent-dip buying performed poorly, so the annual prediction rank must remain the anchor.
- The sample is short and still based on current S&P 500 membership.
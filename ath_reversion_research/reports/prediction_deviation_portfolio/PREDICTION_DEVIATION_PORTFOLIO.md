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
| 326 | 91.64% | 65.33% | 34.74% | 1.88 | 0.22% | 9.25% | -36.23% | 59.20% | base_plus_deviation_brake_vt35 |
| 326 | 91.09% | 64.96% | 34.55% | 1.88 | 0.22% | 9.35% | -36.11% | 58.90% | corr_penalty_05_brake_vt35 |
| 326 | 81.35% | 58.43% | 32.39% | 1.80 | 0.20% | 8.83% | -34.03% | 59.20% | base_plus_deviation_brake_vt30 |
| 326 | 81.06% | 58.23% | 32.31% | 1.80 | 0.20% | 8.94% | -33.92% | 58.90% | corr_penalty_05_brake_vt30 |
| 326 | 64.85% | 47.17% | 29.93% | 1.58 | 0.17% | 6.80% | -34.42% | 59.20% | base_plus_deviation_overlay |
| 326 | 64.53% | 46.95% | 32.38% | 1.45 | 0.17% | 1.94% | -36.11% | 56.13% | baseline_top_quintile_equal |
| 326 | 63.75% | 46.41% | 29.72% | 1.56 | 0.17% | 6.92% | -34.24% | 58.90% | base_plus_deviation_corr_penalty_05 |
| 326 | 62.92% | 45.83% | 29.56% | 1.55 | 0.17% | 7.01% | -34.11% | 58.90% | base_plus_deviation_corr_penalty_10 |
| 326 | 62.63% | 45.63% | 31.34% | 1.46 | 0.17% | 11.66% | -33.85% | 56.44% | prediction_x_deviation_weighted |
| 326 | 60.19% | 43.94% | 31.30% | 1.40 | 0.16% | 9.18% | -36.07% | 55.21% | deviation_pullback_equal |
| 326 | 58.91% | 43.05% | 30.41% | 1.42 | 0.16% | 10.69% | -34.18% | 56.44% | deviation_weighted |
| 326 | -17.21% | -13.58% | 35.79% | -0.38 | -0.03% | 36.47% | -43.11% | 51.53% | recent_dip_high_rank |

## Yearly summary

| observations | total_return | annual_return | annual_std | sharpe | average_daily_return | average_daily_turnover | max_drawdown | hit_rate | year | portfolio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 233 | 38.53% | 42.26% | 34.27% | 1.23 | 0.16% | 0.00% | -36.11% | 58.37% | 2025 | baseline_top_quintile_equal |
| 233 | 26.64% | 29.11% | 34.55% | 0.84 | 0.12% | 0.00% | -36.07% | 54.94% | 2025 | deviation_pullback_equal |
| 233 | 29.85% | 32.65% | 33.81% | 0.97 | 0.13% | 0.00% | -34.18% | 57.51% | 2025 | deviation_weighted |
| 233 | 37.03% | 40.59% | 34.62% | 1.17 | 0.16% | 0.00% | -33.85% | 56.22% | 2025 | prediction_x_deviation_weighted |
| 233 | -13.77% | -14.80% | 39.09% | -0.38 | -0.03% | 0.00% | -43.11% | 54.94% | 2025 | recent_dip_high_rank |
| 233 | 38.42% | 42.14% | 32.94% | 1.28 | 0.16% | 0.00% | -34.42% | 60.94% | 2025 | base_plus_deviation_overlay |
| 233 | 37.54% | 41.16% | 32.71% | 1.26 | 0.16% | 0.00% | -34.24% | 60.52% | 2025 | base_plus_deviation_corr_penalty_05 |
| 233 | 36.85% | 40.40% | 32.54% | 1.24 | 0.16% | 0.00% | -34.11% | 60.52% | 2025 | base_plus_deviation_corr_penalty_10 |
| 233 | 47.33% | 52.05% | 35.54% | 1.46 | 0.19% | 0.00% | -36.23% | 60.94% | 2025 | base_plus_deviation_brake_vt35 |
| 233 | 45.09% | 49.56% | 33.58% | 1.48 | 0.18% | 0.00% | -34.03% | 60.94% | 2025 | base_plus_deviation_brake_vt30 |
| 233 | 46.65% | 51.31% | 35.36% | 1.45 | 0.19% | 0.00% | -36.11% | 60.52% | 2025 | corr_penalty_05_brake_vt35 |
| 233 | 44.60% | 49.01% | 33.43% | 1.47 | 0.18% | 0.00% | -33.92% | 60.52% | 2025 | corr_penalty_05_brake_vt30 |
| 93 | 18.77% | 59.39% | 27.23% | 2.18 | 0.20% | 0.00% | -10.04% | 50.54% | 2026 | baseline_top_quintile_equal |
| 93 | 26.49% | 89.04% | 21.14% | 4.21 | 0.26% | 0.00% | -5.95% | 55.91% | 2026 | deviation_pullback_equal |
| 93 | 22.38% | 72.85% | 19.54% | 3.73 | 0.22% | 0.00% | -5.19% | 53.76% | 2026 | deviation_weighted |
| 93 | 18.69% | 59.08% | 21.15% | 2.79 | 0.19% | 0.00% | -5.15% | 56.99% | 2026 | prediction_x_deviation_weighted |
| 93 | -3.99% | -10.46% | 25.94% | -0.40 | -0.03% | 0.00% | -16.54% | 43.01% | 2026 | recent_dip_high_rank |
| 93 | 19.09% | 60.56% | 20.67% | 2.93 | 0.20% | 0.00% | -5.98% | 54.84% | 2026 | base_plus_deviation_overlay |
| 93 | 19.06% | 60.43% | 20.50% | 2.95 | 0.20% | 0.00% | -6.00% | 54.84% | 2026 | base_plus_deviation_corr_penalty_05 |
| 93 | 19.05% | 60.39% | 20.37% | 2.96 | 0.20% | 0.00% | -6.01% | 54.84% | 2026 | base_plus_deviation_corr_penalty_10 |
| 93 | 30.08% | 103.93% | 32.79% | 3.17 | 0.30% | 0.00% | -9.57% | 54.84% | 2026 | base_plus_deviation_brake_vt35 |
| 93 | 24.99% | 83.01% | 29.33% | 2.83 | 0.26% | 0.00% | -9.11% | 54.84% | 2026 | base_plus_deviation_brake_vt30 |
| 93 | 30.30% | 104.85% | 32.59% | 3.22 | 0.31% | 0.00% | -9.59% | 54.84% | 2026 | corr_penalty_05_brake_vt35 |
| 93 | 25.22% | 83.92% | 29.48% | 2.85 | 0.26% | 0.00% | -9.28% | 54.84% | 2026 | corr_penalty_05_brake_vt30 |

## Interpretation

- The best deviance variant is a blend: keep half the static top-quintile basket and allocate half to high-ranked names lagging their cohort.
- That blend did not increase raw return versus the static top-quintile basket, but it improved Sharpe, hit rate, volatility, and max drawdown.
- Correlation-penalty variants downweight names with high trailing average correlation to the active basket; this is causal because target weights trade the next day.
- Adding a causal drawdown brake and volatility target to the deviance blend produced the strongest return/Sharpe tradeoff in this short sample.
- Pure recent-dip buying performed poorly, so the annual prediction rank must remain the anchor.
- The sample is short and still based on current S&P 500 membership.
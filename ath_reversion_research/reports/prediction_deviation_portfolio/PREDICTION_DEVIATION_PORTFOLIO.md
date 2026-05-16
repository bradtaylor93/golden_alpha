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
| 326 | 90.67% | 64.69% | 34.73% | 1.86 | 0.22% | 9.01% | -36.36% | 59.20% | base_plus_deviation_brake_vt35 |
| 326 | 90.12% | 64.32% | 34.54% | 1.86 | 0.22% | 9.11% | -36.24% | 58.90% | corr_penalty_05_brake_vt35 |
| 326 | 80.51% | 57.86% | 32.36% | 1.79 | 0.20% | 8.60% | -34.13% | 59.20% | base_plus_deviation_brake_vt30 |
| 326 | 80.22% | 57.67% | 32.29% | 1.79 | 0.20% | 8.71% | -34.02% | 58.90% | corr_penalty_05_brake_vt30 |
| 326 | 64.53% | 46.95% | 32.38% | 1.45 | 0.17% | 1.94% | -36.11% | 56.13% | baseline_top_quintile_equal |
| 326 | 64.43% | 46.88% | 29.90% | 1.57 | 0.17% | 6.63% | -34.44% | 59.20% | base_plus_deviation_overlay |
| 326 | 63.35% | 46.13% | 29.69% | 1.55 | 0.17% | 6.75% | -34.25% | 58.90% | base_plus_deviation_corr_penalty_05 |
| 326 | 62.52% | 45.56% | 29.53% | 1.54 | 0.17% | 6.84% | -34.12% | 58.90% | base_plus_deviation_corr_penalty_10 |
| 326 | 61.81% | 45.07% | 31.28% | 1.44 | 0.17% | 11.34% | -33.88% | 56.44% | prediction_x_deviation_weighted |
| 326 | 58.19% | 42.55% | 30.34% | 1.40 | 0.16% | 10.33% | -34.19% | 56.44% | deviation_weighted |
| 326 | 53.89% | 39.54% | 31.08% | 1.27 | 0.15% | 8.50% | -36.05% | 54.91% | deviation_pullback_equal |
| 326 | 50.03% | 36.84% | 38.21% | 0.96 | 0.15% | 7.03% | -37.82% | 55.21% | monthly_weak_blend_brake_vt35 |
| 326 | 45.46% | 33.60% | 34.40% | 0.98 | 0.14% | 6.61% | -35.52% | 55.21% | monthly_weak_blend_brake_vt30 |
| 326 | 41.79% | 30.98% | 34.45% | 0.90 | 0.13% | 3.95% | -39.66% | 55.21% | monthly_weak_blend_static |
| 326 | 27.63% | 20.76% | 37.51% | 0.55 | 0.10% | 5.24% | -32.24% | 50.00% | monthly_weak_1m3m_trend_vol |
| 326 | -17.21% | -13.58% | 35.79% | -0.38 | -0.03% | 36.47% | -43.11% | 51.53% | recent_dip_high_rank |

## Yearly summary

| observations | total_return | annual_return | annual_std | sharpe | average_daily_return | average_daily_turnover | max_drawdown | hit_rate | year | portfolio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 233 | 38.53% | 42.26% | 34.27% | 1.23 | 0.16% | 0.00% | -36.11% | 58.37% | 2025 | baseline_top_quintile_equal |
| 233 | 21.66% | 23.62% | 34.27% | 0.69 | 0.11% | 0.00% | -36.05% | 54.51% | 2025 | deviation_pullback_equal |
| 233 | 29.26% | 32.00% | 33.73% | 0.95 | 0.13% | 0.00% | -34.19% | 57.51% | 2025 | deviation_weighted |
| 233 | 36.33% | 39.82% | 34.54% | 1.15 | 0.16% | 0.00% | -33.88% | 56.22% | 2025 | prediction_x_deviation_weighted |
| 233 | -13.77% | -14.80% | 39.09% | -0.38 | -0.03% | 0.00% | -43.11% | 54.94% | 2025 | recent_dip_high_rank |
| 233 | 38.07% | 41.75% | 32.91% | 1.27 | 0.16% | 0.00% | -34.44% | 60.94% | 2025 | base_plus_deviation_overlay |
| 233 | 22.15% | 24.16% | 38.61% | 0.63 | 0.12% | 0.00% | -32.24% | 49.36% | 2025 | monthly_weak_1m3m_trend_vol |
| 233 | 26.82% | 29.30% | 36.76% | 0.80 | 0.13% | 0.00% | -39.66% | 56.22% | 2025 | monthly_weak_blend_static |
| 233 | 37.20% | 40.78% | 32.68% | 1.25 | 0.16% | 0.00% | -34.25% | 60.52% | 2025 | base_plus_deviation_corr_penalty_05 |
| 233 | 36.52% | 40.03% | 32.51% | 1.23 | 0.15% | 0.00% | -34.12% | 60.52% | 2025 | base_plus_deviation_corr_penalty_10 |
| 233 | 46.58% | 51.23% | 35.53% | 1.44 | 0.19% | 0.00% | -36.36% | 60.94% | 2025 | base_plus_deviation_brake_vt35 |
| 233 | 44.42% | 48.82% | 33.54% | 1.46 | 0.18% | 0.00% | -34.13% | 60.94% | 2025 | base_plus_deviation_brake_vt30 |
| 233 | 35.90% | 39.34% | 39.70% | 0.99 | 0.16% | 0.00% | -37.82% | 56.22% | 2025 | monthly_weak_blend_brake_vt35 |
| 233 | 33.27% | 36.43% | 36.24% | 1.01 | 0.15% | 0.00% | -35.52% | 56.22% | 2025 | monthly_weak_blend_brake_vt30 |
| 233 | 45.92% | 50.48% | 35.34% | 1.43 | 0.19% | 0.00% | -36.24% | 60.52% | 2025 | corr_penalty_05_brake_vt35 |
| 233 | 43.93% | 48.27% | 33.40% | 1.45 | 0.18% | 0.00% | -34.02% | 60.52% | 2025 | corr_penalty_05_brake_vt30 |
| 93 | 18.77% | 59.39% | 27.23% | 2.18 | 0.20% | 0.00% | -10.04% | 50.54% | 2026 | baseline_top_quintile_equal |
| 93 | 26.49% | 89.04% | 21.14% | 4.21 | 0.26% | 0.00% | -5.95% | 55.91% | 2026 | deviation_pullback_equal |
| 93 | 22.38% | 72.85% | 19.54% | 3.73 | 0.22% | 0.00% | -5.19% | 53.76% | 2026 | deviation_weighted |
| 93 | 18.69% | 59.08% | 21.15% | 2.79 | 0.19% | 0.00% | -5.15% | 56.99% | 2026 | prediction_x_deviation_weighted |
| 93 | -3.99% | -10.46% | 25.94% | -0.40 | -0.03% | 0.00% | -16.54% | 43.01% | 2026 | recent_dip_high_rank |
| 93 | 19.09% | 60.56% | 20.67% | 2.93 | 0.20% | 0.00% | -5.98% | 54.84% | 2026 | base_plus_deviation_overlay |
| 93 | 4.49% | 12.64% | 34.81% | 0.36 | 0.07% | 0.00% | -15.33% | 51.61% | 2026 | monthly_weak_1m3m_trend_vol |
| 93 | 11.80% | 35.30% | 28.01% | 1.26 | 0.14% | 0.00% | -9.41% | 52.69% | 2026 | monthly_weak_blend_static |
| 93 | 19.06% | 60.43% | 20.50% | 2.95 | 0.20% | 0.00% | -6.00% | 54.84% | 2026 | base_plus_deviation_corr_penalty_05 |
| 93 | 19.05% | 60.39% | 20.37% | 2.96 | 0.20% | 0.00% | -6.01% | 54.84% | 2026 | base_plus_deviation_corr_penalty_10 |
| 93 | 30.08% | 103.93% | 32.79% | 3.17 | 0.30% | 0.00% | -9.57% | 54.84% | 2026 | base_plus_deviation_brake_vt35 |
| 93 | 24.99% | 83.01% | 29.33% | 2.83 | 0.26% | 0.00% | -9.11% | 54.84% | 2026 | base_plus_deviation_brake_vt30 |
| 93 | 10.40% | 30.76% | 34.40% | 0.89 | 0.13% | 0.00% | -14.13% | 52.69% | 2026 | monthly_weak_blend_brake_vt35 |
| 93 | 9.15% | 26.76% | 29.49% | 0.91 | 0.11% | 0.00% | -12.08% | 52.69% | 2026 | monthly_weak_blend_brake_vt30 |
| 93 | 30.30% | 104.85% | 32.59% | 3.22 | 0.31% | 0.00% | -9.59% | 54.84% | 2026 | corr_penalty_05_brake_vt35 |
| 93 | 25.22% | 83.92% | 29.48% | 2.85 | 0.26% | 0.00% | -9.28% | 54.84% | 2026 | corr_penalty_05_brake_vt30 |

## Interpretation

- The best deviance variant is a blend: keep half the static top-quintile basket and allocate half to high-ranked names lagging their cohort.
- That blend did not increase raw return versus the static top-quintile basket, but it improved Sharpe, hit rate, volatility, and max drawdown.
- Correlation-penalty variants downweight names with high trailing average correlation to the active basket; this is causal because target weights trade the next day.
- Monthly weak-relative-return variants rebalance monthly into high-ranked stocks with weak 1m/3m relative performance, but only when 12m trend and volatility filters pass.
- Adding a causal drawdown brake and volatility target to the deviance blend produced the strongest return/Sharpe tradeoff in this short sample.
- Pure recent-dip buying performed poorly, so the annual prediction rank must remain the anchor.
- The sample is short and still based on current S&P 500 membership.
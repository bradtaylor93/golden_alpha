# DoltHub Quarterly Trend Model

Tests TTM, QoQ, YoY, acceleration, margin-change, cash-flow-quality, and trailing price features from Dolt quarterly statements.

## Dataset

- Quarterly events: 17808.
- Prediction rows: 176844.
- Symbols with predictions: 499.
- Test periods: 35.

## Performance

| feature_set | forward_window | observations | test_periods | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| price_only | fwd_12m_return | 13181 | 29 | 1.39% | 15.69% | 5.91% | 30.35% | 19.46% | 69.09% | 13.71% | 0.1664392693682183 |
| quarterly_annual_price | fwd_12m_return | 13181 | 29 | -0.29% | 13.98% | 5.89% | 28.43% | 16.75% | 67.65% | 14.16% | 0.1427061092414248 |
| quarterly_trend_price | fwd_12m_return | 13181 | 29 | 0.06% | 14.20% | 5.84% | 28.54% | 17.38% | 67.58% | 14.43% | 0.14110217211002685 |
| quarterly_trend_only | fwd_12m_return | 13181 | 29 | -2.38% | 2.83% | 0.10% | 19.77% | 10.83% | 63.71% | 14.35% | 0.05415669751954724 |
| price_only | fwd_3m_return | 15980 | 35 | -4.16% | 7.33% | 6.80% | 9.32% | 7.38% | 68.21% | 4.13% | 0.05190730446875773 |
| quarterly_trend_price | fwd_3m_return | 15980 | 35 | -5.42% | 6.90% | 6.14% | 8.87% | 7.02% | 67.33% | 3.75% | 0.05127034106489872 |
| quarterly_annual_price | fwd_3m_return | 15980 | 35 | -5.96% | 6.51% | 6.05% | 8.63% | 6.88% | 67.33% | 3.98% | 0.046452905905797474 |
| quarterly_trend_only | fwd_3m_return | 15980 | 35 | -1.32% | 0.85% | -0.91% | 4.87% | 3.93% | 60.33% | 4.43% | 0.004353453869311817 |
| price_only | fwd_6m_return | 15050 | 33 | -3.86% | 4.12% | -3.48% | 11.99% | 7.36% | 62.89% | 11.36% | 0.0063106226306310215 |
| quarterly_trend_price | fwd_6m_return | 15050 | 33 | -6.42% | 2.35% | -3.79% | 10.50% | 5.06% | 60.43% | 10.74% | -0.0023519494327057244 |
| quarterly_trend_only | fwd_6m_return | 15050 | 33 | -2.74% | -0.64% | -4.01% | 7.55% | 4.62% | 58.74% | 8.37% | -0.00815683106841869 |
| quarterly_annual_price | fwd_6m_return | 15050 | 33 | -7.14% | 1.81% | -4.21% | 10.52% | 5.14% | 60.43% | 10.87% | -0.0034488069227055773 |

## Interpretation

- This is the requested time-series/rate-of-change fundamentals test.
- Quarterly features are useful only if they beat the annual/price baselines on rank and top-quintile realized returns.
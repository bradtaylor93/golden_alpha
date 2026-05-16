# Quarterly Fundamental Regression Study

This expands the fundamentals sample with quarterly statements and tests raw quarterly ratios, QoQ/YoY deltas, and latest annual context.

## Leakage controls

- Quarterly fundamentals are assumed available only 45 calendar days after quarter end.
- Forward returns start from the first trading day on or after that availability date.
- Tests are run by calendar quarter; training rows are purged unless their forward-return end date is before the test quarter starts.
- Model hyperparameters are fixed.

## Dataset

- Quarterly events: 885.
- Prediction rows: 1974.
- Symbols with predictions: 177.
- Test periods with predictions: 3.
- Yahoo Finance quarterly history is shallow; 3m/6m/12m quarterly tests may have few completed test periods.

## Predictive power by feature set

| feature_set | forward_window | regression_bucket | observations | test_periods | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quarterly_plus_annual | fwd_1m_return | all | 361 | 3 | -58.67% | 0.27% | 3.05% | 0.30% | 1.43% | 53.42% | -0.99% | 1.29% |
| quarterly_raw | fwd_1m_return | all | 361 | 3 | -30.05% | 0.22% | 0.08% | -0.26% | 0.92% | 52.05% | 1.31% | -1.57% |
| quarterly_raw_delta | fwd_1m_return | all | 361 | 3 | -46.23% | 2.12% | 1.57% | 1.01% | 1.83% | 57.53% | 0.76% | 0.24% |
| quarterly_plus_annual | fwd_1m_return | large_cap | 283 | 3 | -10.13% | 7.25% | 4.07% | 2.03% | 1.14% | 52.63% | -0.54% | 2.57% |
| quarterly_raw | fwd_1m_return | large_cap | 283 | 3 | -7.56% | 4.20% | 3.27% | 0.37% | 0.83% | 50.88% | -0.69% | 1.05% |
| quarterly_raw_delta | fwd_1m_return | large_cap | 283 | 3 | -8.31% | 4.11% | 1.53% | -0.68% | -1.78% | 47.37% | -0.71% | 0.03% |
| quarterly_plus_annual | fwd_3m_return | all | 7 | 1 | -14.84% | 77.58% | 75.00% | 20.59% | 20.59% | 100.00% | -15.86% | 36.44% |
| quarterly_raw | fwd_3m_return | all | 7 | 1 | -12.80% | 68.60% | 75.00% | 14.83% | 14.83% | 50.00% | -24.16% | 38.99% |
| quarterly_raw_delta | fwd_3m_return | all | 7 | 1 | 4.02% | 88.33% | 75.00% | 14.83% | 14.83% | 50.00% | -25.34% | 40.17% |
| quarterly_plus_annual | fwd_3m_return | large_cap | 7 | 1 | 12.20% | 89.05% | 75.00% | 18.85% | 18.85% | 100.00% | -19.88% | 38.73% |
| quarterly_raw | fwd_3m_return | large_cap | 7 | 1 | -21.16% | 70.30% | 64.29% | 14.83% | 14.83% | 50.00% | -24.16% | 38.99% |
| quarterly_raw_delta | fwd_3m_return | large_cap | 7 | 1 | 9.73% | 88.54% | 64.29% | 18.85% | 18.85% | 100.00% | -19.88% | 38.73% |

## Conditional subsets

| feature_set | forward_window | regression_bucket | condition | observations | mean_return | median_return | hit_rate | avg_q_yoy_growth | avg_q_qoq_growth | avg_q_operating_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quarterly_plus_annual | fwd_1m_return | all | q_yoy_growth_30pct_positive_margin | 5 | 14.19% | 2.76% | 60.00% | 89.35% | 23.27% | 33.86% |
| quarterly_raw | fwd_1m_return | all | q_yoy_growth_30pct_positive_margin | 5 | 14.19% | 2.76% | 60.00% | 89.35% | 23.27% | 33.86% |
| quarterly_raw_delta | fwd_1m_return | all | q_yoy_growth_30pct_positive_margin | 5 | 14.19% | 2.76% | 60.00% | 89.35% | 23.27% | 33.86% |
| quarterly_raw | fwd_1m_return | all | q_yoy_growth_30pct_model_top_half | 7 | 7.61% | -1.78% | 42.86% | 69.03% | 23.40% | 20.53% |
| quarterly_plus_annual | fwd_1m_return | all | q_yoy_growth_30pct | 8 | 7.00% | 0.49% | 50.00% | 68.59% | 21.33% | 19.50% |
| quarterly_raw | fwd_1m_return | all | q_yoy_growth_30pct | 8 | 7.00% | 0.49% | 50.00% | 68.59% | 21.33% | 19.50% |
| quarterly_raw_delta | fwd_1m_return | all | q_yoy_growth_30pct | 8 | 7.00% | 0.49% | 50.00% | 68.59% | 21.33% | 19.50% |
| quarterly_plus_annual | fwd_1m_return | all | annual_growth_30pct_q_positive | 7 | 6.20% | -4.53% | 28.57% | 65.46% | 21.94% | 19.19% |
| quarterly_raw | fwd_1m_return | all | annual_growth_30pct_q_positive | 7 | 6.20% | -4.53% | 28.57% | 65.46% | 21.94% | 19.19% |
| quarterly_raw_delta | fwd_1m_return | all | annual_growth_30pct_q_positive | 7 | 6.20% | -4.53% | 28.57% | 65.46% | 21.94% | 19.19% |
| quarterly_raw_delta | fwd_1m_return | all | model_top_quintile | 73 | 1.01% | 1.83% | 57.53% | 12.91% | 19.71% | 11.45% |
| quarterly_plus_annual | fwd_1m_return | all | model_top_quintile | 73 | 0.30% | 1.43% | 53.42% | 14.13% | 20.31% | 11.86% |
| quarterly_raw | fwd_1m_return | all | model_top_quintile | 73 | -0.26% | 0.92% | 52.05% | 18.37% | 21.21% | 9.43% |
| quarterly_raw_delta | fwd_1m_return | all | q_yoy_growth_30pct_model_top_half | 5 | -5.67% | -11.62% | 20.00% | 42.74% | 13.88% | -2.36% |
| quarterly_plus_annual | fwd_1m_return | large_cap | q_yoy_growth_30pct_positive_margin | 5 | 14.19% | 2.76% | 60.00% | 89.35% | 23.27% | 33.86% |
| quarterly_raw | fwd_1m_return | large_cap | q_yoy_growth_30pct_positive_margin | 5 | 14.19% | 2.76% | 60.00% | 89.35% | 23.27% | 33.86% |
| quarterly_raw_delta | fwd_1m_return | large_cap | q_yoy_growth_30pct_positive_margin | 5 | 14.19% | 2.76% | 60.00% | 89.35% | 23.27% | 33.86% |
| quarterly_plus_annual | fwd_1m_return | large_cap | q_yoy_growth_30pct_model_top_half | 6 | 11.14% | 3.62% | 50.00% | 74.16% | 20.78% | 20.53% |
| quarterly_plus_annual | fwd_1m_return | large_cap | q_yoy_growth_30pct | 8 | 7.00% | 0.49% | 50.00% | 68.59% | 21.33% | 19.50% |
| quarterly_raw | fwd_1m_return | large_cap | q_yoy_growth_30pct | 8 | 7.00% | 0.49% | 50.00% | 68.59% | 21.33% | 19.50% |
| quarterly_raw_delta | fwd_1m_return | large_cap | q_yoy_growth_30pct | 8 | 7.00% | 0.49% | 50.00% | 68.59% | 21.33% | 19.50% |
| quarterly_plus_annual | fwd_1m_return | large_cap | model_top_quintile | 57 | 2.03% | 1.14% | 52.63% | 47.91% | 8.03% | 22.26% |
| quarterly_raw | fwd_1m_return | large_cap | model_top_quintile | 57 | 0.37% | 0.83% | 50.88% | 15.13% | 4.79% | 18.07% |
| quarterly_raw_delta | fwd_1m_return | large_cap | model_top_quintile | 57 | -0.68% | -1.78% | 47.37% | 15.34% | 4.38% | 19.36% |
| quarterly_raw | fwd_1m_return | large_cap | q_yoy_growth_30pct_model_top_half | 5 | -3.88% | -1.78% | 40.00% | 56.12% | 9.46% | 15.38% |

## Interpretation

- Quarterly data increases event count and creates more test periods, but Yahoo's quarterly history is still shallow.
- The quarterly-plus-annual feature set should be judged by OOS rank/correlation and top-minus-bottom spreads, not R2 alone.
- Any low/mid-cap results with small row counts remain fragile.
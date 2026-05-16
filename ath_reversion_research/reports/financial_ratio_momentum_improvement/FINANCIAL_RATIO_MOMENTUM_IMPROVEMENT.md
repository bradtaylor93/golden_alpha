# Financial Ratio + Momentum Improvement Study

This study tests whether causal trailing price context improves annual financial-ratio regressions.

## Leakage controls

- Starts from financial events already lagged 90 days after fiscal year end.
- Adds only trailing returns, relative strength, volatility, drawdown, and market cap known at the trade date.
- Uses purged walk-forward training; labels overlapping a test year are excluded from training.
- Ridge alpha is fixed and not tuned on validation rows.

## Financial-only versus financial + price context

| feature_set | forward_window | regression_bucket | observations | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| financial_only | fwd_12m_return | all | 150 | -5.12% | 2.51% | 16.23% | 21.99% | 23.33% | 70.00% | 17.75% | 0.04238430202733523 |
| financial_plus_price | fwd_12m_return | all | 150 | -31.79% | -14.25% | 2.40% | 9.12% | 5.68% | 63.33% | 39.95% | -0.3082978718536431 |
| financial_only | fwd_12m_return | large_cap | 117 | 1.89% | 19.65% | 12.75% | 34.51% | 23.74% | 75.00% | 17.47% | 0.17041943558665082 |
| financial_plus_price | fwd_12m_return | large_cap | 117 | -22.28% | -2.59% | 1.35% | 20.28% | 21.04% | 75.00% | 22.02% | -0.01736433243110097 |
| financial_only | fwd_3m_return | all | 181 | -19.26% | -5.74% | -16.84% | 11.90% | 3.42% | 59.46% | 12.24% | -0.0034451679245354333 |
| financial_plus_price | fwd_3m_return | all | 181 | -13.59% | 0.97% | -7.08% | 13.55% | 3.17% | 62.16% | 13.53% | 0.00023498472005661908 |
| financial_only | fwd_3m_return | large_cap | 142 | 1.90% | 33.16% | 14.82% | 14.96% | 11.98% | 68.97% | 2.25% | 0.1271317937072571 |
| financial_plus_price | fwd_3m_return | large_cap | 142 | 1.78% | 18.66% | 3.83% | 14.09% | 3.42% | 65.52% | 7.03% | 0.0706026090799259 |
| financial_only | fwd_3m_return | low_cap | 21 | -70.30% | -32.77% | -35.71% | 1.32% | 0.88% | 60.00% | 33.39% | -0.3206789695303149 |
| financial_plus_price | fwd_3m_return | low_cap | 21 | -83.01% | -19.91% | -19.61% | 10.84% | 0.88% | 60.00% | 20.30% | -0.09458624683742603 |
| financial_only | fwd_3m_return | mid_cap | 18 | -73.33% | -49.14% | -50.05% | -7.17% | 0.20% | 50.00% | 37.56% | -0.4472908661051481 |
| financial_plus_price | fwd_3m_return | mid_cap | 18 | -113.67% | -63.76% | -56.24% | -7.17% | 0.20% | 50.00% | 26.56% | -0.3372510681649792 |
| financial_only | fwd_6m_return | all | 167 | -7.18% | -17.18% | -13.54% | 10.46% | 4.37% | 67.65% | 39.22% | -0.287592771282268 |
| financial_plus_price | fwd_6m_return | all | 167 | -5.52% | -3.31% | 9.93% | 20.58% | 16.75% | 73.53% | 30.98% | -0.10397719164526364 |
| financial_only | fwd_6m_return | large_cap | 128 | 1.97% | 19.62% | 3.93% | 26.56% | 13.57% | 80.77% | 12.50% | 0.14060626191936837 |
| financial_plus_price | fwd_6m_return | large_cap | 128 | 0.82% | 14.23% | 10.72% | 21.58% | 14.07% | 80.77% | 11.36% | 0.10216777777388579 |
| financial_only | fwd_6m_return | low_cap | 21 | -2.73% | 41.33% | 41.95% | 202.05% | 128.68% | 80.00% | -10.84% | 2.128874878422324 |
| financial_plus_price | fwd_6m_return | low_cap | 21 | -4.05% | 31.34% | 48.31% | 178.67% | 11.77% | 80.00% | -14.58% | 1.9325071212128362 |
| financial_only | fwd_6m_return | mid_cap | 18 | -31.03% | -43.38% | -26.73% | -17.54% | -10.71% | 25.00% | 48.93% | -0.6646824276697392 |
| financial_plus_price | fwd_6m_return | mid_cap | 18 | -36.54% | -46.05% | -16.62% | -2.17% | -3.88% | 50.00% | 48.93% | -0.510962754588647 |

## Improved condition subsets

| feature_set | forward_window | regression_bucket | condition | observations | mean_return | median_return | hit_rate | avg_revenue_growth | avg_trailing_6m_return | avg_operating_margin | avg_fcf_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| financial_only | fwd_12m_return | all | model_top_quintile | 30 | 21.99% | 23.33% | 70.00% | 17.80% | -1.47% | 24.44% | 21.38% |
| financial_plus_price | fwd_12m_return | all | model_top_quintile | 30 | 9.12% | 5.68% | 63.33% | 14.90% | 15.77% | 25.76% | 19.51% |
| financial_only | fwd_12m_return | large_cap | model_top_quintile | 24 | 34.51% | 23.74% | 75.00% | 28.18% | -0.96% | 23.84% | 21.77% |
| financial_plus_price | fwd_12m_return | large_cap | model_top_quintile | 24 | 20.28% | 21.04% | 75.00% | 18.49% | 20.48% | 18.00% | 17.36% |
| financial_plus_price | fwd_3m_return | all | model_top_quintile | 37 | 13.55% | 3.17% | 62.16% | 18.79% | 17.83% | 20.82% | 17.58% |
| financial_only | fwd_3m_return | all | model_top_quintile | 37 | 11.90% | 3.42% | 59.46% | 20.97% | 11.18% | 21.03% | 19.72% |
| financial_only | fwd_3m_return | large_cap | model_top_quintile | 29 | 14.96% | 11.98% | 68.97% | 22.38% | 10.47% | 29.46% | 20.99% |
| financial_plus_price | fwd_3m_return | large_cap | model_top_quintile | 29 | 14.09% | 3.42% | 65.52% | 22.91% | 19.39% | 24.05% | 18.09% |
| financial_plus_price | fwd_3m_return | low_cap | model_top_quintile | 5 | 10.84% | 0.88% | 60.00% | 18.57% | -13.12% | 5.04% | 8.79% |
| financial_only | fwd_3m_return | low_cap | model_top_quintile | 5 | 1.32% | 0.88% | 60.00% | 14.04% | -0.14% | -1.33% | 9.65% |
| financial_plus_price | fwd_6m_return | all | model_top_quintile | 34 | 20.58% | 16.75% | 73.53% | 10.33% | 6.21% | 23.91% | 12.05% |
| financial_only | fwd_6m_return | all | model_top_quintile | 34 | 10.46% | 4.37% | 67.65% | 10.68% | 1.01% | 30.14% | 18.58% |
| financial_only | fwd_6m_return | large_cap | model_top_quintile | 26 | 26.56% | 13.57% | 80.77% | 23.29% | 2.46% | 27.54% | 19.73% |
| financial_plus_price | fwd_6m_return | large_cap | model_top_quintile | 26 | 21.58% | 14.07% | 80.77% | 17.54% | 8.29% | 26.35% | 21.32% |
| financial_only | fwd_6m_return | low_cap | model_top_quintile | 5 | 202.05% | 128.68% | 80.00% | -15.04% | -19.29% | -49.16% | -39.64% |
| financial_plus_price | fwd_6m_return | low_cap | model_top_quintile | 5 | 178.67% | 11.77% | 80.00% | 6.11% | -23.67% | -1.69% | -26.69% |

## Interpretation

- Adding price context improves some all-stock ranking metrics, especially the 6m top-quintile mean and hit rate.
- It does not universally improve the model: large-cap financial-only regressions remain stronger than financial-plus-price for several top-quintile tests.
- The most interesting recurring pocket remains 30%+ revenue growth with positive profitability/FCF; trailing momentum can be used as an additional filter, not a guaranteed improvement.
- Mid-cap and low-cap outputs remain too sample-limited for deployment.
- This is still a hypothesis layer; it should be combined with the existing technical/momentum framework rather than traded standalone.
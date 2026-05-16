# LightGBM Fundamental Ranking Study

Fixed-parameter LightGBM compared with the current ridge financial+price model.

## Leakage controls

- Uses the same purged annual walk-forward setup as the ridge model.
- No LightGBM hyperparameter tuning was performed on test rows.
- Features are annual financials plus causal trailing price context.

## LightGBM results

| model | forward_window | regression_bucket | observations | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_fixed | fwd_12m_return | all | 420 | 1.41% | 16.16% | 6.66% | 30.80% | 15.59% | 69.05% | 12.93% | 17.87% |
| lightgbm_fixed | fwd_12m_return | large_cap | 192 | 11.76% | 36.45% | 21.39% | 67.91% | 49.27% | 84.62% | 15.84% | 52.07% |
| lightgbm_fixed | fwd_12m_return | mid_cap | 212 | -14.48% | -4.53% | -0.44% | 7.72% | 9.87% | 60.47% | 22.29% | -14.57% |
| lightgbm_fixed | fwd_3m_return | all | 1011 | -4.83% | 4.73% | -2.69% | 1.87% | -1.38% | 46.31% | 1.26% | 0.62% |
| lightgbm_fixed | fwd_3m_return | large_cap | 235 | 0.21% | 35.80% | 22.43% | 23.12% | 18.64% | 80.85% | 3.58% | 19.54% |
| lightgbm_fixed | fwd_3m_return | mid_cap | 507 | -19.01% | -9.50% | -8.32% | -2.07% | -1.17% | 40.20% | 1.36% | -3.43% |
| lightgbm_fixed | fwd_6m_return | all | 472 | 16.46% | 41.45% | 13.74% | 25.78% | 5.73% | 67.71% | 6.89% | 18.89% |
| lightgbm_fixed | fwd_6m_return | large_cap | 213 | 5.59% | 46.46% | 37.38% | 62.34% | 63.15% | 90.70% | 6.25% | 56.09% |
| lightgbm_fixed | fwd_6m_return | mid_cap | 239 | -2.77% | 12.12% | 7.46% | 11.95% | 4.92% | 62.50% | 5.23% | 6.72% |

## Ridge benchmark rows

| feature_set | forward_window | regression_bucket | observations | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| financial_plus_price | fwd_12m_return | all | 420 | 0.83% | 30.07% | 18.32% | 50.58% | 35.76% | 76.19% | 12.28% | 38.30% |
| financial_plus_price | fwd_12m_return | large_cap | 192 | 22.84% | 52.02% | 40.20% | 81.17% | 61.18% | 92.31% | 11.86% | 69.31% |
| financial_plus_price | fwd_12m_return | mid_cap | 212 | -32.97% | -8.81% | -8.98% | 6.01% | 0.06% | 53.49% | 17.17% | -11.15% |
| financial_plus_price | fwd_3m_return | all | 1011 | -8.96% | 17.35% | 18.22% | 7.26% | 2.71% | 60.10% | -1.83% | 9.09% |
| financial_plus_price | fwd_3m_return | large_cap | 235 | 3.13% | 23.79% | 17.26% | 16.30% | 12.69% | 74.47% | 6.36% | 9.94% |
| financial_plus_price | fwd_3m_return | mid_cap | 507 | -16.90% | 12.55% | 12.21% | 2.22% | 0.40% | 50.98% | -3.71% | 5.93% |
| financial_plus_price | fwd_6m_return | all | 472 | 18.29% | 47.79% | 35.58% | 40.25% | 27.38% | 85.26% | 2.30% | 37.95% |
| financial_plus_price | fwd_6m_return | large_cap | 213 | 6.06% | 28.15% | 25.44% | 46.50% | 38.29% | 88.37% | 20.08% | 26.42% |
| financial_plus_price | fwd_6m_return | mid_cap | 239 | -16.87% | 24.35% | 27.20% | 15.33% | 12.86% | 75.00% | -2.20% | 17.53% |

## LightGBM condition subsets

| forward_window | regression_bucket | condition | observations | mean_return | median_return | hit_rate |
| --- | --- | --- | --- | --- | --- | --- |
| fwd_12m_return | all | growth30_positive_margin | 13 | 58.64% | 33.42% | 76.92% |
| fwd_12m_return | all | growth30_model_top_half | 15 | 49.29% | 22.48% | 66.67% |
| fwd_12m_return | all | model_top_quintile | 84 | 30.80% | 15.59% | 69.05% |
| fwd_12m_return | large_cap | growth30_model_top_half | 7 | 82.73% | 62.00% | 85.71% |
| fwd_12m_return | large_cap | growth30_positive_margin | 10 | 75.68% | 54.19% | 90.00% |
| fwd_12m_return | large_cap | model_top_quintile | 39 | 67.91% | 49.27% | 84.62% |
| fwd_12m_return | mid_cap | model_top_quintile | 43 | 7.72% | 9.87% | 60.47% |
| fwd_3m_return | all | growth30_positive_margin | 39 | 14.31% | 6.00% | 61.54% |
| fwd_3m_return | all | growth30_model_top_half | 27 | 8.43% | 4.60% | 51.85% |
| fwd_3m_return | all | model_top_quintile | 203 | 1.87% | -1.38% | 46.31% |
| fwd_3m_return | large_cap | growth30_positive_margin | 13 | 39.90% | 27.07% | 76.92% |
| fwd_3m_return | large_cap | growth30_model_top_half | 10 | 31.24% | 25.27% | 70.00% |
| fwd_3m_return | large_cap | model_top_quintile | 47 | 23.12% | 18.64% | 80.85% |
| fwd_3m_return | mid_cap | growth30_positive_margin | 15 | -0.86% | -1.62% | 46.67% |
| fwd_3m_return | mid_cap | model_top_quintile | 102 | -2.07% | -1.17% | 40.20% |
| fwd_3m_return | mid_cap | growth30_model_top_half | 6 | -5.96% | -5.12% | 33.33% |
| fwd_6m_return | all | growth30_model_top_half | 13 | 77.86% | 72.26% | 76.92% |
| fwd_6m_return | all | growth30_positive_margin | 16 | 67.74% | 57.68% | 75.00% |
| fwd_6m_return | all | model_top_quintile | 96 | 25.78% | 5.73% | 67.71% |
| fwd_6m_return | large_cap | growth30_positive_margin | 12 | 89.09% | 78.92% | 83.33% |
| fwd_6m_return | large_cap | growth30_model_top_half | 10 | 86.53% | 78.92% | 100.00% |
| fwd_6m_return | large_cap | model_top_quintile | 43 | 62.34% | 63.15% | 90.70% |
| fwd_6m_return | mid_cap | model_top_quintile | 48 | 11.95% | 4.92% | 62.50% |

## Interpretation

- LightGBM is only useful if it beats the ridge financial+price model on rank correlation and top-quintile portfolio characteristics.
- With this short Yahoo/current-S&P sample, nonlinear models are at high risk of overfitting.
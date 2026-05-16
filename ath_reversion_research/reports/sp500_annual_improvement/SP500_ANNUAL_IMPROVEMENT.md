# S&P 500 Annual Improvement Study

Tests feature subsets, causal trailing price context, and sector-neutral ranking on the broader current S&P 500 annual sample.

## Leakage controls

- Uses the same 90-day annual reporting lag and purged walk-forward training as the S&P annual study.
- Price features are trailing-only and known at the financial availability date.
- Sector labels and S&P membership are current, not point-in-time; survivorship bias remains.

## Model variants

| feature_set | forward_window | regression_bucket | observations | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| financial_only | fwd_12m_return | all | 420 | -0.43% | 13.17% | 13.01% | 25.17% | 9.88% | 63.10% | 9.41% | 15.76% |
| financial_plus_price | fwd_12m_return | all | 420 | 0.83% | 30.07% | 18.32% | 50.58% | 35.76% | 76.19% | 12.28% | 38.30% |
| growth_quality_only | fwd_12m_return | all | 420 | 0.57% | 13.24% | 12.34% | 29.55% | 11.66% | 64.29% | 11.54% | 18.01% |
| growth_quality_price | fwd_12m_return | all | 420 | -3.40% | 15.15% | 12.85% | 31.59% | 13.82% | 63.10% | 12.28% | 19.31% |
| financial_only | fwd_12m_return | large_cap | 192 | 7.64% | 27.96% | 25.42% | 52.25% | 31.95% | 79.49% | 10.67% | 41.58% |
| financial_plus_price | fwd_12m_return | large_cap | 192 | 22.84% | 52.02% | 40.20% | 81.17% | 61.18% | 92.31% | 11.86% | 69.31% |
| growth_quality_only | fwd_12m_return | large_cap | 192 | 4.23% | 20.69% | 13.63% | 56.66% | 31.08% | 71.79% | 22.45% | 34.22% |
| growth_quality_price | fwd_12m_return | large_cap | 192 | 13.92% | 43.17% | 35.82% | 76.88% | 61.14% | 87.18% | 12.71% | 64.17% |
| financial_only | fwd_12m_return | mid_cap | 212 | -28.89% | -12.44% | -11.29% | 3.21% | -0.45% | 48.84% | 22.31% | -19.10% |
| financial_plus_price | fwd_12m_return | mid_cap | 212 | -32.97% | -8.81% | -8.98% | 6.01% | 0.06% | 53.49% | 17.17% | -11.15% |
| growth_quality_only | fwd_12m_return | mid_cap | 212 | -16.08% | -15.53% | -11.24% | 2.73% | -0.45% | 48.84% | 18.08% | -15.35% |
| growth_quality_price | fwd_12m_return | mid_cap | 212 | -17.16% | -18.72% | -19.18% | -4.77% | -4.50% | 39.53% | 26.58% | -31.35% |
| financial_only | fwd_3m_return | all | 1011 | -16.57% | -7.68% | -11.14% | 0.65% | 0.58% | 52.22% | 4.58% | -3.93% |
| financial_plus_price | fwd_3m_return | all | 1011 | -8.96% | 17.35% | 18.22% | 7.26% | 2.71% | 60.10% | -1.83% | 9.09% |
| growth_quality_only | fwd_3m_return | all | 1011 | -14.88% | -11.34% | -14.40% | 0.84% | 0.01% | 50.25% | 5.15% | -4.31% |
| growth_quality_price | fwd_3m_return | all | 1011 | -7.78% | 14.24% | 14.66% | 4.89% | 0.93% | 54.19% | -1.38% | 6.26% |
| financial_only | fwd_3m_return | large_cap | 235 | -3.37% | 31.07% | 16.07% | 20.89% | 14.75% | 72.34% | 4.98% | 15.92% |
| financial_plus_price | fwd_3m_return | large_cap | 235 | 3.13% | 23.79% | 17.26% | 16.30% | 12.69% | 74.47% | 6.36% | 9.94% |
| growth_quality_only | fwd_3m_return | large_cap | 235 | -5.26% | 29.59% | 14.83% | 21.57% | 15.72% | 65.96% | 4.18% | 17.39% |
| growth_quality_price | fwd_3m_return | large_cap | 235 | 6.37% | 30.31% | 21.51% | 19.06% | 13.85% | 76.60% | 8.15% | 10.92% |
| financial_only | fwd_3m_return | mid_cap | 507 | -16.82% | -8.65% | -11.29% | -1.63% | -0.56% | 46.08% | 1.42% | -3.04% |
| financial_plus_price | fwd_3m_return | mid_cap | 507 | -16.90% | 12.55% | 12.21% | 2.22% | 0.40% | 50.98% | -3.71% | 5.93% |
| growth_quality_only | fwd_3m_return | mid_cap | 507 | -18.83% | -13.59% | -14.98% | -1.99% | -1.59% | 42.16% | 1.88% | -3.86% |
| growth_quality_price | fwd_3m_return | mid_cap | 507 | -17.12% | 5.10% | 7.32% | 0.28% | -2.04% | 42.16% | -1.39% | 1.68% |
| financial_only | fwd_6m_return | all | 472 | -0.94% | 2.83% | -2.33% | 14.47% | 6.45% | 65.26% | 14.34% | 0.14% |
| financial_plus_price | fwd_6m_return | all | 472 | 18.29% | 47.79% | 35.58% | 40.25% | 27.38% | 85.26% | 2.30% | 37.95% |
| growth_quality_only | fwd_6m_return | all | 472 | -0.84% | 1.56% | -6.31% | 16.83% | 4.73% | 61.05% | 18.03% | -1.19% |
| growth_quality_price | fwd_6m_return | all | 472 | 11.85% | 41.05% | 25.49% | 29.96% | 13.47% | 72.63% | 2.46% | 27.50% |
| financial_only | fwd_6m_return | large_cap | 213 | -2.30% | 16.78% | 6.73% | 34.72% | 12.49% | 72.09% | 24.49% | 10.23% |
| financial_plus_price | fwd_6m_return | large_cap | 213 | 6.06% | 28.15% | 25.44% | 46.50% | 38.29% | 88.37% | 20.08% | 26.42% |
| growth_quality_only | fwd_6m_return | large_cap | 213 | -3.68% | 8.68% | 0.16% | 33.34% | 12.73% | 74.42% | 24.56% | 8.77% |
| growth_quality_price | fwd_6m_return | large_cap | 213 | 8.64% | 36.34% | 30.16% | 42.36% | 28.57% | 83.72% | 13.31% | 29.05% |
| financial_only | fwd_6m_return | mid_cap | 239 | -13.17% | -4.90% | -1.49% | 5.49% | 0.70% | 52.08% | 2.81% | 2.68% |
| financial_plus_price | fwd_6m_return | mid_cap | 239 | -16.87% | 24.35% | 27.20% | 15.33% | 12.86% | 75.00% | -2.20% | 17.53% |
| growth_quality_only | fwd_6m_return | mid_cap | 239 | -9.61% | -5.36% | -1.07% | 5.86% | 0.80% | 52.08% | 6.81% | -0.95% |
| growth_quality_price | fwd_6m_return | mid_cap | 239 | -14.67% | 8.61% | 20.11% | 6.01% | 9.41% | 60.42% | -3.34% | 9.34% |

## Sector-neutral ranking

| feature_set | forward_window | regression_bucket | top_observations | top_mean_return | top_median_return | top_hit_rate | bottom_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| financial_only | fwd_12m_return | all | 90 | 24.22% | 13.95% | 65.56% | 12.58% | 11.65% |
| financial_plus_price | fwd_12m_return | all | 90 | 34.88% | 24.98% | 76.67% | 10.15% | 24.74% |
| growth_quality_only | fwd_12m_return | all | 91 | 28.65% | 19.61% | 70.33% | 10.94% | 17.72% |
| growth_quality_price | fwd_12m_return | all | 90 | 32.46% | 18.17% | 70.00% | 12.43% | 20.04% |
| financial_only | fwd_12m_return | large_cap | 45 | 51.63% | 39.22% | 86.67% | 12.57% | 39.05% |
| financial_plus_price | fwd_12m_return | large_cap | 44 | 61.78% | 46.74% | 88.64% | 8.62% | 53.17% |
| growth_quality_only | fwd_12m_return | large_cap | 44 | 44.33% | 31.52% | 79.55% | 24.52% | 19.82% |
| growth_quality_price | fwd_12m_return | large_cap | 44 | 59.79% | 42.78% | 86.36% | 13.95% | 45.84% |
| financial_only | fwd_12m_return | mid_cap | 47 | 5.51% | 0.06% | 53.19% | 20.35% | -14.84% |
| financial_plus_price | fwd_12m_return | mid_cap | 47 | 5.88% | 2.84% | 57.45% | 13.66% | -7.77% |
| growth_quality_only | fwd_12m_return | mid_cap | 47 | 6.37% | 0.01% | 51.06% | 20.14% | -13.77% |
| growth_quality_price | fwd_12m_return | mid_cap | 47 | 2.03% | 0.06% | 53.19% | 25.96% | -23.93% |
| financial_only | fwd_3m_return | all | 208 | 0.39% | 0.01% | 50.00% | 3.75% | -3.36% |
| financial_plus_price | fwd_3m_return | all | 208 | 7.40% | 3.88% | 63.46% | -0.66% | 8.06% |
| growth_quality_only | fwd_3m_return | all | 208 | -0.37% | -0.38% | 48.56% | 5.58% | -5.95% |
| growth_quality_price | fwd_3m_return | all | 208 | 7.28% | 3.19% | 62.02% | -0.01% | 7.29% |
| financial_only | fwd_3m_return | large_cap | 54 | 18.20% | 11.79% | 72.22% | 3.40% | 14.80% |
| financial_plus_price | fwd_3m_return | large_cap | 53 | 16.70% | 12.69% | 71.70% | 5.05% | 11.64% |
| growth_quality_only | fwd_3m_return | large_cap | 54 | 20.12% | 13.36% | 77.78% | 2.62% | 17.50% |
| growth_quality_price | fwd_3m_return | large_cap | 53 | 22.27% | 15.72% | 84.91% | 6.49% | 15.77% |
| financial_only | fwd_3m_return | mid_cap | 106 | -2.13% | -1.59% | 42.45% | 0.81% | -2.94% |
| financial_plus_price | fwd_3m_return | mid_cap | 106 | 2.33% | 0.78% | 53.77% | -2.80% | 5.14% |
| growth_quality_only | fwd_3m_return | mid_cap | 106 | -1.80% | -1.11% | 42.45% | 0.80% | -2.60% |
| growth_quality_price | fwd_3m_return | mid_cap | 106 | 1.24% | -0.91% | 46.23% | -1.59% | 2.83% |
| financial_only | fwd_6m_return | all | 99 | 21.73% | 10.48% | 66.67% | 13.38% | 8.35% |
| financial_plus_price | fwd_6m_return | all | 99 | 38.10% | 24.42% | 79.80% | 2.30% | 35.79% |
| growth_quality_only | fwd_6m_return | all | 99 | 18.10% | 5.45% | 64.65% | 16.24% | 1.85% |
| growth_quality_price | fwd_6m_return | all | 99 | 29.63% | 15.85% | 74.75% | 2.86% | 26.76% |
| financial_only | fwd_6m_return | large_cap | 47 | 35.18% | 12.49% | 74.47% | 24.47% | 10.70% |
| financial_plus_price | fwd_6m_return | large_cap | 47 | 40.60% | 23.84% | 85.11% | 18.62% | 21.98% |
| growth_quality_only | fwd_6m_return | large_cap | 47 | 33.92% | 17.29% | 74.47% | 19.85% | 14.07% |
| growth_quality_price | fwd_6m_return | large_cap | 47 | 42.30% | 28.57% | 82.98% | 4.94% | 37.36% |
| financial_only | fwd_6m_return | mid_cap | 53 | 4.11% | 0.21% | 50.94% | 7.59% | -3.48% |
| financial_plus_price | fwd_6m_return | mid_cap | 53 | 13.93% | 13.30% | 73.58% | -0.92% | 14.84% |
| growth_quality_only | fwd_6m_return | mid_cap | 53 | 2.38% | -3.04% | 43.40% | 13.27% | -10.89% |
| growth_quality_price | fwd_6m_return | mid_cap | 53 | 5.71% | 9.32% | 62.26% | -1.78% | 7.49% |

## Condition subsets

| feature_set | forward_window | regression_bucket | condition | observations | mean_return | median_return | hit_rate | avg_growth | avg_trailing_6m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| financial_only | fwd_12m_return | all | growth30_positive_margin | 13 | 58.64% | 33.42% | 76.92% | 59.27% | 15.40% |
| financial_plus_price | fwd_12m_return | all | growth30_positive_margin | 13 | 58.64% | 33.42% | 76.92% | 59.27% | 15.40% |
| growth_quality_only | fwd_12m_return | all | growth30_positive_margin | 13 | 58.64% | 33.42% | 76.92% | 59.27% | 15.40% |
| growth_quality_price | fwd_12m_return | all | growth30_positive_margin | 13 | 58.64% | 33.42% | 76.92% | 59.27% | 15.40% |
| financial_only | fwd_12m_return | all | growth30_model_top_half | 14 | 53.35% | 27.95% | 71.43% | 59.04% | 13.75% |
| financial_plus_price | fwd_12m_return | all | model_top_quintile | 84 | 50.58% | 35.76% | 76.19% | 18.65% | 3.62% |
| financial_plus_price | fwd_12m_return | all | growth30_model_top_half | 15 | 49.29% | 22.48% | 66.67% | 57.12% | 12.48% |
| growth_quality_only | fwd_12m_return | all | growth30_model_top_half | 15 | 49.29% | 22.48% | 66.67% | 57.12% | 12.48% |
| growth_quality_price | fwd_12m_return | all | growth30_model_top_half | 15 | 49.29% | 22.48% | 66.67% | 57.12% | 12.48% |
| financial_only | fwd_12m_return | all | growth30_positive_margin_mom | 5 | 38.16% | 46.37% | 80.00% | 52.43% | 57.63% |
| financial_plus_price | fwd_12m_return | all | growth30_positive_margin_mom | 5 | 38.16% | 46.37% | 80.00% | 52.43% | 57.63% |
| growth_quality_only | fwd_12m_return | all | growth30_positive_margin_mom | 5 | 38.16% | 46.37% | 80.00% | 52.43% | 57.63% |
| growth_quality_price | fwd_12m_return | all | growth30_positive_margin_mom | 5 | 38.16% | 46.37% | 80.00% | 52.43% | 57.63% |
| growth_quality_price | fwd_12m_return | all | model_top_quintile | 84 | 31.59% | 13.82% | 63.10% | 22.12% | 2.55% |
| growth_quality_only | fwd_12m_return | all | model_top_quintile | 84 | 29.55% | 11.66% | 64.29% | 20.57% | 2.79% |
| financial_only | fwd_12m_return | all | model_top_quintile | 84 | 25.17% | 9.88% | 63.10% | 19.55% | 3.90% |
| financial_plus_price | fwd_12m_return | large_cap | model_top_quintile | 39 | 81.17% | 61.18% | 92.31% | 23.73% | 7.02% |
| growth_quality_price | fwd_12m_return | large_cap | model_top_quintile | 39 | 76.88% | 61.14% | 87.18% | 23.49% | 3.09% |
| financial_only | fwd_12m_return | large_cap | growth30_positive_margin | 10 | 75.68% | 54.19% | 90.00% | 61.10% | 15.02% |
| financial_plus_price | fwd_12m_return | large_cap | growth30_positive_margin | 10 | 75.68% | 54.19% | 90.00% | 61.10% | 15.02% |
| growth_quality_only | fwd_12m_return | large_cap | growth30_positive_margin | 10 | 75.68% | 54.19% | 90.00% | 61.10% | 15.02% |
| growth_quality_price | fwd_12m_return | large_cap | growth30_positive_margin | 10 | 75.68% | 54.19% | 90.00% | 61.10% | 15.02% |
| growth_quality_only | fwd_12m_return | large_cap | growth30_model_top_half | 8 | 74.33% | 39.22% | 75.00% | 55.70% | 7.56% |
| financial_only | fwd_12m_return | large_cap | growth30_model_top_half | 10 | 71.89% | 54.19% | 80.00% | 63.49% | 14.68% |
| financial_plus_price | fwd_12m_return | large_cap | growth30_model_top_half | 10 | 71.89% | 54.19% | 80.00% | 63.49% | 14.68% |
| growth_quality_price | fwd_12m_return | large_cap | growth30_model_top_half | 11 | 67.39% | 46.37% | 81.82% | 60.64% | 12.96% |
| growth_quality_only | fwd_12m_return | large_cap | model_top_quintile | 39 | 56.66% | 31.08% | 71.79% | 22.20% | 7.36% |
| financial_only | fwd_12m_return | large_cap | model_top_quintile | 39 | 52.25% | 31.95% | 79.49% | 20.52% | 7.47% |
| financial_plus_price | fwd_12m_return | mid_cap | model_top_quintile | 43 | 6.01% | 0.06% | 53.49% | 9.68% | 3.39% |
| financial_only | fwd_12m_return | mid_cap | model_top_quintile | 43 | 3.21% | -0.45% | 48.84% | 8.66% | 0.08% |
| growth_quality_only | fwd_12m_return | mid_cap | model_top_quintile | 43 | 2.73% | -0.45% | 48.84% | 8.79% | -1.11% |
| growth_quality_price | fwd_12m_return | mid_cap | model_top_quintile | 43 | -4.77% | -4.50% | 39.53% | 9.44% | 2.37% |
| growth_quality_price | fwd_3m_return | all | growth30_model_top_half | 21 | 19.65% | 14.75% | 61.90% | 51.65% | 18.63% |
| financial_plus_price | fwd_3m_return | all | growth30_model_top_half | 19 | 17.51% | 14.75% | 57.89% | 54.87% | 11.61% |
| financial_only | fwd_3m_return | all | growth30_positive_margin | 39 | 14.31% | 6.00% | 61.54% | 67.97% | 31.23% |
| financial_plus_price | fwd_3m_return | all | growth30_positive_margin | 39 | 14.31% | 6.00% | 61.54% | 67.97% | 31.23% |
| growth_quality_only | fwd_3m_return | all | growth30_positive_margin | 39 | 14.31% | 6.00% | 61.54% | 67.97% | 31.23% |
| growth_quality_price | fwd_3m_return | all | growth30_positive_margin | 39 | 14.31% | 6.00% | 61.54% | 67.97% | 31.23% |
| growth_quality_only | fwd_3m_return | all | growth30_model_top_half | 17 | 14.18% | 4.60% | 52.94% | 54.16% | 45.06% |
| financial_only | fwd_3m_return | all | growth30_positive_margin_mom | 27 | 14.05% | 11.00% | 66.67% | 68.96% | 52.36% |
| financial_plus_price | fwd_3m_return | all | growth30_positive_margin_mom | 27 | 14.05% | 11.00% | 66.67% | 68.96% | 52.36% |
| growth_quality_only | fwd_3m_return | all | growth30_positive_margin_mom | 27 | 14.05% | 11.00% | 66.67% | 68.96% | 52.36% |
| growth_quality_price | fwd_3m_return | all | growth30_positive_margin_mom | 27 | 14.05% | 11.00% | 66.67% | 68.96% | 52.36% |
| financial_only | fwd_3m_return | all | growth30_model_top_half | 24 | 13.02% | 10.18% | 54.17% | 56.83% | 40.25% |
| financial_plus_price | fwd_3m_return | all | model_top_quintile | 203 | 7.26% | 2.71% | 60.10% | 6.61% | 14.47% |
| growth_quality_price | fwd_3m_return | all | model_top_quintile | 203 | 4.89% | 0.93% | 54.19% | 6.95% | 17.70% |
| growth_quality_only | fwd_3m_return | all | model_top_quintile | 203 | 0.84% | 0.01% | 50.25% | -1.22% | 22.46% |
| financial_only | fwd_3m_return | all | model_top_quintile | 203 | 0.65% | 0.58% | 52.22% | -0.57% | 21.74% |
| financial_only | fwd_3m_return | large_cap | growth30_positive_margin_mom | 6 | 47.79% | 40.55% | 83.33% | 52.61% | 122.56% |
| financial_plus_price | fwd_3m_return | large_cap | growth30_positive_margin_mom | 6 | 47.79% | 40.55% | 83.33% | 52.61% | 122.56% |
| growth_quality_only | fwd_3m_return | large_cap | growth30_positive_margin_mom | 6 | 47.79% | 40.55% | 83.33% | 52.61% | 122.56% |
| growth_quality_price | fwd_3m_return | large_cap | growth30_positive_margin_mom | 6 | 47.79% | 40.55% | 83.33% | 52.61% | 122.56% |
| financial_only | fwd_3m_return | large_cap | growth30_positive_margin | 13 | 39.90% | 27.07% | 76.92% | 57.65% | 50.39% |
| financial_plus_price | fwd_3m_return | large_cap | growth30_positive_margin | 13 | 39.90% | 27.07% | 76.92% | 57.65% | 50.39% |
| growth_quality_only | fwd_3m_return | large_cap | growth30_positive_margin | 13 | 39.90% | 27.07% | 76.92% | 57.65% | 50.39% |
| growth_quality_price | fwd_3m_return | large_cap | growth30_positive_margin | 13 | 39.90% | 27.07% | 76.92% | 57.65% | 50.39% |
| growth_quality_only | fwd_3m_return | large_cap | growth30_model_top_half | 14 | 37.77% | 25.27% | 78.57% | 57.53% | 46.24% |
| growth_quality_price | fwd_3m_return | large_cap | growth30_model_top_half | 11 | 34.95% | 22.42% | 72.73% | 60.64% | 12.96% |
| financial_only | fwd_3m_return | large_cap | growth30_model_top_half | 10 | 34.65% | 25.27% | 80.00% | 62.41% | 59.23% |
| financial_plus_price | fwd_3m_return | large_cap | growth30_model_top_half | 10 | 34.11% | 24.74% | 80.00% | 62.55% | 20.65% |

## Interpretation

- The original financial-only large-cap 12m model remains hard to beat on rank correlation.
- Price context helps some shorter-horizon all-stock metrics but does not improve the large-cap 12m Spearman result.
- Sector-neutral ranking is useful as a robustness check but did not produce a clear improvement over the broad large-cap top quintile.
- Best use remains a large-cap 30%+ growth/profitability filter combined with the financial-only 12m ranking score.
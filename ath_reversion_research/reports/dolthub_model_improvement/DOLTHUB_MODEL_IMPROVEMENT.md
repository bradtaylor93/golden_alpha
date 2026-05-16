# DoltHub Model Improvement

Compares feature subsets on the deeper DoltHub annual sample.

## Performance

| feature_set | forward_window | observations | test_years | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| price_only | fwd_12m_return | 4605 | 10 | 5.11% | 23.03% | 22.79% | 43.59% | 31.54% | 78.28% | 7.27% | 0.3632589717991299 |
| income_price | fwd_12m_return | 4605 | 10 | 5.36% | 23.48% | 22.29% | 43.88% | 32.37% | 78.61% | 7.52% | 0.3636023723246887 |
| full | fwd_12m_return | 4605 | 10 | 3.63% | 20.89% | 21.17% | 36.27% | 24.89% | 73.72% | 6.29% | 0.2997782624985755 |
| growth_quality_price | fwd_12m_return | 4605 | 10 | 1.83% | 18.02% | 17.44% | 27.67% | 17.40% | 70.90% | 7.64% | 0.20028402922307997 |
| income_only | fwd_12m_return | 4605 | 10 | -0.31% | 6.04% | -0.64% | 19.52% | 9.59% | 63.74% | 14.67% | 0.04857563200785214 |
| price_only | fwd_3m_return | 5122 | 12 | -2.00% | 4.87% | 4.68% | 6.51% | 5.79% | 63.22% | 2.97% | 0.03544360071410452 |
| income_price | fwd_3m_return | 5122 | 12 | -4.47% | 0.33% | 0.54% | 5.64% | 4.58% | 61.37% | 3.76% | 0.018800375730478942 |
| growth_quality_price | fwd_3m_return | 5122 | 12 | -5.29% | -0.17% | -0.68% | 5.22% | 3.38% | 57.95% | 4.95% | 0.0027267280716030615 |
| full | fwd_3m_return | 5122 | 12 | -5.31% | 0.05% | -1.33% | 5.63% | 3.52% | 58.44% | 4.29% | 0.013358447801101639 |
| income_only | fwd_3m_return | 5122 | 12 | -4.14% | -11.25% | -12.89% | 0.12% | -0.81% | 46.93% | 5.07% | -0.04957199704550599 |
| full | fwd_6m_return | 5080 | 11 | -4.29% | 5.01% | -4.77% | 11.67% | 5.96% | 58.86% | 9.02% | 0.02646071911206556 |
| income_price | fwd_6m_return | 5080 | 11 | -4.14% | 3.72% | -5.24% | 12.56% | 7.84% | 61.91% | 9.06% | 0.035027736311417354 |
| price_only | fwd_6m_return | 5080 | 11 | -2.78% | 5.55% | -5.64% | 13.78% | 9.22% | 63.09% | 9.24% | 0.0453555906211045 |
| growth_quality_price | fwd_6m_return | 5080 | 11 | -5.17% | 2.77% | -6.31% | 10.23% | 5.31% | 58.37% | 9.28% | 0.009504679379668868 |
| income_only | fwd_6m_return | 5080 | 11 | -4.22% | -8.87% | -12.91% | 2.44% | 0.67% | 51.28% | 9.89% | -0.07444558697431257 |

## Conditions

| feature_set | forward_window | condition | observations | mean_return | median_return | hit_rate |
| --- | --- | --- | --- | --- | --- | --- |
| income_price | fwd_12m_return | model_top_quintile | 921 | 43.88% | 32.37% | 78.61% |
| price_only | fwd_12m_return | model_top_quintile | 921 | 43.59% | 31.54% | 78.28% |
| full | fwd_12m_return | model_top_quintile | 921 | 36.27% | 24.89% | 73.72% |
| income_only | fwd_12m_return | sales_growth_30pct_top_half | 137 | 32.36% | 16.62% | 68.61% |
| growth_quality_price | fwd_12m_return | sales_growth_30pct_top_half | 218 | 31.17% | 19.71% | 71.56% |
| growth_quality_price | fwd_12m_return | model_top_quintile | 921 | 27.67% | 17.40% | 70.90% |
| full | fwd_12m_return | sales_growth_30pct_top_half | 219 | 27.52% | 18.03% | 68.95% |
| income_price | fwd_12m_return | sales_growth_30pct_top_half | 230 | 25.47% | 15.02% | 67.83% |
| price_only | fwd_12m_return | sales_growth_30pct_top_half | 270 | 23.48% | 13.94% | 65.19% |
| full | fwd_12m_return | sales_growth_30pct | 401 | 21.30% | 14.75% | 65.84% |
| growth_quality_price | fwd_12m_return | sales_growth_30pct | 401 | 21.30% | 14.75% | 65.84% |
| income_only | fwd_12m_return | sales_growth_30pct | 401 | 21.30% | 14.75% | 65.84% |
| income_price | fwd_12m_return | sales_growth_30pct | 401 | 21.30% | 14.75% | 65.84% |
| price_only | fwd_12m_return | sales_growth_30pct | 401 | 21.30% | 14.75% | 65.84% |
| income_only | fwd_12m_return | model_top_quintile | 921 | 19.52% | 9.59% | 63.74% |
| price_only | fwd_3m_return | model_top_quintile | 1025 | 6.51% | 5.79% | 63.22% |
| income_price | fwd_3m_return | model_top_quintile | 1025 | 5.64% | 4.58% | 61.37% |
| full | fwd_3m_return | model_top_quintile | 1025 | 5.63% | 3.52% | 58.44% |
| growth_quality_price | fwd_3m_return | model_top_quintile | 1025 | 5.22% | 3.38% | 57.95% |
| full | fwd_3m_return | sales_growth_30pct | 441 | 3.69% | 2.09% | 53.74% |
| growth_quality_price | fwd_3m_return | sales_growth_30pct | 441 | 3.69% | 2.09% | 53.74% |
| income_only | fwd_3m_return | sales_growth_30pct | 441 | 3.69% | 2.09% | 53.74% |
| income_price | fwd_3m_return | sales_growth_30pct | 441 | 3.69% | 2.09% | 53.74% |
| price_only | fwd_3m_return | sales_growth_30pct | 441 | 3.69% | 2.09% | 53.74% |
| price_only | fwd_3m_return | sales_growth_30pct_top_half | 288 | 2.78% | -0.57% | 48.96% |
| income_price | fwd_3m_return | sales_growth_30pct_top_half | 263 | 2.17% | -0.60% | 48.67% |
| full | fwd_3m_return | sales_growth_30pct_top_half | 265 | 1.30% | -0.55% | 49.06% |
| income_only | fwd_3m_return | sales_growth_30pct_top_half | 254 | 0.97% | -0.57% | 48.82% |
| growth_quality_price | fwd_3m_return | sales_growth_30pct_top_half | 254 | 0.89% | -0.60% | 48.03% |
| income_only | fwd_3m_return | model_top_quintile | 1025 | 0.12% | -0.81% | 46.93% |
| price_only | fwd_6m_return | model_top_quintile | 1016 | 13.78% | 9.22% | 63.09% |
| income_price | fwd_6m_return | model_top_quintile | 1016 | 12.56% | 7.84% | 61.91% |
| full | fwd_6m_return | model_top_quintile | 1016 | 11.67% | 5.96% | 58.86% |
| growth_quality_price | fwd_6m_return | model_top_quintile | 1016 | 10.23% | 5.31% | 58.37% |
| full | fwd_6m_return | sales_growth_30pct | 440 | 8.06% | 5.44% | 59.09% |
| growth_quality_price | fwd_6m_return | sales_growth_30pct | 440 | 8.06% | 5.44% | 59.09% |
| income_only | fwd_6m_return | sales_growth_30pct | 440 | 8.06% | 5.44% | 59.09% |
| income_price | fwd_6m_return | sales_growth_30pct | 440 | 8.06% | 5.44% | 59.09% |
| price_only | fwd_6m_return | sales_growth_30pct | 440 | 8.06% | 5.44% | 59.09% |
| income_price | fwd_6m_return | sales_growth_30pct_top_half | 242 | 6.62% | -2.17% | 47.93% |
| price_only | fwd_6m_return | sales_growth_30pct_top_half | 294 | 6.45% | 0.50% | 51.02% |
| full | fwd_6m_return | sales_growth_30pct_top_half | 235 | 5.70% | -2.05% | 48.51% |
| growth_quality_price | fwd_6m_return | sales_growth_30pct_top_half | 226 | 4.86% | -3.16% | 47.79% |
| income_only | fwd_6m_return | model_top_quintile | 1016 | 2.44% | 0.67% | 51.28% |
| income_only | fwd_6m_return | sales_growth_30pct_top_half | 212 | 1.28% | -3.92% | 45.75% |

## Interpretation

- On the longer Dolt sample, price-only and income+price are very close for 12m ranking.
- Adding balance sheet/cash-flow ratios did not improve the model; the full feature set underperformed income+price.
- The longer-history validation supports a simple 12m annual price/fundamentals signal, but not an increasingly complex feature stack.
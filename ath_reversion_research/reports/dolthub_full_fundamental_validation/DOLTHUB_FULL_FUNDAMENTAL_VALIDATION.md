# Full Local Dolt Fundamentals Validation

Uses local DoltHub earnings clones for income, balance sheet, and cash flow fundamentals. This run still uses Yahoo adjusted prices because local DoltHub OHLCV needs indexing/export for this universe size.

## Data

- Feature events: 5871.
- Prediction rows: 14807.
- Symbols with predictions: 495.
- Test years: [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026].

## Leakage controls

- Annual fundamentals assumed available 90 days after fiscal year end.
- Forward returns start after availability.
- Training rows are purged unless their forward-return end date is before the test year starts.
- Uses local Dolt fundamentals; this run uses Yahoo adjusted prices because local Dolt OHLCV needs indexing/export for this universe size.

## Performance

| forward_window | observations | test_years | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fwd_12m_return | 4605 | 10 | 3.63% | 20.89% | 21.17% | 36.27% | 24.89% | 73.72% | 6.29% | 0.2997782624985755 |
| fwd_3m_return | 5122 | 12 | -5.31% | 0.05% | -1.33% | 5.63% | 3.52% | 58.44% | 4.29% | 0.013358447801101618 |
| fwd_6m_return | 5080 | 11 | -4.29% | 5.01% | -4.77% | 11.67% | 5.96% | 58.86% | 9.02% | 0.02646071911206556 |

## Conditions

| forward_window | condition | observations | mean_return | median_return | hit_rate | avg_revenue_growth |
| --- | --- | --- | --- | --- | --- | --- |
| fwd_12m_return | model_top_quintile | 921 | 36.27% | 24.89% | 73.72% | 15.22% |
| fwd_12m_return | revenue_growth_30pct_top_half | 219 | 27.52% | 18.03% | 68.95% | 80.75% |
| fwd_12m_return | revenue_growth_30pct | 401 | 21.30% | 14.75% | 65.84% | 179.98% |
| fwd_3m_return | model_top_quintile | 1025 | 5.63% | 3.52% | 58.44% | 17.00% |
| fwd_3m_return | revenue_growth_30pct | 441 | 3.69% | 2.09% | 53.74% | 169.87% |
| fwd_3m_return | revenue_growth_30pct_top_half | 265 | 1.30% | -0.55% | 49.06% | 78.61% |
| fwd_6m_return | model_top_quintile | 1016 | 11.67% | 5.96% | 58.86% | 15.97% |
| fwd_6m_return | revenue_growth_30pct | 440 | 8.06% | 5.44% | 59.09% | 170.15% |
| fwd_6m_return | revenue_growth_30pct_top_half | 235 | 5.70% | -2.05% | 48.51% | 78.12% |

## Interpretation

- This is the closest open-data validation pass so far for fundamentals: local Dolt income, balance sheet, and cash flow.
- It includes balance sheet and cash-flow features, but still uses current S&P 500 symbols.
- If 12m predictive power remains positive, the next step is historical constituents/delisted universe construction.
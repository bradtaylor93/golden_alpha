# Full Local Dolt Fundamentals Validation

Uses local DoltHub earnings and stocks clones for fundamentals and OHLCV.

## Data

- Feature events: 5791.
- Prediction rows: 14628.
- Symbols with predictions: 492.
- Test years: [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026].

## Leakage controls

- Annual fundamentals assumed available 90 days after fiscal year end.
- Forward returns start after availability.
- Training rows are purged unless their forward-return end date is before the test year starts.
- Uses local Dolt OHLCV instead of Yahoo for prices.

## Performance

| forward_window | observations | test_years | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fwd_12m_return | 4556 | 10 | -0.20% | 8.71% | 20.96% | 33.68% | 21.97% | 71.38% | 8.45% | 0.25228182510948316 |
| fwd_3m_return | 5057 | 12 | -1.83% | 0.04% | 3.82% | 5.84% | 4.42% | 61.86% | 2.29% | 0.035542305694307275 |
| fwd_6m_return | 5015 | 11 | -2.26% | 0.54% | -3.52% | 10.65% | 5.98% | 59.82% | 6.57% | 0.040758567267051404 |

## Conditions

| forward_window | condition | observations | mean_return | median_return | hit_rate | avg_revenue_growth |
| --- | --- | --- | --- | --- | --- | --- |
| fwd_12m_return | model_top_quintile | 912 | 33.68% | 21.97% | 71.38% | 15.49% |
| fwd_12m_return | revenue_growth_30pct_top_half | 208 | 24.45% | 15.54% | 63.94% | 83.65% |
| fwd_12m_return | revenue_growth_30pct | 389 | 22.81% | 11.64% | 62.72% | 181.22% |
| fwd_3m_return | model_top_quintile | 1012 | 5.84% | 4.42% | 61.86% | 15.55% |
| fwd_3m_return | revenue_growth_30pct | 428 | 5.32% | 1.21% | 51.87% | 171.01% |
| fwd_3m_return | revenue_growth_30pct_top_half | 261 | 2.24% | 0.38% | 50.57% | 76.72% |
| fwd_6m_return | model_top_quintile | 1003 | 10.65% | 5.98% | 59.82% | 15.62% |
| fwd_6m_return | revenue_growth_30pct_top_half | 232 | 9.60% | -4.94% | 45.26% | 77.63% |
| fwd_6m_return | revenue_growth_30pct | 427 | 8.66% | 4.47% | 56.67% | 171.29% |

## Interpretation

- This is the closest open-data validation pass so far: local Dolt fundamentals plus local Dolt OHLCV.
- It includes balance sheet and cash-flow features, but still uses current S&P 500 symbols.
- If 12m predictive power remains positive, the next step is historical constituents/delisted universe construction.
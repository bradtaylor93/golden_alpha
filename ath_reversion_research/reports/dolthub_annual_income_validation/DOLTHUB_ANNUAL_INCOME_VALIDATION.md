# DoltHub Annual Income Validation

First validation pass using DoltHub annual income statements and Yahoo adjusted prices.

## Data

- Income-statement events: 5871.
- Prediction rows: 14807.
- Symbols with predictions: 495.
- Test years: [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026].

## Leakage controls

- Annual income statements are assumed available 90 days after fiscal year end.
- Forward returns start after that availability date.
- Training rows are purged unless their forward-return end date is before the test year starts.
- DoltHub is used for fundamentals; Yahoo is still used for prices in this first pass due DoltHub OHLCV API timeouts.

## Performance

| forward_window | observations | test_years | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fwd_12m_return | 4605 | 10 | 5.36% | 23.48% | 22.29% | 43.88% | 32.37% | 78.61% | 7.52% | 0.3636023747312773 |
| fwd_3m_return | 5122 | 12 | -4.47% | 0.33% | 0.54% | 5.64% | 4.58% | 61.37% | 3.76% | 0.0188003907480073 |
| fwd_6m_return | 5080 | 11 | -4.14% | 3.72% | -5.24% | 12.56% | 7.84% | 61.91% | 9.06% | 0.0350277151353644 |

## Conditions

| forward_window | condition | observations | mean_return | median_return | hit_rate | avg_sales_growth | avg_operating_margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fwd_12m_return | model_top_quintile | 921 | 43.88% | 32.37% | 78.61% | 13.88% | 9.85% |
| fwd_12m_return | sales_growth_30pct_top_half | 230 | 25.47% | 15.02% | 67.83% | 81.14% | 15.45% |
| fwd_12m_return | sales_growth_30pct | 401 | 21.30% | 14.75% | 65.84% | 179.98% | 16.75% |
| fwd_12m_return | sales_growth_30pct_positive_margin | 343 | 18.80% | 14.16% | 65.89% | 193.51% | 21.62% |
| fwd_3m_return | model_top_quintile | 1025 | 5.64% | 4.58% | 61.37% | 15.46% | 9.15% |
| fwd_3m_return | sales_growth_30pct | 441 | 3.69% | 2.09% | 53.74% | 169.87% | 16.17% |
| fwd_3m_return | sales_growth_30pct_positive_margin | 375 | 3.01% | 1.78% | 52.80% | 183.17% | 21.37% |
| fwd_3m_return | sales_growth_30pct_top_half | 263 | 2.17% | -0.60% | 48.67% | 236.17% | 11.91% |
| fwd_6m_return | model_top_quintile | 1016 | 12.56% | 7.84% | 61.91% | 13.64% | 11.19% |
| fwd_6m_return | sales_growth_30pct | 440 | 8.06% | 5.44% | 59.09% | 170.15% | 16.15% |
| fwd_6m_return | sales_growth_30pct_positive_margin | 374 | 7.22% | 5.16% | 58.56% | 183.53% | 21.35% |
| fwd_6m_return | sales_growth_30pct_top_half | 242 | 6.62% | -2.17% | 47.93% | 246.26% | 14.18% |

## Interpretation

- DoltHub materially increases annual statement history versus Yahoo.
- This first pass uses income-statement features only, so it is not apples-to-apples with the richer Yahoo financial+price model.
- If predictive power is positive here, the next step is cloning Dolt locally or using a PIT vendor to add balance sheet/cash-flow fields and faster price access.
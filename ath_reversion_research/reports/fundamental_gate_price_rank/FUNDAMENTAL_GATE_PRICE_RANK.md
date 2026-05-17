# Fundamental Gates on Price Rank

Tests using fundamentals as gates on top of the Dolt price-only large-cap 12m ranker.

## Event-level results

| gate | events | avg_names_per_trade_date | mean_return | median_return | hit_rate | avg_revenue_growth | avg_operating_margin | avg_fcf_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| price_only_ungated | 463 | 4.53921568627451 | 36.67% | 22.09% | 70.63% | 14.21% | 13.86% | 16.46% |
| growth_positive | 459 | 4.59 | 33.55% | 20.55% | 72.98% | 22.43% | 18.38% | 18.64% |
| growth_and_profit | 459 | 4.59 | 28.49% | 20.46% | 73.20% | 18.92% | 21.18% | 19.59% |
| growth_profit_fcf | 459 | 4.59 | 27.02% | 19.87% | 72.77% | 17.82% | 21.32% | 21.88% |
| growth_profit_low_debt | 458 | 4.626262626262626 | 26.18% | 19.45% | 73.58% | 19.92% | 21.01% | 18.58% |
| growth_profit_fcf_low_debt | 458 | 4.626262626262626 | 25.98% | 20.01% | 73.36% | 18.80% | 21.05% | 20.99% |
| growth30_profit_fcf | 160 | 4.102564102564102 | 20.82% | 15.02% | 69.38% | 323.44% | 24.93% | 22.57% |

## Portfolio results

| observations | total_return | annual_return | annual_std | sharpe | average_daily_return | average_daily_turnover | max_drawdown | hit_rate | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2589 | 1081.26% | 27.17% | 24.85% | 1.09 | 0.11% | 1.46% | -33.88% | 55.23% | price_only_ungated |
| 2589 | 1056.62% | 26.91% | 32.33% | 0.83 | 0.12% | 1.37% | -51.52% | 53.53% | growth30_profit_fcf |
| 2589 | 1014.88% | 26.45% | 24.12% | 1.10 | 0.10% | 1.47% | -33.91% | 55.89% | growth_positive |
| 2589 | 844.54% | 24.43% | 22.55% | 1.08 | 0.10% | 1.47% | -31.54% | 54.85% | growth_profit_low_debt |
| 2589 | 830.83% | 24.25% | 22.52% | 1.08 | 0.10% | 1.47% | -31.74% | 54.92% | growth_profit_fcf_low_debt |
| 2589 | 778.11% | 23.55% | 23.80% | 0.99 | 0.10% | 1.46% | -35.84% | 55.12% | growth_and_profit |
| 2589 | 762.50% | 23.33% | 23.76% | 0.98 | 0.09% | 1.46% | -35.85% | 55.04% | growth_profit_fcf |

## Interpretation

- If gates improve event returns or portfolio metrics, fundamentals are useful as filters even when not useful as continuous model predictors.
- Strict high-growth gates can reduce breadth; portfolio results matter more than event means.
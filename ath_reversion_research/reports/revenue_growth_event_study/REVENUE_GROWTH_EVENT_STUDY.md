# Revenue Growth Event Study

Question: does YoY annual revenue growth predict 3m, 6m, or 12m stock returns, and is there an opportunity in 30%+ growers?

## Leakage controls

- Annual revenue is assumed tradable only 90 calendar days after fiscal year end.
- Forward returns are measured from the first trading day on or after that availability date.
- The study does not use the fiscal period-end date as the trade date.
- Exact filing timestamps are not available from Yahoo Finance here; this is a conservative lag approximation, not point-in-time fundamentals.

## Dataset

- Events with at least one forward-return window: 531.
- Symbols with usable events: 177.
- Trade-date range: 2023-06-29 00:00:00 to 2026-05-01 00:00:00.

## Correlation between revenue growth and future returns

| forward_window | observations | pearson_corr | spearman_corr |
| --- | --- | --- | --- |
| fwd_3m_return | 385 | 20.77% | 11.73% |
| fwd_6m_return | 371 | -1.35% | 4.60% |
| fwd_12m_return | 354 | -5.84% | -6.36% |

## Revenue growth segment performance

| growth_segment | observations | avg_revenue_growth | fwd_3m_return_mean | fwd_3m_return_median | fwd_3m_return_hit_rate | fwd_6m_return_mean | fwd_6m_return_median | fwd_6m_return_hit_rate | fwd_12m_return_mean | fwd_12m_return_median | fwd_12m_return_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <0% | 101 | -10.00% | -0.37% | 1.45% | 43.56% | 18.12% | 7.37% | 52.48% | 18.37% | 10.13% | 53.47% |
| 0-10% | 209 | 5.13% | 3.37% | 0.95% | 37.80% | 9.48% | 6.14% | 44.02% | 15.04% | 11.76% | 43.54% |
| 10-20% | 129 | 14.09% | 3.23% | 1.07% | 40.31% | 7.36% | 4.08% | 43.41% | 9.33% | 7.41% | 40.31% |
| 20-30% | 49 | 24.36% | 13.14% | 14.66% | 48.98% | 16.10% | 15.59% | 46.94% | 6.83% | 3.15% | 32.65% |
| 30-50% | 29 | 37.15% | 6.92% | -2.36% | 34.48% | 6.12% | 3.97% | 44.83% | 7.57% | 7.50% | 37.93% |
| 50%+ | 14 | 83.40% | 18.42% | 12.40% | 50.00% | 29.04% | 27.42% | 57.14% | 27.05% | 11.02% | 42.86% |

## 30%+ revenue growth filter

| segment | observations | avg_revenue_growth | median_revenue_growth | fwd_3m_return_mean | fwd_3m_return_median | fwd_3m_return_hit_rate | fwd_6m_return_mean | fwd_6m_return_median | fwd_6m_return_hit_rate | fwd_12m_return_mean | fwd_12m_return_median | fwd_12m_return_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| >=30% revenue growth | 43 | 52.21% | 42.09% | 10.63% | 6.71% | 39.53% | 13.76% | 8.09% | 48.84% | 14.06% | 7.50% | 39.53% |
| <30% revenue growth | 488 | 6.30% | 6.98% | 3.38% | 1.41% | 40.78% | 11.62% | 6.68% | 45.90% | 13.58% | 9.25% | 43.65% |

## Growth deciles

| growth_decile | observations | min_growth | max_growth | avg_growth | fwd_3m_return_mean | fwd_3m_return_median | fwd_6m_return_mean | fwd_6m_return_median | fwd_12m_return_mean | fwd_12m_return_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 54 | -69.71% | -3.35% | -17.29% | -0.01% | 2.56% | 25.51% | 7.47% | 23.86% | 9.19% |
| 2 | 53 | -3.25% | 0.33% | -1.43% | -1.32% | -4.06% | 6.29% | 3.54% | 8.74% | 7.71% |
| 3 | 53 | 0.42% | 2.90% | 1.67% | 3.96% | 1.56% | 10.84% | 4.95% | 12.57% | 8.32% |
| 4 | 53 | 2.97% | 5.63% | 4.29% | 2.34% | 0.95% | 7.15% | 7.83% | 18.38% | 16.63% |
| 5 | 53 | 5.64% | 7.86% | 6.75% | 2.83% | 0.42% | 10.47% | 6.68% | 13.98% | 11.30% |
| 6 | 53 | 7.88% | 10.51% | 9.22% | 6.23% | 6.81% | 12.06% | 7.47% | 18.64% | 13.23% |
| 7 | 53 | 10.53% | 13.69% | 11.94% | 2.12% | 0.41% | 4.55% | 3.57% | 8.48% | 8.02% |
| 8 | 53 | 13.72% | 17.96% | 15.61% | 2.13% | 0.90% | 6.43% | 6.35% | 6.76% | 2.66% |
| 9 | 53 | 17.96% | 27.06% | 22.09% | 11.01% | 13.79% | 13.68% | 15.08% | 9.73% | 7.24% |
| 10 | 53 | 27.68% | 157.37% | 47.80% | 12.17% | 7.54% | 17.51% | 11.77% | 13.65% | 8.03% |

## Interpretation

- On this Yahoo-based sample, the correlation between annual revenue growth and later returns is weak and not stable enough to trade alone.
- The 30%+ growth filter identifies higher-growth companies, but the return advantage is not uniformly stronger across 3m/6m/12m horizons.
- The opportunity, if any, is likely in combining revenue growth with quality, valuation, margin expansion, or price momentum rather than using revenue growth as a standalone signal.
- Results are limited by annual-statement history depth, current-universe survivorship bias, and approximate reporting availability dates.
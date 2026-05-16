# DoltHub Rich Feature Model

Tests richer annual fundamental trend, quality, sector-relative, and interaction features on the DoltHub aligned sample.

## Predictive performance

| feature_set | forward_window | regression_bucket | observations | test_years | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| financial_price | fwd_12m_return | all | 4605 | 10 | 3.63% | 20.89% | 21.17% | 36.27% | 24.89% | 73.72% | 6.29% | 0.2997782624985755 |
| price_only | fwd_12m_return | all | 4605 | 10 | 4.85% | 22.62% | 22.74% | 42.28% | 30.89% | 78.28% | 6.77% | 0.3550426527419078 |
| quality_growth_price | fwd_12m_return | all | 4605 | 10 | 5.02% | 23.95% | 23.02% | 34.17% | 22.18% | 74.38% | 6.71% | 0.27457792170076006 |
| rich_fundamental | fwd_12m_return | all | 4605 | 10 | -1.37% | 13.16% | 10.21% | 24.16% | 13.73% | 69.06% | 9.48% | 0.14679644477066284 |
| rich_fundamental_price | fwd_12m_return | all | 4605 | 10 | 4.22% | 22.59% | 21.09% | 38.49% | 25.78% | 74.27% | 6.18% | 0.32310057326504554 |
| financial_price | fwd_12m_return | large_cap | 2101 | 10 | 5.76% | 26.15% | 24.72% | 46.14% | 31.46% | 76.72% | 9.68% | 0.36464196782950076 |
| price_only | fwd_12m_return | large_cap | 2101 | 10 | 6.21% | 25.89% | 25.45% | 50.20% | 34.12% | 79.57% | 10.14% | 0.40058411885290973 |
| quality_growth_price | fwd_12m_return | large_cap | 2101 | 10 | 6.69% | 28.57% | 24.39% | 45.23% | 29.61% | 78.15% | 12.79% | 0.3244256995846446 |
| rich_fundamental | fwd_12m_return | large_cap | 2101 | 10 | -3.84% | 14.89% | 12.78% | 31.35% | 19.28% | 75.06% | 14.00% | 0.17351141338178655 |
| rich_fundamental_price | fwd_12m_return | large_cap | 2101 | 10 | 3.40% | 24.01% | 22.21% | 43.00% | 30.82% | 75.53% | 10.44% | 0.32563092099826185 |
| financial_price | fwd_12m_return | mid_cap | 2302 | 10 | -0.48% | 14.47% | 17.79% | 24.82% | 13.94% | 69.85% | 6.34% | 0.1847951994410308 |
| price_only | fwd_12m_return | mid_cap | 2302 | 10 | 1.17% | 16.14% | 18.73% | 29.33% | 19.08% | 71.58% | 6.55% | 0.22783448728106326 |
| quality_growth_price | fwd_12m_return | mid_cap | 2302 | 10 | 1.67% | 17.61% | 17.89% | 22.46% | 15.72% | 69.63% | 6.26% | 0.16194959976794282 |
| rich_fundamental | fwd_12m_return | mid_cap | 2302 | 10 | -2.93% | 7.13% | 5.31% | 18.34% | 11.55% | 67.68% | 10.93% | 0.0741207058856206 |
| rich_fundamental_price | fwd_12m_return | mid_cap | 2302 | 10 | -1.29% | 14.53% | 15.63% | 24.77% | 15.14% | 70.93% | 6.61% | 0.1815542020461841 |
| financial_price | fwd_3m_return | all | 5122 | 12 | -5.31% | 0.05% | -1.33% | 5.63% | 3.52% | 58.44% | 4.29% | 0.013358447801101639 |
| price_only | fwd_3m_return | all | 5122 | 12 | -2.45% | 4.39% | 2.19% | 6.70% | 5.43% | 63.02% | 3.65% | 0.030525319331526114 |
| quality_growth_price | fwd_3m_return | all | 5122 | 12 | -3.70% | 5.10% | 6.28% | 6.18% | 5.21% | 63.41% | 3.74% | 0.024431777908337463 |
| rich_fundamental | fwd_3m_return | all | 5122 | 12 | -7.32% | -3.74% | -4.89% | 3.67% | 2.17% | 56.39% | 4.37% | -0.007002216406641025 |
| rich_fundamental_price | fwd_3m_return | all | 5122 | 12 | -9.79% | -3.89% | -3.33% | 3.82% | 1.86% | 54.93% | 4.77% | -0.009543858096146204 |
| financial_price | fwd_3m_return | large_cap | 2336 | 12 | -5.22% | 2.58% | 0.55% | 5.88% | 4.47% | 60.04% | 4.27% | 0.01609594321721705 |
| price_only | fwd_3m_return | large_cap | 2336 | 12 | -1.76% | 6.64% | 2.73% | 9.23% | 7.22% | 67.52% | 4.74% | 0.044951086562166835 |
| quality_growth_price | fwd_3m_return | large_cap | 2336 | 12 | -4.29% | 5.32% | 4.34% | 7.83% | 6.04% | 63.25% | 4.08% | 0.03755303143334069 |
| rich_fundamental | fwd_3m_return | large_cap | 2336 | 12 | -7.57% | -2.64% | -4.81% | 4.62% | 2.96% | 56.84% | 5.31% | -0.006942991480771495 |
| rich_fundamental_price | fwd_3m_return | large_cap | 2336 | 12 | -11.13% | -2.59% | -4.13% | 4.56% | 2.77% | 54.91% | 5.20% | -0.006448196306873731 |
| financial_price | fwd_3m_return | mid_cap | 2559 | 12 | -7.56% | -2.57% | -3.11% | 4.14% | 1.62% | 55.08% | 4.14% | -2.9742247601494132e-05 |
| price_only | fwd_3m_return | mid_cap | 2559 | 12 | -4.71% | -0.82% | -0.41% | 5.02% | 4.40% | 60.16% | 3.75% | 0.01270884127573177 |
| quality_growth_price | fwd_3m_return | mid_cap | 2559 | 12 | -3.44% | 6.13% | 6.96% | 4.81% | 3.12% | 60.55% | 2.64% | 0.021698013626121785 |
| rich_fundamental | fwd_3m_return | mid_cap | 2559 | 12 | -10.23% | -3.98% | -6.46% | 2.39% | 1.26% | 54.10% | 4.12% | -0.017276201374852633 |
| rich_fundamental_price | fwd_3m_return | mid_cap | 2559 | 12 | -11.91% | -3.39% | -4.47% | 3.42% | 1.39% | 54.49% | 3.67% | -0.002519900298243352 |
| financial_price | fwd_6m_return | all | 5080 | 11 | -4.29% | 5.01% | -4.77% | 11.67% | 5.96% | 58.86% | 9.02% | 0.02646071911206556 |
| price_only | fwd_6m_return | all | 5080 | 11 | -2.33% | 6.92% | -5.59% | 13.66% | 9.87% | 63.58% | 9.21% | 0.04450313314758188 |
| quality_growth_price | fwd_6m_return | all | 5080 | 11 | -4.15% | 6.78% | 0.40% | 12.77% | 7.85% | 63.39% | 6.86% | 0.059050306346097786 |
| rich_fundamental | fwd_6m_return | all | 5080 | 11 | -6.04% | 1.24% | -1.47% | 8.41% | 5.25% | 62.01% | 6.53% | 0.018784807089648817 |
| rich_fundamental_price | fwd_6m_return | all | 5080 | 11 | -7.59% | 3.22% | -4.40% | 11.38% | 6.20% | 59.15% | 8.05% | 0.033320165291846754 |
| financial_price | fwd_6m_return | large_cap | 2314 | 11 | -3.33% | 10.16% | -0.69% | 17.16% | 12.11% | 66.95% | 10.79% | 0.0637222452580616 |
| price_only | fwd_6m_return | large_cap | 2314 | 11 | -1.03% | 12.22% | -0.98% | 19.29% | 12.68% | 70.63% | 10.81% | 0.08479866347101159 |
| quality_growth_price | fwd_6m_return | large_cap | 2314 | 11 | -3.67% | 9.47% | 1.97% | 15.03% | 9.44% | 62.85% | 9.35% | 0.05674275074881101 |
| rich_fundamental | fwd_6m_return | large_cap | 2314 | 11 | -7.91% | -0.47% | -1.62% | 9.23% | 6.72% | 62.20% | 8.90% | 0.003303508705026978 |
| rich_fundamental_price | fwd_6m_return | large_cap | 2314 | 11 | -9.00% | 5.18% | -2.12% | 13.56% | 8.84% | 61.12% | 9.21% | 0.04342087488875242 |
| financial_price | fwd_6m_return | mid_cap | 2540 | 11 | -8.87% | -3.54% | -8.58% | 7.08% | 2.83% | 54.92% | 8.20% | -0.011222113412230852 |
| price_only | fwd_6m_return | mid_cap | 2540 | 11 | -6.76% | -2.81% | -9.44% | 7.69% | 3.85% | 56.10% | 9.45% | -0.01757138617724413 |
| quality_growth_price | fwd_6m_return | mid_cap | 2540 | 11 | -5.12% | 5.12% | 1.85% | 10.90% | 7.24% | 64.17% | 5.64% | 0.05260203009383427 |
| rich_fundamental | fwd_6m_return | mid_cap | 2540 | 11 | -8.57% | -0.46% | -2.48% | 6.45% | 3.36% | 58.46% | 6.31% | 0.001320901025659016 |
| rich_fundamental_price | fwd_6m_return | mid_cap | 2540 | 11 | -11.87% | -2.27% | -6.71% | 6.66% | 2.79% | 54.53% | 7.31% | -0.006453169770843009 |

## Portfolio summary

_No rows._

## Interpretation

- This answers whether more engineered fundamentals improve the longer-history Dolt validation.
- If rich feature sets do not beat income+price or price-only, the simple annual model remains preferable.
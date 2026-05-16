# DoltHub Aligned Replication

Aligns the DoltHub fundamentals validation more closely to the Yahoo S&P annual model and compares feature sets by market-cap bucket.

## Performance

| feature_set | forward_window | regression_bucket | observations | test_years | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| financial_only | fwd_12m_return | all | 4605 | 10 | -0.94% | 5.67% | 0.55% | 21.38% | 11.65% | 65.15% | 13.18% | 0.08206588314288246 |
| financial_plus_price | fwd_12m_return | all | 4605 | 10 | 3.63% | 20.89% | 21.17% | 36.27% | 24.89% | 73.72% | 6.29% | 0.2997782624985755 |
| income_only | fwd_12m_return | all | 4605 | 10 | -0.31% | 6.04% | -0.64% | 19.52% | 9.59% | 63.74% | 14.67% | 0.04857563200785214 |
| price_only | fwd_12m_return | all | 4605 | 10 | 4.85% | 22.62% | 22.74% | 42.28% | 30.89% | 78.28% | 6.77% | 0.3550426527419078 |
| financial_only | fwd_12m_return | large_cap | 2101 | 10 | 0.64% | 17.15% | 11.80% | 34.76% | 21.29% | 75.06% | 14.36% | 0.203975105429194 |
| financial_plus_price | fwd_12m_return | large_cap | 2101 | 10 | 5.76% | 26.15% | 24.72% | 46.14% | 31.46% | 76.72% | 9.68% | 0.36464196782950076 |
| income_only | fwd_12m_return | large_cap | 2101 | 10 | 1.29% | 18.76% | 12.11% | 36.59% | 21.44% | 75.77% | 15.03% | 0.21565998678815562 |
| price_only | fwd_12m_return | large_cap | 2101 | 10 | 6.21% | 25.89% | 25.45% | 50.20% | 34.12% | 79.57% | 10.14% | 0.40058411885290973 |
| financial_only | fwd_12m_return | mid_cap | 2302 | 10 | -1.75% | 2.31% | -0.75% | 16.32% | 9.69% | 65.08% | 13.28% | 0.030317708169081137 |
| financial_plus_price | fwd_12m_return | mid_cap | 2302 | 10 | -0.48% | 14.47% | 17.79% | 24.82% | 13.94% | 69.85% | 6.34% | 0.1847951994410308 |
| income_only | fwd_12m_return | mid_cap | 2302 | 10 | -0.61% | 2.64% | -0.82% | 13.99% | 8.10% | 62.26% | 12.84% | 0.01153666071201323 |
| price_only | fwd_12m_return | mid_cap | 2302 | 10 | 1.17% | 16.14% | 18.73% | 29.33% | 19.08% | 71.58% | 6.55% | 0.22783448728106326 |
| financial_only | fwd_3m_return | all | 5122 | 12 | -4.44% | -8.41% | -10.66% | 1.89% | -0.16% | 49.46% | 4.66% | -0.027776347102008915 |
| financial_plus_price | fwd_3m_return | all | 5122 | 12 | -5.31% | 0.05% | -1.33% | 5.63% | 3.52% | 58.44% | 4.29% | 0.013358447801101639 |
| income_only | fwd_3m_return | all | 5122 | 12 | -4.14% | -11.25% | -12.89% | 0.12% | -0.81% | 46.93% | 5.07% | -0.04957199704550599 |
| price_only | fwd_3m_return | all | 5122 | 12 | -2.45% | 4.39% | 2.19% | 6.70% | 5.43% | 63.02% | 3.65% | 0.030525319331526114 |
| financial_only | fwd_3m_return | large_cap | 2336 | 12 | -3.65% | -2.53% | -5.48% | 4.18% | 1.96% | 54.27% | 4.92% | -0.007390670692759914 |
| financial_plus_price | fwd_3m_return | large_cap | 2336 | 12 | -5.22% | 2.58% | 0.55% | 5.88% | 4.47% | 60.04% | 4.27% | 0.01609594321721705 |
| income_only | fwd_3m_return | large_cap | 2336 | 12 | -3.46% | -7.14% | -9.81% | 3.23% | 1.58% | 52.99% | 5.60% | -0.02362009146074045 |
| price_only | fwd_3m_return | large_cap | 2336 | 12 | -1.76% | 6.64% | 2.73% | 9.23% | 7.22% | 67.52% | 4.74% | 0.044951086562166835 |
| financial_only | fwd_3m_return | mid_cap | 2559 | 12 | -5.01% | -6.47% | -7.66% | 1.42% | 0.67% | 51.95% | 4.02% | -0.026001466889725204 |
| financial_plus_price | fwd_3m_return | mid_cap | 2559 | 12 | -7.56% | -2.57% | -3.11% | 4.14% | 1.62% | 55.08% | 4.14% | -2.9742247601494132e-05 |
| income_only | fwd_3m_return | mid_cap | 2559 | 12 | -4.53% | -10.27% | -11.19% | -0.56% | -0.44% | 47.46% | 3.93% | -0.044911689739223735 |
| price_only | fwd_3m_return | mid_cap | 2559 | 12 | -4.71% | -0.82% | -0.41% | 5.02% | 4.40% | 60.16% | 3.75% | 0.01270884127573177 |
| financial_only | fwd_6m_return | all | 5080 | 11 | -4.58% | -5.55% | -10.41% | 5.05% | 2.08% | 54.04% | 8.74% | -0.036930986170321096 |
| financial_plus_price | fwd_6m_return | all | 5080 | 11 | -4.29% | 5.01% | -4.77% | 11.67% | 5.96% | 58.86% | 9.02% | 0.02646071911206556 |
| income_only | fwd_6m_return | all | 5080 | 11 | -4.22% | -8.87% | -12.91% | 2.44% | 0.67% | 51.28% | 9.89% | -0.07444558697431257 |
| price_only | fwd_6m_return | all | 5080 | 11 | -2.33% | 6.92% | -5.59% | 13.66% | 9.87% | 63.58% | 9.21% | 0.04450313314758188 |
| financial_only | fwd_6m_return | large_cap | 2314 | 11 | -4.23% | -1.94% | -8.21% | 7.89% | 4.73% | 57.24% | 9.35% | -0.014639503530399986 |
| financial_plus_price | fwd_6m_return | large_cap | 2314 | 11 | -3.33% | 10.16% | -0.69% | 17.16% | 12.11% | 66.95% | 10.79% | 0.0637222452580616 |
| income_only | fwd_6m_return | large_cap | 2314 | 11 | -4.41% | -8.08% | -12.71% | 6.26% | 2.68% | 55.51% | 11.79% | -0.055309452644955404 |
| price_only | fwd_6m_return | large_cap | 2314 | 11 | -1.03% | 12.22% | -0.98% | 19.29% | 12.68% | 70.63% | 10.81% | 0.08479866347101159 |
| financial_only | fwd_6m_return | mid_cap | 2540 | 11 | -5.70% | -6.51% | -8.29% | 3.96% | 2.57% | 56.50% | 8.22% | -0.04260254598455267 |
| financial_plus_price | fwd_6m_return | mid_cap | 2540 | 11 | -8.87% | -3.54% | -8.58% | 7.08% | 2.83% | 54.92% | 8.20% | -0.011222113412230852 |
| income_only | fwd_6m_return | mid_cap | 2540 | 11 | -4.43% | -7.61% | -9.65% | 3.01% | 2.47% | 55.71% | 8.84% | -0.058309511316700466 |
| price_only | fwd_6m_return | mid_cap | 2540 | 11 | -6.76% | -2.81% | -9.44% | 7.69% | 3.85% | 56.10% | 9.45% | -0.01757138617724413 |

## Prediction-weighted portfolio

| observations | total_return | annual_return | annual_std | sharpe | average_daily_return | average_daily_turnover | max_drawdown | hit_rate | portfolio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2589 | 1090.35% | 27.26% | 24.20% | 1.1266438530964695 | 0.11% | 0.01462461029006655 | -31.95% | 55.50% | dolthub_aligned_largecap_top_quintile_equal |

## Interpretation

- This is the apples-to-apples Dolt check requested after noticing price-only strength.
- Feature engineering is closer to the Yahoo model, and large-cap filtering uses the same current market-cap buckets from the Yahoo S&P run.
- If price-only remains close to financial+price, fundamentals should be viewed as a filter/anchor rather than the sole driver.
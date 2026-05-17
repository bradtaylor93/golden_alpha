# DoltHub Annual + Latest Quarter Model

Keeps annual events as the prediction unit and attaches latest available quarterly/TTM features before each annual trade date.

## Performance

| feature_set | regression_bucket | observations | test_years | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| latest_quarter_only | all | 4605 | 10 | 5.13% | 23.19% | 22.55% | 41.60% | 29.60% | 79.04% | 7.69% | 0.3391478032492476 |
| annual_price | all | 4605 | 10 | 3.60% | 20.89% | 21.15% | 36.16% | 24.83% | 73.51% | 6.71% | 0.2945287792706617 |
| annual_price_latest_quarter | all | 4605 | 10 | 3.55% | 20.97% | 20.98% | 37.40% | 25.91% | 75.14% | 6.96% | 0.3044597251119462 |
| annual_price | large_cap | 2101 | 10 | 5.77% | 26.20% | 24.80% | 46.14% | 31.46% | 76.72% | 10.13% | 0.3601013584496266 |
| latest_quarter_only | large_cap | 2101 | 10 | 6.13% | 26.14% | 24.75% | 48.93% | 33.65% | 79.33% | 11.04% | 0.37894656110942554 |
| annual_price_latest_quarter | large_cap | 2101 | 10 | 5.11% | 25.60% | 24.19% | 46.15% | 31.42% | 76.25% | 9.47% | 0.3668198670711991 |
| latest_quarter_only | mid_cap | 2302 | 10 | 1.72% | 17.35% | 19.42% | 30.91% | 19.22% | 73.32% | 6.65% | 0.24256793445703653 |
| annual_price_latest_quarter | mid_cap | 2302 | 10 | -0.13% | 15.43% | 18.14% | 25.61% | 13.82% | 69.85% | 7.13% | 0.18475559696875676 |
| annual_price | mid_cap | 2302 | 10 | -0.55% | 14.43% | 17.58% | 24.17% | 13.56% | 68.98% | 6.71% | 0.17467645399805506 |

## Interpretation

- This tests the requested combination: annual + latest quarter + price.
- If latest-quarter features do not improve the annual model, quarterly trends are not adding incremental ranking power in this setup.
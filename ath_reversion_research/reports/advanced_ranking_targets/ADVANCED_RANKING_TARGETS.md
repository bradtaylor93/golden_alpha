# Advanced Ranking Targets

Tests risk-adjusted sector-excess targets and grouped LightGBM ranking objectives on the Dolt aligned sample.

## Results

| experiment | bucket | observations | test_years | raw_return_spearman | target_spearman | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_return | all | 4605 | 10 | 27.65% | 27.65% | 38.31% | 26.86% | 74.70% | 2.89% | 0.35417655415497334 |
| sector_excess | all | 4561 | 10 | 24.40% | 15.27% | 33.35% | 22.44% | 74.48% | 3.96% | 0.29396864792749455 |
| risk_adj_sector_excess | all | 4561 | 10 | 19.80% | 16.83% | 30.94% | 19.71% | 74.81% | 7.59% | 0.2335252519062343 |
| lgbm_rank_raw | all | 4605 | 10 | 15.57% | 15.57% | 35.51% | 22.38% | 77.52% | 12.23% | 0.23280939901284864 |
| lgbm_rank_sector_excess | all | 4561 | 10 | 13.01% | 11.26% | 37.02% | 23.12% | 74.92% | 14.78% | 0.22240421445341302 |
| sector_excess | large_cap | 2076 | 10 | 28.76% | 9.53% | 48.23% | 31.80% | 79.09% | 7.20% | 0.4102949273389124 |
| raw_return | large_cap | 2101 | 10 | 28.38% | 28.38% | 49.66% | 35.19% | 77.67% | 7.67% | 0.4198840879011993 |
| risk_adj_sector_excess | large_cap | 2076 | 10 | 22.61% | 12.54% | 42.74% | 27.80% | 79.81% | 9.98% | 0.3275866632929123 |
| lgbm_rank_raw | large_cap | 2101 | 10 | 14.47% | 14.47% | 45.81% | 28.02% | 78.15% | 17.10% | 0.28712503772713677 |
| lgbm_rank_sector_excess | large_cap | 2076 | 10 | 13.61% | 13.60% | 40.34% | 22.30% | 74.52% | 15.02% | 0.2531982162406857 |
| raw_return | mid_cap | 2302 | 10 | 23.37% | 23.37% | 27.66% | 18.82% | 71.15% | 3.64% | 0.24025670571329413 |
| sector_excess | mid_cap | 2284 | 10 | 16.92% | 15.38% | 21.99% | 15.04% | 70.24% | 6.79% | 0.15201639205044049 |
| risk_adj_sector_excess | mid_cap | 2284 | 10 | 16.17% | 16.45% | 21.28% | 13.94% | 72.21% | 8.07% | 0.13205234124343326 |
| lgbm_rank_sector_excess | mid_cap | 2284 | 10 | 7.82% | 5.38% | 21.67% | 15.04% | 66.74% | 10.97% | 0.1069935929077586 |
| lgbm_rank_raw | mid_cap | 2302 | 10 | 5.26% | 5.26% | 21.94% | 12.13% | 67.25% | 11.06% | 0.10883294207520997 |

## Interpretation

- A target is useful only if it improves realized raw 12m return ranking.
- LightGBM ranker is useful only if it beats ridge on raw-return Spearman and top-quintile realized returns.
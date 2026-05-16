# DoltHub Upgrade Ablation

Runs proposed target/valuation/rank upgrades across the longer Dolt sample.

## Performance

| experiment | target_type | regression_bucket | observations | test_years | oos_r2_on_training_target | raw_return_spearman | target_spearman | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spy_excess_raw_return | spy_excess_return | all | 4605 | 10 | 4.39% | 27.88% | 19.22% | 36.62% | 23.95% | 74.27% | 3.30% | 0.3331512103876403 |
| valuation_raw_return | raw_return | all | 4605 | 10 | 6.24% | 27.65% | 27.65% | 38.31% | 26.86% | 74.70% | 2.89% | 0.35417655415497334 |
| valuation_top_quintile_target | top_quintile_target | all | 4605 | 10 | 2.30% | 24.44% | 17.72% | 36.99% | 22.46% | 73.07% | 4.77% | 0.3222399726762156 |
| sector_excess_raw_return | sector_excess_return | all | 4561 | 10 | 3.61% | 24.41% | 15.27% | 33.35% | 22.44% | 74.48% | 3.96% | 0.29396864792749455 |
| valuation_rank_target | rank_target | all | 4605 | 10 | -0.17% | 21.81% | 16.28% | 32.13% | 20.73% | 75.35% | 7.39% | 0.24739028064424323 |
| baseline_raw_return | raw_return | all | 4605 | 10 | 3.63% | 21.17% | 21.17% | 36.27% | 24.89% | 73.72% | 6.29% | 0.2997782624985755 |
| sector_excess_rank_target | sector_excess_rank_target | all | 4561 | 10 | -0.02% | 17.41% | 14.42% | 29.18% | 19.78% | 75.03% | 10.49% | 0.186908814115395 |
| baseline_top_quintile_target | top_quintile_target | all | 4605 | 10 | 0.13% | 16.50% | 11.76% | 32.39% | 19.78% | 70.25% | 8.22% | 0.24170862173081675 |
| baseline_rank_target | rank_target | all | 4605 | 10 | -3.65% | 10.42% | 2.27% | 24.37% | 17.03% | 70.47% | 12.77% | 0.11595398110342944 |
| sector_excess_raw_return | sector_excess_return | large_cap | 2076 | 10 | 3.42% | 28.76% | 9.53% | 48.23% | 31.80% | 79.09% | 7.20% | 0.4102949273389124 |
| valuation_raw_return | raw_return | large_cap | 2101 | 10 | 6.90% | 28.38% | 28.38% | 49.66% | 35.19% | 77.67% | 7.67% | 0.4198840879011993 |
| spy_excess_raw_return | spy_excess_return | large_cap | 2101 | 10 | 5.26% | 28.38% | 16.81% | 47.63% | 33.30% | 76.72% | 7.60% | 0.4002702746890546 |
| baseline_raw_return | raw_return | large_cap | 2101 | 10 | 5.76% | 24.72% | 24.72% | 46.14% | 31.46% | 76.72% | 9.68% | 0.36464196782950076 |
| valuation_top_quintile_target | top_quintile_target | large_cap | 2101 | 10 | 2.67% | 23.00% | 19.74% | 42.63% | 26.75% | 76.48% | 8.75% | 0.33880289277984244 |
| valuation_rank_target | rank_target | large_cap | 2101 | 10 | -1.11% | 16.85% | 16.85% | 38.95% | 25.61% | 77.91% | 15.12% | 0.2383203609132248 |
| baseline_top_quintile_target | top_quintile_target | large_cap | 2101 | 10 | 0.75% | 16.25% | 14.46% | 36.75% | 24.61% | 75.30% | 12.22% | 0.24529520372697555 |
| sector_excess_rank_target | sector_excess_rank_target | large_cap | 2076 | 10 | -1.61% | 12.57% | 14.13% | 37.20% | 25.15% | 78.85% | 17.24% | 0.1996637323797615 |
| baseline_rank_target | rank_target | large_cap | 2101 | 10 | -3.71% | 7.74% | 7.22% | 32.70% | 22.21% | 75.06% | 20.11% | 0.12587610476569555 |
| valuation_raw_return | raw_return | mid_cap | 2302 | 10 | 2.18% | 23.37% | 23.37% | 27.66% | 18.82% | 71.15% | 3.64% | 0.24025670571329413 |
| spy_excess_raw_return | spy_excess_return | mid_cap | 2302 | 10 | -1.19% | 22.39% | 16.84% | 24.22% | 17.55% | 71.80% | 4.21% | 0.20005798870257452 |
| valuation_top_quintile_target | top_quintile_target | mid_cap | 2302 | 10 | -3.42% | 20.52% | 7.89% | 29.22% | 21.50% | 74.40% | 6.13% | 0.23080669387772204 |
| valuation_rank_target | rank_target | mid_cap | 2302 | 10 | -4.02% | 17.86% | 10.44% | 24.18% | 18.56% | 73.97% | 6.64% | 0.17537097416599046 |
| baseline_raw_return | raw_return | mid_cap | 2302 | 10 | -0.48% | 17.79% | 17.79% | 24.82% | 13.94% | 69.85% | 6.34% | 0.1847951994410308 |
| sector_excess_raw_return | sector_excess_return | mid_cap | 2284 | 10 | -0.49% | 16.93% | 15.38% | 21.92% | 15.00% | 70.24% | 6.79% | 0.1512207702924404 |
| sector_excess_rank_target | sector_excess_rank_target | mid_cap | 2284 | 10 | -2.22% | 15.65% | 11.94% | 22.36% | 17.54% | 73.52% | 7.66% | 0.1470025492591588 |
| baseline_top_quintile_target | top_quintile_target | mid_cap | 2302 | 10 | -4.40% | 13.86% | 2.85% | 28.02% | 19.53% | 72.67% | 8.60% | 0.19417447195467735 |
| baseline_rank_target | rank_target | mid_cap | 2302 | 10 | -6.48% | 7.73% | -2.01% | 19.08% | 13.40% | 69.20% | 14.03% | 0.050536284448630814 |

## Interpretation

- This is the long-history counterpart to the Yahoo/current-S&P ablation.
- Upgrades are only useful if they improve realized raw-return ranking, not merely the transformed training target.
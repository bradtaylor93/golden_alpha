# Fundamental Upgrade Ablation

Tests proposed upgrades: valuation-style features, excess-return targets, rank/classification targets, sector/macro features, and combined variants.

## Predictive performance

| experiment | target_type | observations | oos_r2_on_training_target | raw_return_spearman | target_spearman | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sector_excess_raw_return | sector_excess_return | 192 | 25.47% | 49.64% | 43.97% | 86.10% | 63.90% | 89.74% | 7.43% | 0.7867501315653881 |
| spy_excess_raw_return | spy_excess_return | 192 | 29.42% | 47.92% | 47.40% | 86.32% | 62.00% | 92.31% | 7.86% | 0.7845885520620232 |
| valuation_raw_return | raw_return | 192 | 31.07% | 46.05% | 46.05% | 85.94% | 64.88% | 94.87% | 9.70% | 0.7623969437916133 |
| combined_raw_return | raw_return | 192 | 30.63% | 45.81% | 45.81% | 82.33% | 62.00% | 89.74% | 8.43% | 0.7390818138391917 |
| baseline_raw_return | raw_return | 192 | 22.84% | 40.20% | 40.20% | 81.17% | 61.18% | 92.31% | 11.86% | 0.6930716438914587 |
| sector_macro_raw_return | raw_return | 192 | 24.10% | 37.82% | 37.82% | 82.57% | 63.90% | 92.31% | 13.37% | 0.6919804201505646 |
| sector_excess_rank_target | sector_excess_rank_target | 192 | -24.50% | 32.82% | 26.71% | 65.81% | 46.37% | 87.18% | 13.31% | 0.5250390466142972 |
| combined_top_quintile_target | top_quintile_target | 192 | -9.91% | 31.40% | 36.37% | 62.89% | 36.14% | 79.49% | 14.20% | 0.48692613879760904 |
| valuation_top_quintile_target | top_quintile_target | 192 | -9.60% | 29.60% | 35.70% | 65.70% | 38.31% | 87.18% | 16.78% | 0.4892434961120026 |
| combined_rank_target | rank_target | 192 | -34.03% | 29.15% | 25.06% | 63.01% | 46.25% | 87.18% | 13.39% | 0.4961967202509332 |
| valuation_rank_target | rank_target | 192 | -21.33% | 27.11% | 23.12% | 67.91% | 47.11% | 84.62% | 12.72% | 0.5519044877686772 |
| baseline_top_quintile_target | top_quintile_target | 192 | -0.95% | 17.15% | 31.05% | 60.05% | 36.14% | 79.49% | 14.17% | 0.45879431860039366 |
| baseline_rank_target | rank_target | 192 | -21.25% | 16.33% | 13.52% | 51.14% | 39.30% | 79.49% | 17.05% | 0.34095455929634466 |

## Event top-quintile proxy

| experiment | event_top_quintile_mean_return | event_top_quintile_median_return | event_top_quintile_hit_rate | event_count |
| --- | --- | --- | --- | --- |
| spy_excess_raw_return | 86.32% | 62.00% | 92.31% | 39 |
| sector_excess_raw_return | 86.10% | 63.90% | 89.74% | 39 |
| valuation_raw_return | 85.94% | 64.88% | 94.87% | 39 |
| sector_macro_raw_return | 82.57% | 63.90% | 92.31% | 39 |
| combined_raw_return | 82.33% | 62.00% | 89.74% | 39 |
| baseline_raw_return | 81.17% | 61.18% | 92.31% | 39 |
| valuation_rank_target | 67.91% | 47.11% | 84.62% | 39 |
| sector_excess_rank_target | 65.81% | 46.37% | 87.18% | 39 |
| valuation_top_quintile_target | 65.70% | 38.31% | 87.18% | 39 |
| combined_rank_target | 63.01% | 46.25% | 87.18% | 39 |
| combined_top_quintile_target | 62.89% | 36.14% | 79.49% | 39 |
| baseline_top_quintile_target | 60.05% | 36.14% | 79.49% | 39 |
| baseline_rank_target | 51.14% | 39.30% | 79.49% | 39 |

## Interpretation

- This suite evaluates whether changes improve the actual raw-return rank and top-quintile returns.
- Excess-return/rank/classification targets are only useful if raw forward-return portfolio outcomes improve.
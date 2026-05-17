# Peer and cluster valuation feature test

This ablation tests under/over-valuation features relative to sectors, learned peer clusters, and fold-local expected valuation models.

## No-leakage setup

- Clusters are fit inside each walk-forward fold using only train rows whose 12m forward window is complete.
- Sector and cluster valuation medians are computed from train rows only.
- Expected valuation residuals are from ridge models trained only on prior train rows.
- Final return predictions use the same purged annual walk-forward split as the valuation study.

## Feature groups tested

- Sector-relative valuation yields and quality/value/trend interactions.
- Dynamic cluster-relative valuation yields from return/fundamental embeddings.
- Peer-implied fair value upside using cluster median sales, gross-profit, earnings, and FCF yields.
- Expected valuation residuals: actual log valuation minus valuation predicted from growth, quality, risk, and trend.

## Input coverage

| field                    | coverage | unique |
| ------------------------ | -------- | ------ |
| historical_market_cap    | 1.0000   | 5871   |
| gics_sector              | 1.0000   | 11     |
| hist_sales_to_market_cap | 0.9998   | 5870   |
| hist_earnings_yield      | 1.0000   | 5871   |
| hist_fcf_yield           | 0.9906   | 5816   |

## Performance summary

| experiment                          | target_type          | regression_bucket | rows | years | spearman | target_spearman | top_quintile_mean_return | top_quintile_hit_rate | mean_yearly_top_minus_bottom | positive_yearly_spread_rate |
| ----------------------------------- | -------------------- | ----------------- | ---- | ----- | -------- | --------------- | ------------------------ | --------------------- | ---------------------------- | --------------------------- |
| valuation_residuals                 | raw_return           | all               | 4605 | 10    | 0.3090   | 0.3090          | 0.4200                   | 0.7744                | 0.2491                       | 1.0000                      |
| best_known_current_plus_hist        | raw_return           | all               | 4605 | 10    | 0.3087   | 0.3087          | 0.4325                   | 0.7755                | 0.2809                       | 1.0000                      |
| cluster_relative_value              | raw_return           | all               | 4605 | 10    | 0.3072   | 0.3072          | 0.4324                   | 0.7668                | 0.2783                       | 0.9000                      |
| cluster_fair_value_only             | raw_return           | all               | 4605 | 10    | 0.3061   | 0.3061          | 0.4322                   | 0.7690                | 0.2825                       | 1.0000                      |
| sales_residual_selected             | raw_return           | all               | 4605 | 10    | 0.3033   | 0.3033          | 0.4187                   | 0.7625                | 0.2628                       | 1.0000                      |
| sector_relative_value               | raw_return           | all               | 4605 | 10    | 0.3006   | 0.3006          | 0.4280                   | 0.7581                | 0.2681                       | 1.0000                      |
| selected_peer_value                 | raw_return           | all               | 4605 | 10    | 0.2985   | 0.2985          | 0.4149                   | 0.7581                | 0.2540                       | 1.0000                      |
| combined_peer_cluster_value         | raw_return           | all               | 4605 | 10    | 0.2938   | 0.2938          | 0.4021                   | 0.7592                | 0.2479                       | 0.9000                      |
| combined_peer_cluster_sector_excess | sector_excess_return | all               | 4561 | 10    | 0.2680   | 0.1936          | 0.3433                   | 0.7568                | 0.2431                       | 1.0000                      |
| baseline_no_valuation               | raw_return           | all               | 4605 | 10    | 0.2295   | 0.2295          | 0.3693                   | 0.7430                | 0.1581                       | 0.7000                      |
| best_known_current_plus_hist        | raw_return           | large_cap         | 2101 | 10    | 0.2935   | 0.2935          | 0.5013                   | 0.7933                | 0.2993                       | 0.9000                      |
| cluster_fair_value_only             | raw_return           | large_cap         | 2101 | 10    | 0.2875   | 0.2875          | 0.5068                   | 0.7933                | 0.3004                       | 0.9000                      |
| sector_relative_value               | raw_return           | large_cap         | 2101 | 10    | 0.2866   | 0.2866          | 0.5044                   | 0.7886                | 0.2938                       | 0.9000                      |
| sales_residual_selected             | raw_return           | large_cap         | 2101 | 10    | 0.2787   | 0.2787          | 0.4955                   | 0.7838                | 0.2930                       | 0.9000                      |
| cluster_relative_value              | raw_return           | large_cap         | 2101 | 10    | 0.2787   | 0.2787          | 0.4972                   | 0.7862                | 0.2853                       | 0.9000                      |
| valuation_residuals                 | raw_return           | large_cap         | 2101 | 10    | 0.2779   | 0.2779          | 0.4893                   | 0.7886                | 0.2781                       | 0.9000                      |
| selected_peer_value                 | raw_return           | large_cap         | 2101 | 10    | 0.2709   | 0.2709          | 0.4860                   | 0.7696                | 0.2975                       | 0.9000                      |
| combined_peer_cluster_value         | raw_return           | large_cap         | 2101 | 10    | 0.2640   | 0.2640          | 0.4713                   | 0.7791                | 0.2695                       | 0.9000                      |
| combined_peer_cluster_sector_excess | sector_excess_return | large_cap         | 2076 | 10    | 0.2521   | 0.1072          | 0.4351                   | 0.7909                | 0.2628                       | 1.0000                      |
| baseline_no_valuation               | raw_return           | large_cap         | 2101 | 10    | 0.2499   | 0.2499          | 0.4631                   | 0.7672                | 0.2013                       | 0.9000                      |
| valuation_residuals                 | raw_return           | mid_cap           | 2302 | 10    | 0.2783   | 0.2783          | 0.3191                   | 0.7549                | 0.1804                       | 0.9000                      |
| combined_peer_cluster_value         | raw_return           | mid_cap           | 2302 | 10    | 0.2742   | 0.2742          | 0.3292                   | 0.7701                | 0.1593                       | 1.0000                      |
| best_known_current_plus_hist        | raw_return           | mid_cap           | 2302 | 10    | 0.2724   | 0.2724          | 0.3363                   | 0.7505                | 0.1751                       | 1.0000                      |
| cluster_relative_value              | raw_return           | mid_cap           | 2302 | 10    | 0.2722   | 0.2722          | 0.3500                   | 0.7744                | 0.1700                       | 1.0000                      |
| sales_residual_selected             | raw_return           | mid_cap           | 2302 | 10    | 0.2719   | 0.2719          | 0.3253                   | 0.7549                | 0.1682                       | 1.0000                      |
| cluster_fair_value_only             | raw_return           | mid_cap           | 2302 | 10    | 0.2686   | 0.2686          | 0.3324                   | 0.7549                | 0.1703                       | 0.9000                      |
| selected_peer_value                 | raw_return           | mid_cap           | 2302 | 10    | 0.2662   | 0.2662          | 0.3195                   | 0.7505                | 0.1701                       | 1.0000                      |
| sector_relative_value               | raw_return           | mid_cap           | 2302 | 10    | 0.2618   | 0.2618          | 0.3279                   | 0.7527                | 0.1814                       | 1.0000                      |
| combined_peer_cluster_sector_excess | sector_excess_return | mid_cap           | 2284 | 10    | 0.2392   | 0.2029          | 0.2711                   | 0.7527                | 0.1564                       | 0.8000                      |
| baseline_no_valuation               | raw_return           | mid_cap           | 2302 | 10    | 0.1953   | 0.1953          | 0.2582                   | 0.7093                | 0.0741                       | 0.5000                      |

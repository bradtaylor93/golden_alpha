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
| sector_aware_residuals              | raw_return           | all               | 4605 | 10    | 0.3089   | 0.3089          | 0.4282                   | 0.7809                | 0.2569                       | 1.0000                      |
| best_known_current_plus_hist        | raw_return           | all               | 4605 | 10    | 0.3087   | 0.3087          | 0.4325                   | 0.7755                | 0.2809                       | 1.0000                      |
| cluster_relative_value              | raw_return           | all               | 4605 | 10    | 0.3072   | 0.3072          | 0.4324                   | 0.7668                | 0.2783                       | 0.9000                      |
| cluster_fair_value_only             | raw_return           | all               | 4605 | 10    | 0.3061   | 0.3061          | 0.4322                   | 0.7690                | 0.2825                       | 1.0000                      |
| sales_residual_selected             | raw_return           | all               | 4605 | 10    | 0.3033   | 0.3033          | 0.4187                   | 0.7625                | 0.2628                       | 1.0000                      |
| sector_relative_value               | raw_return           | all               | 4605 | 10    | 0.3006   | 0.3006          | 0.4280                   | 0.7581                | 0.2681                       | 1.0000                      |
| self_relative_value                 | raw_return           | all               | 4605 | 10    | 0.2985   | 0.2985          | 0.4436                   | 0.7722                | 0.2619                       | 0.9000                      |
| selected_peer_value                 | raw_return           | all               | 4605 | 10    | 0.2985   | 0.2985          | 0.4149                   | 0.7581                | 0.2540                       | 1.0000                      |
| residual_plus_self_value            | raw_return           | all               | 4605 | 10    | 0.2962   | 0.2962          | 0.4166                   | 0.7777                | 0.2405                       | 0.9000                      |
| clean_improved_value                | raw_return           | all               | 4605 | 10    | 0.2943   | 0.2943          | 0.4303                   | 0.7820                | 0.2462                       | 0.9000                      |
| combined_peer_cluster_value         | raw_return           | all               | 4605 | 10    | 0.2938   | 0.2938          | 0.4021                   | 0.7592                | 0.2479                       | 0.9000                      |
| combined_peer_cluster_sector_excess | sector_excess_return | all               | 4561 | 10    | 0.2680   | 0.1936          | 0.3433                   | 0.7568                | 0.2431                       | 1.0000                      |
| baseline_no_valuation               | raw_return           | all               | 4605 | 10    | 0.2295   | 0.2295          | 0.3693                   | 0.7430                | 0.1581                       | 0.7000                      |
| best_known_current_plus_hist        | raw_return           | large_cap         | 2101 | 10    | 0.2935   | 0.2935          | 0.5013                   | 0.7933                | 0.2993                       | 0.9000                      |
| self_relative_value                 | raw_return           | large_cap         | 2101 | 10    | 0.2887   | 0.2887          | 0.5033                   | 0.7862                | 0.3104                       | 0.9000                      |
| cluster_fair_value_only             | raw_return           | large_cap         | 2101 | 10    | 0.2875   | 0.2875          | 0.5068                   | 0.7933                | 0.3004                       | 0.9000                      |
| sector_relative_value               | raw_return           | large_cap         | 2101 | 10    | 0.2866   | 0.2866          | 0.5044                   | 0.7886                | 0.2938                       | 0.9000                      |
| sector_aware_residuals              | raw_return           | large_cap         | 2101 | 10    | 0.2789   | 0.2789          | 0.5027                   | 0.7957                | 0.2822                       | 0.9000                      |
| sales_residual_selected             | raw_return           | large_cap         | 2101 | 10    | 0.2787   | 0.2787          | 0.4955                   | 0.7838                | 0.2930                       | 0.9000                      |
| cluster_relative_value              | raw_return           | large_cap         | 2101 | 10    | 0.2787   | 0.2787          | 0.4972                   | 0.7862                | 0.2853                       | 0.9000                      |
| residual_plus_self_value            | raw_return           | large_cap         | 2101 | 10    | 0.2779   | 0.2779          | 0.4897                   | 0.8005                | 0.2960                       | 0.9000                      |
| valuation_residuals                 | raw_return           | large_cap         | 2101 | 10    | 0.2779   | 0.2779          | 0.4893                   | 0.7886                | 0.2781                       | 0.9000                      |
| selected_peer_value                 | raw_return           | large_cap         | 2101 | 10    | 0.2709   | 0.2709          | 0.4860                   | 0.7696                | 0.2975                       | 0.9000                      |
| clean_improved_value                | raw_return           | large_cap         | 2101 | 10    | 0.2674   | 0.2674          | 0.4936                   | 0.7981                | 0.2754                       | 0.9000                      |
| combined_peer_cluster_value         | raw_return           | large_cap         | 2101 | 10    | 0.2640   | 0.2640          | 0.4713                   | 0.7791                | 0.2695                       | 0.9000                      |
| combined_peer_cluster_sector_excess | sector_excess_return | large_cap         | 2076 | 10    | 0.2521   | 0.1072          | 0.4351                   | 0.7909                | 0.2628                       | 1.0000                      |
| baseline_no_valuation               | raw_return           | large_cap         | 2101 | 10    | 0.2499   | 0.2499          | 0.4631                   | 0.7672                | 0.2013                       | 0.9000                      |
| sector_aware_residuals              | raw_return           | mid_cap           | 2302 | 10    | 0.2795   | 0.2795          | 0.3349                   | 0.7679                | 0.1752                       | 1.0000                      |
| valuation_residuals                 | raw_return           | mid_cap           | 2302 | 10    | 0.2783   | 0.2783          | 0.3191                   | 0.7549                | 0.1804                       | 0.9000                      |
| combined_peer_cluster_value         | raw_return           | mid_cap           | 2302 | 10    | 0.2742   | 0.2742          | 0.3292                   | 0.7701                | 0.1593                       | 1.0000                      |
| best_known_current_plus_hist        | raw_return           | mid_cap           | 2302 | 10    | 0.2724   | 0.2724          | 0.3363                   | 0.7505                | 0.1751                       | 1.0000                      |
| cluster_relative_value              | raw_return           | mid_cap           | 2302 | 10    | 0.2722   | 0.2722          | 0.3500                   | 0.7744                | 0.1700                       | 1.0000                      |
| sales_residual_selected             | raw_return           | mid_cap           | 2302 | 10    | 0.2719   | 0.2719          | 0.3253                   | 0.7549                | 0.1682                       | 1.0000                      |
| cluster_fair_value_only             | raw_return           | mid_cap           | 2302 | 10    | 0.2686   | 0.2686          | 0.3324                   | 0.7549                | 0.1703                       | 0.9000                      |
| selected_peer_value                 | raw_return           | mid_cap           | 2302 | 10    | 0.2662   | 0.2662          | 0.3195                   | 0.7505                | 0.1701                       | 1.0000                      |
| residual_plus_self_value            | raw_return           | mid_cap           | 2302 | 10    | 0.2620   | 0.2620          | 0.3216                   | 0.7636                | 0.1570                       | 0.9000                      |
| sector_relative_value               | raw_return           | mid_cap           | 2302 | 10    | 0.2618   | 0.2618          | 0.3279                   | 0.7527                | 0.1814                       | 1.0000                      |
| clean_improved_value                | raw_return           | mid_cap           | 2302 | 10    | 0.2615   | 0.2615          | 0.3300                   | 0.7657                | 0.1419                       | 0.9000                      |
| self_relative_value                 | raw_return           | mid_cap           | 2302 | 10    | 0.2559   | 0.2559          | 0.3031                   | 0.7397                | 0.1603                       | 0.9000                      |
| combined_peer_cluster_sector_excess | sector_excess_return | mid_cap           | 2284 | 10    | 0.2392   | 0.2029          | 0.2711                   | 0.7527                | 0.1564                       | 0.8000                      |
| baseline_no_valuation               | raw_return           | mid_cap           | 2302 | 10    | 0.1953   | 0.1953          | 0.2582                   | 0.7093                | 0.0741                       | 0.5000                      |

## Top-quintile annual portfolio Sharpe

Sharpe here is the mean/std of yearly top-quintile 12-month forward returns. It is a coarse annual portfolio proxy, not a daily marked-to-market live portfolio Sharpe.

| experiment                          | regression_bucket | years | annual_topq_mean | annual_topq_std | annual_topq_sharpe | min_year_return | min_return_year |
| ----------------------------------- | ----------------- | ----- | ---------------- | --------------- | ------------------ | --------------- | --------------- |
| combined_peer_cluster_sector_excess | all               | 10    | 0.3373           | 0.2735          | 1.2333             | 0.0129          | 2022            |
| sector_aware_residuals              | all               | 10    | 0.3396           | 0.2770          | 1.2260             | 0.0160          | 2022            |
| combined_peer_cluster_value         | all               | 10    | 0.3324           | 0.2738          | 1.2143             | 0.0134          | 2022            |
| valuation_residuals                 | all               | 10    | 0.3343           | 0.2756          | 1.2130             | 0.0240          | 2022            |
| selected_peer_value                 | all               | 10    | 0.3307           | 0.2789          | 1.1860             | 0.0162          | 2022            |
| clean_improved_value                | all               | 10    | 0.3348           | 0.2888          | 1.1593             | 0.0001          | 2022            |
| sales_residual_selected             | all               | 10    | 0.3360           | 0.2938          | 1.1440             | 0.0166          | 2022            |
| cluster_relative_value              | all               | 10    | 0.3527           | 0.3104          | 1.1364             | 0.0115          | 2022            |
| best_known_current_plus_hist        | all               | 10    | 0.3534           | 0.3114          | 1.1348             | 0.0216          | 2022            |
| residual_plus_self_value            | all               | 10    | 0.3354           | 0.2960          | 1.1331             | -0.0053         | 2022            |
| baseline_no_valuation               | all               | 10    | 0.2833           | 0.2542          | 1.1145             | -0.0156         | 2022            |
| sector_relative_value               | all               | 10    | 0.3464           | 0.3113          | 1.1128             | 0.0113          | 2022            |
| cluster_fair_value_only             | all               | 10    | 0.3514           | 0.3175          | 1.1065             | 0.0153          | 2022            |
| self_relative_value                 | all               | 10    | 0.3417           | 0.3127          | 1.0929             | 0.0038          | 2022            |
| best_known_current_plus_hist        | large_cap         | 10    | 0.4227           | 0.3091          | 1.3673             | -0.0244         | 2022            |
| cluster_relative_value              | large_cap         | 10    | 0.4345           | 0.3237          | 1.3426             | -0.0108         | 2022            |
| combined_peer_cluster_sector_excess | large_cap         | 10    | 0.4039           | 0.3090          | 1.3073             | -0.0155         | 2022            |
| self_relative_value                 | large_cap         | 10    | 0.4393           | 0.3390          | 1.2957             | -0.0237         | 2022            |
| sales_residual_selected             | large_cap         | 10    | 0.4214           | 0.3266          | 1.2901             | -0.0223         | 2022            |
| sector_aware_residuals              | large_cap         | 10    | 0.4202           | 0.3303          | 1.2721             | -0.0212         | 2022            |
| combined_peer_cluster_value         | large_cap         | 10    | 0.4248           | 0.3355          | 1.2662             | -0.0432         | 2022            |
| cluster_fair_value_only             | large_cap         | 10    | 0.4372           | 0.3473          | 1.2589             | -0.0089         | 2022            |
| valuation_residuals                 | large_cap         | 10    | 0.4140           | 0.3290          | 1.2584             | -0.0263         | 2022            |
| sector_relative_value               | large_cap         | 10    | 0.4193           | 0.3339          | 1.2558             | -0.0113         | 2022            |
| selected_peer_value                 | large_cap         | 10    | 0.4182           | 0.3344          | 1.2505             | -0.0223         | 2022            |
| residual_plus_self_value            | large_cap         | 10    | 0.4210           | 0.3472          | 1.2125             | -0.0376         | 2022            |
| baseline_no_valuation               | large_cap         | 10    | 0.3642           | 0.3009          | 1.2103             | -0.0742         | 2022            |
| clean_improved_value                | large_cap         | 10    | 0.4230           | 0.3504          | 1.2072             | -0.0260         | 2022            |
| combined_peer_cluster_value         | mid_cap           | 10    | 0.2359           | 0.2263          | 1.0425             | 0.0253          | 2019            |
| combined_peer_cluster_sector_excess | mid_cap           | 10    | 0.2523           | 0.2446          | 1.0312             | -0.0019         | 2019            |
| sector_aware_residuals              | mid_cap           | 10    | 0.2485           | 0.2519          | 0.9865             | 0.0385          | 2019            |
| clean_improved_value                | mid_cap           | 10    | 0.2367           | 0.2406          | 0.9837             | 0.0074          | 2019            |
| selected_peer_value                 | mid_cap           | 10    | 0.2481           | 0.2532          | 0.9798             | 0.0135          | 2019            |
| residual_plus_self_value            | mid_cap           | 10    | 0.2415           | 0.2549          | 0.9475             | 0.0060          | 2019            |
| valuation_residuals                 | mid_cap           | 10    | 0.2521           | 0.2695          | 0.9354             | 0.0086          | 2019            |
| cluster_relative_value              | mid_cap           | 10    | 0.2525           | 0.2729          | 0.9255             | 0.0324          | 2019            |
| sales_residual_selected             | mid_cap           | 10    | 0.2470           | 0.2693          | 0.9174             | -0.0048         | 2019            |
| sector_relative_value               | mid_cap           | 10    | 0.2582           | 0.2845          | 0.9075             | 0.0421          | 2019            |
| cluster_fair_value_only             | mid_cap           | 10    | 0.2533           | 0.2833          | 0.8943             | 0.0300          | 2019            |
| best_known_current_plus_hist        | mid_cap           | 10    | 0.2536           | 0.2852          | 0.8892             | 0.0362          | 2019            |
| self_relative_value                 | mid_cap           | 10    | 0.2491           | 0.2894          | 0.8610             | 0.0022          | 2019            |
| baseline_no_valuation               | mid_cap           | 10    | 0.1925           | 0.2377          | 0.8097             | -0.0055         | 2022            |

## Interpretation

- all: best raw-return ranker is `valuation_residuals` with Spearman 0.3090, delta +0.0003 versus current+historical valuation.
- large_cap: best raw-return ranker is `best_known_current_plus_hist` with Spearman 0.2935, delta +0.0000 versus current+historical valuation.
- mid_cap: best raw-return ranker is `sector_aware_residuals` with Spearman 0.2795, delta +0.0071 versus current+historical valuation.
- Broad combined peer/cluster feature sets underperformed, so the useful signal is selective: expected valuation residuals and some cluster/self-relative diagnostics, not every peer feature at once.
- Sector-aware residuals were the best mid-cap variant, suggesting valuation expectations should account for industry context when modeling smaller names.
- all: best annual top-quintile Sharpe is `combined_peer_cluster_sector_excess` at 1.23, delta +0.10 versus current+historical valuation.
- large_cap: best annual top-quintile Sharpe is `best_known_current_plus_hist` at 1.37, delta +0.00 versus current+historical valuation.
- mid_cap: best annual top-quintile Sharpe is `combined_peer_cluster_value` at 1.04, delta +0.15 versus current+historical valuation.

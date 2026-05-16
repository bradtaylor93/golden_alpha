# Yahoo vs Dolt Matched Model Comparison

Runs models on exact matched Yahoo/Dolt rows to isolate data definitions from sample-period effects.

## Dataset

- Matched rows with 12m target: 982.
- Matched symbols: 494.
- Trade years: [2023, 2024, 2025].

## Model comparison

| model | bucket | observations | test_years | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dolt_price_only | all | 414 | 1 | 0.90% | 35.12% | 25.92% | 46.40% | 31.77% | 77.11% | 2.93% | 0.4346884664117962 |
| yahoo_price_only | all | 414 | 1 | 0.90% | 35.12% | 25.92% | 46.40% | 31.77% | 77.11% | 2.93% | 0.4346884664117962 |
| dolt_financial_plus_price | all | 414 | 1 | -0.54% | 31.51% | 21.62% | 43.20% | 31.83% | 78.31% | 7.00% | 0.36197350664310257 |
| yahoo_financial_plus_price | all | 414 | 1 | -0.94% | 30.34% | 18.04% | 46.31% | 31.77% | 78.31% | 7.93% | 0.38376469909517114 |
| dolt_financial_only | all | 414 | 1 | -1.49% | 11.47% | 8.22% | 29.25% | 16.15% | 66.27% | 11.48% | 0.17766104857699866 |
| yahoo_financial_only | all | 414 | 1 | -1.96% | 9.09% | 4.20% | 26.80% | 15.73% | 66.27% | 13.68% | 0.13120359608804327 |
| dolt_price_only | large_cap | 189 | 1 | 19.08% | 50.87% | 39.54% | 73.38% | 55.20% | 86.84% | 3.96% | 0.694174482177028 |
| yahoo_price_only | large_cap | 189 | 1 | 19.08% | 50.87% | 39.54% | 73.38% | 55.20% | 86.84% | 3.96% | 0.694174482177028 |
| dolt_financial_plus_price | large_cap | 189 | 1 | 21.70% | 50.76% | 37.90% | 82.54% | 61.59% | 92.11% | 12.92% | 0.6961942698820823 |
| yahoo_financial_plus_price | large_cap | 189 | 1 | 21.07% | 50.84% | 37.74% | 79.56% | 61.16% | 86.84% | 11.95% | 0.6761483004278191 |
| yahoo_financial_only | large_cap | 189 | 1 | 5.14% | 23.14% | 18.32% | 58.89% | 39.20% | 76.32% | 15.13% | 0.43758266255624956 |
| dolt_financial_only | large_cap | 189 | 1 | 5.14% | 22.91% | 17.77% | 51.18% | 39.20% | 76.32% | 19.89% | 0.3129215720220917 |
| dolt_financial_plus_price | mid_cap | 209 | 1 | -26.89% | -11.20% | -4.68% | 3.90% | -0.22% | 50.00% | 28.68% | -0.2477910331043386 |
| dolt_price_only | mid_cap | 209 | 1 | -24.52% | -3.41% | -8.74% | 8.55% | -0.33% | 50.00% | 7.42% | 0.011239118689596511 |
| yahoo_price_only | mid_cap | 209 | 1 | -24.52% | -3.41% | -8.74% | 8.55% | -0.33% | 50.00% | 7.42% | 0.011239118689596511 |
| dolt_financial_only | mid_cap | 209 | 1 | -22.79% | -16.78% | -10.48% | 5.58% | -0.22% | 50.00% | 20.93% | -0.15352234711863633 |
| yahoo_financial_plus_price | mid_cap | 209 | 1 | -25.33% | -14.44% | -11.73% | 2.24% | -0.43% | 50.00% | 27.60% | -0.25360871722509937 |
| yahoo_financial_only | mid_cap | 209 | 1 | -22.08% | -21.13% | -19.69% | 2.91% | -0.66% | 47.62% | 22.04% | -0.19125269725192698 |

## Interpretation

- If Yahoo and Dolt financial+price behave similarly on matched rows, the longer-history difference is time/sample-period driven.
- If Yahoo beats Dolt on matched rows, feature definitions are still materially different.
- If price-only remains competitive on matched rows, price context is a major driver even in the recent Yahoo period.
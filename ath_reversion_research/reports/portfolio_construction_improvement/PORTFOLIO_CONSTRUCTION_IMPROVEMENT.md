# Portfolio construction improvement test

This study keeps the no-leakage annual model predictions fixed and tests weighting/portfolio construction variants on yearly top-quintile forward returns.

## Variants

- Equal-weight top 20% baseline.
- Rank-confidence weighting.
- Rank-confidence divided by trailing realized volatility.
- Volatility-adjusted confidence weights with 5% name cap and 25% sector cap.
- Bucket sleeves: large-cap prior-best model plus mid-cap sector-aware residual model.
- Dynamic sleeve allocation based only on trailing realized sleeve Sharpe.

Row-level selected predictions are skipped by default to avoid large generated files. Set `SAVE_ROW_LEVEL_PORTFOLIO_CONSTRUCTION=1` to export them locally.

## Summary

| portfolio                        | years | mean_return | std_return | sharpe | min_return | positive_year_rate | avg_holdings | avg_sector_count | avg_large_weight | avg_mid_weight |
| -------------------------------- | ----- | ----------- | ---------- | ------ | ---------- | ------------------ | ------------ | ---------------- | ---------------- | -------------- |
| bucket_equal_sleeves             | 10    | 0.3356      | 0.2690     | 1.2473 | 0.0188     | 1.0000             | 89.0000      | 10.5000          | 0.5000           | 0.5000         |
| all_residual_equal_top20         | 10    | 0.3395      | 0.2771     | 1.2253 | 0.0160     | 1.0000             | 92.6000      | 10.5000          | 0.5200           | 0.4451         |
| all_residual_vol_conf_sector_cap | 10    | 0.3189      | 0.2696     | 1.1828 | 0.0075     | 1.0000             | 92.6000      | 10.5000          | 0.5182           | 0.4475         |
| bucket_vol_conf_sector_cap       | 10    | 0.3060      | 0.2624     | 1.1660 | -0.0021    | 0.9000             | 89.0000      | 10.5000          | 0.5000           | 0.5000         |
| bucket_dynamic_sharpe_sleeves    | 10    | 0.3038      | 0.2625     | 1.1573 | -0.0101    | 0.9000             | 89.0000      | 10.5000          | 0.5266           | 0.4734         |
| all_vol_conf_top20               | 10    | 0.3302      | 0.2858     | 1.1556 | 0.0054     | 1.0000             | 92.6000      | 10.5000          | 0.5391           | 0.4311         |
| all_equal_top20                  | 10    | 0.3533      | 0.3115     | 1.1343 | 0.0216     | 1.0000             | 92.6000      | 10.5000          | 0.5232           | 0.4399         |
| all_vol_conf_sector_cap          | 10    | 0.3269      | 0.2951     | 1.1079 | 0.0003     | 1.0000             | 92.6000      | 10.5000          | 0.5260           | 0.4429         |
| all_confidence_top20             | 10    | 0.3902      | 0.3641     | 1.0717 | -0.0288    | 0.9000             | 92.6000      | 10.5000          | 0.5387           | 0.4233         |

## Annual returns

| portfolio                        | test_year | return  | holdings | max_weight | sector_count | large_weight | mid_weight |
| -------------------------------- | --------- | ------- | -------- | ---------- | ------------ | ------------ | ---------- |
| all_confidence_top20             | 2016      | 0.3432  | 88       | 0.0226     | 11           | 0.4583       | 0.4870     |
| all_confidence_top20             | 2017      | 0.2021  | 90       | 0.0222     | 11           | 0.5406       | 0.4302     |
| all_confidence_top20             | 2018      | 0.3208  | 92       | 0.0217     | 11           | 0.5532       | 0.4104     |
| all_confidence_top20             | 2019      | 0.0939  | 93       | 0.0213     | 9            | 0.5859       | 0.3912     |
| all_confidence_top20             | 2020      | 1.1360  | 95       | 0.0210     | 11           | 0.5895       | 0.3964     |
| all_confidence_top20             | 2021      | 0.0892  | 94       | 0.0211     | 11           | 0.4842       | 0.4581     |
| all_confidence_top20             | 2022      | -0.0288 | 96       | 0.0207     | 9            | 0.5468       | 0.3805     |
| all_confidence_top20             | 2023      | 0.8015  | 97       | 0.0204     | 11           | 0.5491       | 0.4191     |
| all_confidence_top20             | 2024      | 0.3020  | 98       | 0.0202     | 11           | 0.4873       | 0.4921     |
| all_confidence_top20             | 2025      | 0.6422  | 83       | 0.0239     | 10           | 0.5916       | 0.3677     |
| all_equal_top20                  | 2016      | 0.3492  | 88       | 0.0114     | 11           | 0.4432       | 0.5227     |
| all_equal_top20                  | 2017      | 0.2265  | 90       | 0.0111     | 11           | 0.5444       | 0.4222     |
| all_equal_top20                  | 2018      | 0.2892  | 92       | 0.0109     | 11           | 0.5543       | 0.4130     |
| all_equal_top20                  | 2019      | 0.0978  | 93       | 0.0108     | 9            | 0.5806       | 0.3871     |
| all_equal_top20                  | 2020      | 1.0718  | 95       | 0.0105     | 11           | 0.5474       | 0.4316     |
| all_equal_top20                  | 2021      | 0.1229  | 94       | 0.0106     | 11           | 0.4894       | 0.4681     |
| all_equal_top20                  | 2022      | 0.0216  | 96       | 0.0104     | 9            | 0.5104       | 0.4375     |
| all_equal_top20                  | 2023      | 0.6116  | 97       | 0.0103     | 11           | 0.5567       | 0.4124     |
| all_equal_top20                  | 2024      | 0.2278  | 98       | 0.0102     | 11           | 0.4388       | 0.5306     |
| all_equal_top20                  | 2025      | 0.5145  | 83       | 0.0120     | 10           | 0.5663       | 0.3735     |
| all_residual_equal_top20         | 2016      | 0.3451  | 88       | 0.0114     | 11           | 0.4659       | 0.5000     |
| all_residual_equal_top20         | 2017      | 0.1994  | 90       | 0.0111     | 11           | 0.5222       | 0.4444     |
| all_residual_equal_top20         | 2018      | 0.2727  | 92       | 0.0109     | 11           | 0.5326       | 0.4457     |
| all_residual_equal_top20         | 2019      | 0.0998  | 93       | 0.0108     | 10           | 0.5914       | 0.3763     |
| all_residual_equal_top20         | 2020      | 0.9447  | 95       | 0.0105     | 11           | 0.5474       | 0.4316     |
| all_residual_equal_top20         | 2021      | 0.1835  | 94       | 0.0106     | 11           | 0.4894       | 0.4681     |
| all_residual_equal_top20         | 2022      | 0.0160  | 96       | 0.0104     | 9            | 0.5000       | 0.4479     |
| all_residual_equal_top20         | 2023      | 0.6097  | 97       | 0.0103     | 11           | 0.5464       | 0.4227     |
| all_residual_equal_top20         | 2024      | 0.2205  | 98       | 0.0102     | 11           | 0.4388       | 0.5408     |
| all_residual_equal_top20         | 2025      | 0.5035  | 83       | 0.0120     | 9            | 0.5663       | 0.3735     |
| all_residual_vol_conf_sector_cap | 2016      | 0.2868  | 88       | 0.0434     | 11           | 0.4858       | 0.4795     |
| all_residual_vol_conf_sector_cap | 2017      | 0.1621  | 90       | 0.0309     | 11           | 0.5340       | 0.4367     |
| all_residual_vol_conf_sector_cap | 2018      | 0.2701  | 92       | 0.0293     | 11           | 0.5451       | 0.4295     |
| all_residual_vol_conf_sector_cap | 2019      | 0.0668  | 93       | 0.0314     | 10           | 0.5111       | 0.4403     |
| all_residual_vol_conf_sector_cap | 2020      | 0.8522  | 95       | 0.0308     | 11           | 0.6078       | 0.3826     |
| all_residual_vol_conf_sector_cap | 2021      | 0.1139  | 94       | 0.0302     | 11           | 0.4706       | 0.4760     |
| all_residual_vol_conf_sector_cap | 2022      | 0.0075  | 96       | 0.0224     | 9            | 0.5217       | 0.4261     |
| all_residual_vol_conf_sector_cap | 2023      | 0.5725  | 97       | 0.0289     | 11           | 0.5376       | 0.4357     |
| all_residual_vol_conf_sector_cap | 2024      | 0.2688  | 98       | 0.0222     | 11           | 0.4232       | 0.5620     |
| all_residual_vol_conf_sector_cap | 2025      | 0.5882  | 83       | 0.0266     | 9            | 0.5451       | 0.4061     |
| all_vol_conf_sector_cap          | 2016      | 0.2922  | 88       | 0.0417     | 11           | 0.4765       | 0.4892     |
| all_vol_conf_sector_cap          | 2017      | 0.1897  | 90       | 0.0304     | 11           | 0.5549       | 0.4120     |
| all_vol_conf_sector_cap          | 2018      | 0.2795  | 92       | 0.0301     | 11           | 0.5432       | 0.4284     |
| all_vol_conf_sector_cap          | 2019      | 0.0663  | 93       | 0.0359     | 9            | 0.5386       | 0.4343     |
| all_vol_conf_sector_cap          | 2020      | 0.9621  | 95       | 0.0309     | 11           | 0.6112       | 0.3779     |
| all_vol_conf_sector_cap          | 2021      | 0.0995  | 94       | 0.0330     | 11           | 0.4901       | 0.4695     |
| all_vol_conf_sector_cap          | 2022      | 0.0003  | 96       | 0.0235     | 9            | 0.5134       | 0.4334     |
| all_vol_conf_sector_cap          | 2023      | 0.5720  | 97       | 0.0294     | 11           | 0.5402       | 0.4366     |
| all_vol_conf_sector_cap          | 2024      | 0.2307  | 98       | 0.0247     | 11           | 0.4332       | 0.5493     |
| all_vol_conf_sector_cap          | 2025      | 0.5767  | 83       | 0.0318     | 10           | 0.5590       | 0.3988     |
| all_vol_conf_top20               | 2016      | 0.2982  | 88       | 0.0415     | 11           | 0.4809       | 0.4860     |
| all_vol_conf_top20               | 2017      | 0.1897  | 90       | 0.0304     | 11           | 0.5549       | 0.4120     |
| all_vol_conf_top20               | 2018      | 0.2871  | 92       | 0.0278     | 11           | 0.5621       | 0.4122     |
| all_vol_conf_top20               | 2019      | 0.0984  | 93       | 0.0265     | 9            | 0.5818       | 0.3982     |
| all_vol_conf_top20               | 2020      | 0.9348  | 95       | 0.0380     | 11           | 0.6266       | 0.3634     |
| all_vol_conf_top20               | 2021      | 0.0951  | 94       | 0.0320     | 11           | 0.4912       | 0.4662     |
| all_vol_conf_top20               | 2022      | 0.0054  | 96       | 0.0238     | 9            | 0.5300       | 0.4190     |
| all_vol_conf_top20               | 2023      | 0.5720  | 97       | 0.0294     | 11           | 0.5402       | 0.4366     |
| all_vol_conf_top20               | 2024      | 0.2364  | 98       | 0.0238     | 11           | 0.4451       | 0.5376     |
| all_vol_conf_top20               | 2025      | 0.5852  | 83       | 0.0297     | 10           | 0.5786       | 0.3803     |
| bucket_dynamic_sharpe_sleeves    | 2016      | 0.2654  | 85       | 0.0250     | 11           | 0.5000       | 0.5000     |
| bucket_dynamic_sharpe_sleeves    | 2017      | 0.1971  | 86       | 0.0250     | 11           | 0.5000       | 0.5000     |
| bucket_dynamic_sharpe_sleeves    | 2018      | 0.1956  | 88       | 0.0350     | 11           | 0.7000       | 0.3000     |
| bucket_dynamic_sharpe_sleeves    | 2019      | 0.0787  | 90       | 0.0281     | 11           | 0.5625       | 0.4375     |
| bucket_dynamic_sharpe_sleeves    | 2020      | 0.8401  | 91       | 0.0275     | 11           | 0.5493       | 0.4507     |
| bucket_dynamic_sharpe_sleeves    | 2021      | 0.1409  | 91       | 0.0256     | 10           | 0.5115       | 0.4885     |
| bucket_dynamic_sharpe_sleeves    | 2022      | -0.0101 | 92       | 0.0290     | 9            | 0.5810       | 0.4190     |
| bucket_dynamic_sharpe_sleeves    | 2023      | 0.5364  | 93       | 0.0259     | 10           | 0.5186       | 0.4814     |
| bucket_dynamic_sharpe_sleeves    | 2024      | 0.2225  | 94       | 0.0229     | 11           | 0.4471       | 0.5529     |
| bucket_dynamic_sharpe_sleeves    | 2025      | 0.5713  | 80       | 0.0244     | 10           | 0.3957       | 0.6043     |
| bucket_equal_sleeves             | 2016      | 0.3183  | 85       | 0.0122     | 11           | 0.5000       | 0.5000     |
| bucket_equal_sleeves             | 2017      | 0.2038  | 86       | 0.0122     | 11           | 0.5000       | 0.5000     |
| bucket_equal_sleeves             | 2018      | 0.2119  | 88       | 0.0119     | 11           | 0.5000       | 0.5000     |
| bucket_equal_sleeves             | 2019      | 0.1017  | 90       | 0.0116     | 11           | 0.5000       | 0.5000     |
| bucket_equal_sleeves             | 2020      | 0.9134  | 91       | 0.0116     | 11           | 0.5000       | 0.5000     |
| bucket_equal_sleeves             | 2021      | 0.2077  | 91       | 0.0116     | 10           | 0.5000       | 0.5000     |
| bucket_equal_sleeves             | 2022      | 0.0188  | 92       | 0.0114     | 9            | 0.5000       | 0.5000     |
| bucket_equal_sleeves             | 2023      | 0.5879  | 93       | 0.0111     | 10           | 0.5000       | 0.5000     |
| bucket_equal_sleeves             | 2024      | 0.2509  | 94       | 0.0111     | 11           | 0.5000       | 0.5000     |
| bucket_equal_sleeves             | 2025      | 0.5415  | 80       | 0.0132     | 10           | 0.5000       | 0.5000     |
| bucket_vol_conf_sector_cap       | 2016      | 0.2654  | 85       | 0.0250     | 11           | 0.5000       | 0.5000     |
| bucket_vol_conf_sector_cap       | 2017      | 0.1971  | 86       | 0.0250     | 11           | 0.5000       | 0.5000     |
| bucket_vol_conf_sector_cap       | 2018      | 0.1937  | 88       | 0.0250     | 11           | 0.5000       | 0.5000     |
| bucket_vol_conf_sector_cap       | 2019      | 0.0759  | 90       | 0.0250     | 11           | 0.5000       | 0.5000     |
| bucket_vol_conf_sector_cap       | 2020      | 0.8352  | 91       | 0.0250     | 11           | 0.5000       | 0.5000     |
| bucket_vol_conf_sector_cap       | 2021      | 0.1387  | 91       | 0.0250     | 10           | 0.5000       | 0.5000     |
| bucket_vol_conf_sector_cap       | 2022      | -0.0021 | 92       | 0.0250     | 9            | 0.5000       | 0.5000     |
| bucket_vol_conf_sector_cap       | 2023      | 0.5270  | 93       | 0.0250     | 10           | 0.5000       | 0.5000     |
| bucket_vol_conf_sector_cap       | 2024      | 0.2337  | 94       | 0.0250     | 11           | 0.5000       | 0.5000     |
| bucket_vol_conf_sector_cap       | 2025      | 0.5955  | 80       | 0.0250     | 10           | 0.5000       | 0.5000     |

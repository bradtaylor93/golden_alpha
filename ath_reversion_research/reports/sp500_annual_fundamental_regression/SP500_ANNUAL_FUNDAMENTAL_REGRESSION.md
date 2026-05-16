# S&P 500 Annual Fundamentals Regression

This reruns the annual financial-ratio model on current S&P 500 constituents to increase cross-sectional sample size.

## Leakage controls

- Annual financials are assumed available only 90 calendar days after fiscal year end.
- Forward returns start from first trading day on or after that availability date.
- Training rows are purged unless their forward-return end date is before the test year starts.
- Current S&P 500 membership is not point-in-time, so survivorship bias remains.

## Sample size

- Current S&P 500 symbols requested: 503.
- Events: 1507 versus 531 in the prior expanded-stock annual study.
- Prediction rows: 3795 versus 963 prior.
- Symbols with predictions: 503.

## Predictive power

| forward_window | regression_bucket | observations | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fwd_12m_return | all | 421 | 0.98% | 10.48% | 13.36% | 43.00% | 10.17% | 63.53% | 9.23% | 33.77% |
| fwd_12m_return | large_cap | 193 | 4.71% | 29.31% | 26.54% | 90.69% | 31.95% | 79.49% | 10.67% | 80.02% |
| fwd_12m_return | mid_cap | 212 | -28.89% | -12.44% | -11.29% | 3.21% | -0.45% | 48.84% | 22.31% | -19.10% |
| fwd_3m_return | all | 1019 | -15.84% | -7.66% | -11.24% | 0.52% | 0.57% | 51.96% | 4.57% | -4.05% |
| fwd_3m_return | large_cap | 465 | -7.30% | 3.04% | -0.37% | 6.25% | 2.91% | 58.06% | 4.39% | 1.86% |
| fwd_3m_return | low_cap | 21 | -126.52% | -3.43% | -15.06% | 4.07% | -1.45% | 40.00% | 6.94% | -2.88% |
| fwd_3m_return | mid_cap | 512 | -17.42% | -9.31% | -11.93% | -2.15% | -0.75% | 45.63% | 1.01% | -3.16% |
| fwd_6m_return | all | 476 | -0.63% | 3.79% | -1.35% | 15.56% | 6.21% | 63.54% | 15.07% | 0.49% |
| fwd_6m_return | large_cap | 215 | -0.24% | 24.84% | 11.50% | 46.91% | 14.11% | 72.09% | 22.96% | 23.95% |
| fwd_6m_return | low_cap | 20 | -30.93% | -23.88% | -20.90% | 0.29% | 2.80% | 75.00% | 22.66% | -22.38% |
| fwd_6m_return | mid_cap | 241 | -13.10% | -6.49% | -3.22% | 4.61% | 0.62% | 51.02% | 4.21% | 0.40% |

## Higher-value subsets

| forward_window | regression_bucket | condition | observations | avg_prediction | mean_return | median_return | hit_rate | avg_revenue_growth | avg_operating_margin | avg_fcf_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fwd_12m_return | all | rev_growth_30pct_positive_margin | 13 | 0.4477472877480103 | 58.64% | 33.42% | 76.92% | 59.27% | 27.32% | 17.15% |
| fwd_12m_return | all | rev_growth_30pct_model_top_half | 14 | 0.44106289549174293 | 53.35% | 27.95% | 71.43% | 59.04% | 27.32% | 18.16% |
| fwd_12m_return | all | rev_growth_30pct_positive_fcf | 13 | 0.41393247830040425 | 49.87% | 16.43% | 61.54% | 58.95% | 25.41% | 24.77% |
| fwd_12m_return | all | revenue_growth_30pct | 15 | 0.42476770366971306 | 49.29% | 22.48% | 66.67% | 57.12% | 27.32% | 17.75% |
| fwd_12m_return | all | model_top_quintile | 85 | 0.40467692607690414 | 43.00% | 10.17% | 63.53% | 19.43% | 19.01% | 14.73% |
| fwd_12m_return | all | positive_fcf_top_model | 72 | 0.40478279942021345 | 24.11% | 5.60% | 61.11% | 20.11% | 20.26% | 20.94% |
| fwd_12m_return | large_cap | model_top_quintile | 39 | 0.4783482479559073 | 90.69% | 31.95% | 79.49% | 20.61% | 11.32% | 11.57% |
| fwd_12m_return | large_cap | rev_growth_30pct_positive_margin | 10 | 0.43862088020905865 | 75.68% | 54.19% | 90.00% | 61.10% | 34.48% | 19.75% |
| fwd_12m_return | large_cap | rev_growth_30pct_positive_fcf | 9 | 0.47408146213858704 | 72.27% | 46.37% | 77.78% | 64.07% | 33.65% | 31.61% |
| fwd_12m_return | large_cap | rev_growth_30pct_model_top_half | 10 | 0.47059445604959915 | 71.89% | 54.19% | 80.00% | 63.49% | 33.88% | 27.75% |
| fwd_12m_return | large_cap | revenue_growth_30pct | 11 | 0.4475935847585683 | 67.39% | 46.37% | 81.82% | 60.64% | 34.48% | 20.79% |
| fwd_12m_return | large_cap | positive_fcf_top_model | 33 | 0.47456312547074825 | 52.77% | 31.08% | 78.79% | 22.18% | 13.37% | 16.02% |
| fwd_12m_return | mid_cap | model_top_quintile | 43 | 0.4150608135650571 | 3.21% | -0.45% | 48.84% | 8.66% | 20.68% | 10.40% |
| fwd_12m_return | mid_cap | positive_fcf_top_model | 38 | 0.41036379727442435 | 0.33% | -2.32% | 44.74% | 8.00% | 21.38% | 15.39% |
| fwd_3m_return | all | rev_growth_30pct_positive_margin | 39 | 0.02087513837039551 | 14.31% | 6.00% | 61.54% | 67.97% | 25.64% | 15.95% |
| fwd_3m_return | all | rev_growth_30pct_positive_fcf | 41 | 0.02499354734015921 | 12.13% | 9.03% | 60.98% | 62.31% | 23.68% | 23.05% |
| fwd_3m_return | all | revenue_growth_30pct | 47 | 0.022858498479965947 | 11.62% | 4.69% | 57.45% | 64.27% | 22.95% | 17.94% |
| fwd_3m_return | all | rev_growth_30pct_model_top_half | 25 | 0.07243389801574873 | 11.10% | 9.03% | 52.00% | 56.42% | 23.10% | 21.37% |
| fwd_3m_return | all | model_top_quintile | 204 | 0.11950565795271062 | 0.52% | 0.57% | 51.96% | -0.56% | 18.53% | 19.93% |
| fwd_3m_return | all | positive_fcf_top_model | 191 | 0.11969784774722962 | 0.51% | 0.54% | 51.31% | 0.50% | 19.73% | 22.27% |
| fwd_3m_return | large_cap | rev_growth_30pct_positive_margin | 22 | 0.08480728215952614 | 27.46% | 16.87% | 77.27% | 69.87% | 32.91% | 20.30% |
| fwd_3m_return | large_cap | revenue_growth_30pct | 27 | 0.08114491836948465 | 21.70% | 13.34% | 70.37% | 66.13% | 27.52% | 21.96% |
| fwd_3m_return | large_cap | rev_growth_30pct_positive_fcf | 24 | 0.08462670208765821 | 19.63% | 14.04% | 70.83% | 64.08% | 27.15% | 27.49% |
| fwd_3m_return | large_cap | rev_growth_30pct_model_top_half | 19 | 0.10646555180188581 | 18.00% | 14.75% | 68.42% | 65.04% | 24.35% | 23.65% |
| fwd_3m_return | large_cap | model_top_quintile | 93 | 0.15151781973110487 | 6.25% | 2.91% | 58.06% | 9.82% | 19.33% | 11.80% |
| fwd_3m_return | large_cap | positive_fcf_top_model | 81 | 0.15152768179013112 | 5.22% | 2.44% | 55.56% | 9.35% | 20.80% | 15.73% |
| fwd_3m_return | low_cap | model_top_quintile | 5 | 0.01136542670708893 | 4.07% | -1.45% | 40.00% | 10.34% | 26.73% | 29.27% |
| fwd_3m_return | low_cap | positive_fcf_top_model | 5 | 0.01136542670708893 | 4.07% | -1.45% | 40.00% | 10.34% | 26.73% | 29.27% |
| fwd_3m_return | mid_cap | rev_growth_30pct_positive_fcf | 16 | -0.05577678947463424 | 2.74% | 0.68% | 50.00% | 58.93% | 18.36% | 17.07% |
| fwd_3m_return | mid_cap | revenue_growth_30pct | 18 | -0.0621209060579129 | -0.38% | -1.39% | 44.44% | 60.26% | 16.58% | 13.72% |
| fwd_3m_return | mid_cap | rev_growth_30pct_positive_margin | 15 | -0.06467611574620383 | -0.86% | -1.62% | 46.67% | 64.23% | 16.58% | 11.46% |
| fwd_3m_return | mid_cap | model_top_quintile | 103 | 0.07203914786158089 | -2.15% | -0.75% | 45.63% | -4.78% | 13.52% | 16.46% |
| fwd_3m_return | mid_cap | positive_fcf_top_model | 97 | 0.07283005741126111 | -2.50% | -1.52% | 44.33% | -4.10% | 15.05% | 18.26% |
| fwd_6m_return | all | rev_growth_30pct_positive_margin | 16 | 0.1648123943428598 | 67.74% | 57.68% | 75.00% | 56.67% | 25.27% | 15.77% |
| fwd_6m_return | all | revenue_growth_30pct | 18 | 0.16102878136255852 | 61.28% | 39.77% | 72.22% | 55.16% | 25.27% | 16.42% |
| fwd_6m_return | all | rev_growth_30pct_positive_fcf | 16 | 0.16658240505536703 | 54.26% | 39.77% | 75.00% | 56.40% | 23.48% | 21.96% |
| fwd_6m_return | all | rev_growth_30pct_model_top_half | 16 | 0.17141091128731137 | 51.41% | 38.03% | 68.75% | 56.33% | 26.12% | 17.93% |
| fwd_6m_return | all | positive_fcf_top_model | 77 | 0.1727899210330779 | 17.86% | 6.45% | 63.64% | 16.07% | 25.32% | 24.94% |
| fwd_6m_return | all | model_top_quintile | 96 | 0.16946538699056124 | 15.56% | 6.21% | 63.54% | 13.84% | 24.41% | 16.62% |
| fwd_6m_return | large_cap | rev_growth_30pct_positive_margin | 12 | 0.1538311964967107 | 89.09% | 78.92% | 83.33% | 58.38% | 32.35% | 18.32% |
| fwd_6m_return | large_cap | revenue_growth_30pct | 13 | 0.15243496160864115 | 84.07% | 76.03% | 84.62% | 58.20% | 32.35% | 19.32% |
| fwd_6m_return | large_cap | rev_growth_30pct_positive_fcf | 11 | 0.17321166153839396 | 78.01% | 76.03% | 90.91% | 60.56% | 31.26% | 27.90% |
| fwd_6m_return | large_cap | rev_growth_30pct_model_top_half | 8 | 0.20386136725992196 | 72.11% | 74.14% | 100.00% | 68.57% | 34.54% | 35.26% |
| fwd_6m_return | large_cap | model_top_quintile | 43 | 0.21766230986369806 | 46.91% | 14.11% | 72.09% | 16.51% | 24.42% | 23.38% |
| fwd_6m_return | large_cap | positive_fcf_top_model | 40 | 0.2146091378297354 | 29.96% | 12.61% | 70.00% | 17.33% | 26.78% | 25.87% |
| fwd_6m_return | mid_cap | model_top_quintile | 49 | 0.19914784280701026 | 4.61% | 0.62% | 51.02% | 9.96% | 21.42% | 11.88% |
| fwd_6m_return | mid_cap | positive_fcf_top_model | 33 | 0.20588111788066094 | 2.07% | -2.00% | 45.45% | 12.53% | 20.89% | 24.14% |
| fwd_6m_return | mid_cap | revenue_growth_30pct | 5 | 0.1596762017611478 | 2.02% | -3.70% | 40.00% | 47.26% | 4.01% | 8.89% |
| fwd_6m_return | mid_cap | rev_growth_30pct_positive_fcf | 5 | 0.1596762017611478 | 2.02% | -3.70% | 40.00% | 47.26% | 4.01% | 8.89% |

## Average standardized coefficients

| forward_window | regression_bucket | feature | mean | median | count |
| --- | --- | --- | --- | --- | --- |
| fwd_12m_return | all | revenue_growth_yoy | 7.81% | 7.81% | 1 |
| fwd_12m_return | all | rd_to_revenue | 6.26% | 6.26% | 1 |
| fwd_12m_return | all | cash_to_assets | 6.19% | 6.19% | 1 |
| fwd_12m_return | all | debt_to_assets | 4.15% | 4.15% | 1 |
| fwd_12m_return | all | asset_turnover | 2.35% | 2.35% | 1 |
| fwd_12m_return | all | operating_margin | 1.10% | 1.10% | 1 |
| fwd_12m_return | all | sga_to_revenue | 0.94% | 0.94% | 1 |
| fwd_12m_return | all | cfo_margin | 0.83% | 0.83% | 1 |
| fwd_12m_return | all | capex_to_revenue | 0.65% | 0.65% | 1 |
| fwd_12m_return | all | net_margin | 0.03% | 0.03% | 1 |
| fwd_12m_return | all | revenue_growth_accel | 0.00% | 0.00% | 1 |
| fwd_12m_return | all | current_ratio | -1.23% | -1.23% | 1 |
| fwd_12m_return | all | ebitda_margin | -1.26% | -1.26% | 1 |
| fwd_12m_return | all | debt_to_equity | -2.16% | -2.16% | 1 |
| fwd_12m_return | all | fcf_margin | -2.18% | -2.18% | 1 |
| fwd_12m_return | all | gross_margin | -4.61% | -4.61% | 1 |
| fwd_12m_return | large_cap | revenue_growth_yoy | 10.42% | 10.42% | 1 |
| fwd_12m_return | large_cap | current_ratio | 4.85% | 4.85% | 1 |
| fwd_12m_return | large_cap | rd_to_revenue | 4.16% | 4.16% | 1 |
| fwd_12m_return | large_cap | debt_to_assets | 1.18% | 1.18% | 1 |
| fwd_12m_return | large_cap | fcf_margin | 0.33% | 0.33% | 1 |
| fwd_12m_return | large_cap | revenue_growth_accel | 0.00% | 0.00% | 1 |
| fwd_12m_return | large_cap | cfo_margin | -0.36% | -0.36% | 1 |
| fwd_12m_return | large_cap | debt_to_equity | -2.01% | -2.01% | 1 |
| fwd_12m_return | large_cap | capex_to_revenue | -2.40% | -2.40% | 1 |
| fwd_12m_return | large_cap | sga_to_revenue | -2.52% | -2.52% | 1 |
| fwd_12m_return | large_cap | operating_margin | -2.63% | -2.63% | 1 |
| fwd_12m_return | large_cap | net_margin | -2.64% | -2.64% | 1 |
| fwd_12m_return | large_cap | ebitda_margin | -3.17% | -3.17% | 1 |
| fwd_12m_return | large_cap | cash_to_assets | -3.41% | -3.41% | 1 |
| fwd_12m_return | large_cap | asset_turnover | -3.47% | -3.47% | 1 |
| fwd_12m_return | large_cap | gross_margin | -5.07% | -5.07% | 1 |
| fwd_12m_return | mid_cap | cash_to_assets | 11.07% | 11.07% | 1 |
| fwd_12m_return | mid_cap | net_margin | 5.80% | 5.80% | 1 |
| fwd_12m_return | mid_cap | asset_turnover | 5.80% | 5.80% | 1 |
| fwd_12m_return | mid_cap | revenue_growth_yoy | 3.49% | 3.49% | 1 |
| fwd_12m_return | mid_cap | debt_to_assets | 1.82% | 1.82% | 1 |
| fwd_12m_return | mid_cap | rd_to_revenue | 1.63% | 1.63% | 1 |
| fwd_12m_return | mid_cap | debt_to_equity | 1.09% | 1.09% | 1 |
| fwd_12m_return | mid_cap | ebitda_margin | 1.02% | 1.02% | 1 |
| fwd_12m_return | mid_cap | operating_margin | 0.85% | 0.85% | 1 |
| fwd_12m_return | mid_cap | sga_to_revenue | 0.03% | 0.03% | 1 |
| fwd_12m_return | mid_cap | revenue_growth_accel | 0.00% | 0.00% | 1 |
| fwd_12m_return | mid_cap | fcf_margin | -0.09% | -0.09% | 1 |
| fwd_12m_return | mid_cap | cfo_margin | -0.38% | -0.38% | 1 |
| fwd_12m_return | mid_cap | capex_to_revenue | -1.15% | -1.15% | 1 |
| fwd_12m_return | mid_cap | gross_margin | -2.03% | -2.03% | 1 |
| fwd_12m_return | mid_cap | current_ratio | -6.08% | -6.08% | 1 |
| fwd_3m_return | all | rd_to_revenue | 2.16% | 1.82% | 3 |
| fwd_3m_return | all | ebitda_margin | 1.53% | 1.28% | 3 |

## Interpretation

- Using current S&P 500 constituents increases annual sample size materially, but adds survivorship bias.
- Compare the large-cap rows to the prior annual study; they are the most relevant apples-to-apples results.
- If predictive power improves mainly from broader current membership, the next step should be point-in-time constituents rather than more current-list expansion.
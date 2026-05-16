# Financial Ratio Regression Study

Question: can many annual financial ratios predict 3m, 6m, or 12m forward stock returns, and are there subsets with higher predictive value?

## Leakage controls

- Annual financials are assumed available only 90 calendar days after fiscal year end.
- Forward returns start from the first trading day on or after that availability date.
- For each test year and horizon, training rows are purged unless their forward-return end date is before the test year starts.
- Regression uses fixed ridge alpha, not tuned on the validation rows.
- Market-cap buckets use current Yahoo market cap, not point-in-time market cap; treat bucket results as approximate.

## Dataset

- Financial events: 531.
- Prediction rows: 963.
- Symbols with predictions: 177.

## Walk-forward predictive power

| forward_window | regression_bucket | observations | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fwd_12m_return | all | 150 | -5.12% | 2.51% | 16.23% | 21.99% | 23.33% | 70.00% | 17.75% | 4.24% |
| fwd_12m_return | large_cap | 117 | 1.89% | 19.65% | 12.75% | 34.51% | 23.74% | 75.00% | 17.47% | 17.04% |
| fwd_3m_return | all | 181 | -19.26% | -5.74% | -16.84% | 11.90% | 3.42% | 59.46% | 12.24% | -0.34% |
| fwd_3m_return | large_cap | 142 | 1.90% | 33.16% | 14.82% | 14.96% | 11.98% | 68.97% | 2.25% | 12.71% |
| fwd_3m_return | low_cap | 21 | -70.30% | -32.77% | -35.71% | 1.32% | 0.88% | 60.00% | 33.39% | -32.07% |
| fwd_3m_return | mid_cap | 18 | -73.33% | -49.14% | -50.05% | -7.17% | 0.20% | 50.00% | 37.56% | -44.73% |
| fwd_6m_return | all | 167 | -7.18% | -17.18% | -13.54% | 10.46% | 4.37% | 67.65% | 39.22% | -28.76% |
| fwd_6m_return | large_cap | 128 | 1.97% | 19.62% | 3.93% | 26.56% | 13.57% | 80.77% | 12.50% | 14.06% |
| fwd_6m_return | low_cap | 21 | -2.73% | 41.33% | 41.95% | 202.05% | 128.68% | 80.00% | -10.84% | 212.89% |
| fwd_6m_return | mid_cap | 18 | -31.03% | -43.38% | -26.73% | -17.54% | -10.71% | 25.00% | 48.93% | -66.47% |

## Higher-value condition subsets

| forward_window | regression_bucket | condition | observations | avg_prediction | mean_return | median_return | hit_rate | avg_revenue_growth | avg_operating_margin | avg_fcf_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fwd_12m_return | all | rev_growth_30pct_model_top_half | 6 | 0.37820524946442946 | 37.44% | 23.29% | 50.00% | 59.35% | 27.14% | 27.24% |
| fwd_12m_return | all | model_top_quintile | 30 | 0.3597301587486987 | 21.99% | 23.33% | 70.00% | 17.80% | 24.44% | 21.38% |
| fwd_12m_return | all | positive_fcf_top_model | 30 | 0.3597301587486987 | 21.99% | 23.33% | 70.00% | 17.80% | 24.44% | 21.38% |
| fwd_12m_return | all | rev_growth_30pct_positive_margin | 9 | 0.27506889132324885 | 16.93% | 0.44% | 55.56% | 57.79% | 26.92% | 25.84% |
| fwd_12m_return | all | rev_growth_30pct_positive_fcf | 11 | 0.24740905907427593 | 11.00% | -15.42% | 45.45% | 55.31% | 20.21% | 24.06% |
| fwd_12m_return | all | revenue_growth_30pct | 12 | 0.22025133640125039 | 6.11% | -15.71% | 41.67% | 53.27% | 16.67% | 21.31% |
| fwd_12m_return | large_cap | rev_growth_30pct_positive_margin | 6 | 0.3796021294480653 | 45.84% | 39.22% | 83.33% | 68.43% | 33.22% | 29.73% |
| fwd_12m_return | large_cap | revenue_growth_30pct | 7 | 0.3718204629284128 | 37.08% | 16.43% | 71.43% | 66.66% | 33.22% | 29.96% |
| fwd_12m_return | large_cap | rev_growth_30pct_model_top_half | 7 | 0.3718204629284128 | 37.08% | 16.43% | 71.43% | 66.66% | 33.22% | 29.96% |
| fwd_12m_return | large_cap | rev_growth_30pct_positive_fcf | 7 | 0.3718204629284128 | 37.08% | 16.43% | 71.43% | 66.66% | 33.22% | 29.96% |
| fwd_12m_return | large_cap | model_top_quintile | 24 | 0.3679368107465912 | 34.51% | 23.74% | 75.00% | 28.18% | 23.84% | 21.77% |
| fwd_12m_return | large_cap | positive_fcf_top_model | 24 | 0.3679368107465912 | 34.51% | 23.74% | 75.00% | 28.18% | 23.84% | 21.77% |
| fwd_3m_return | all | rev_growth_30pct_positive_margin | 10 | 0.037014335111057416 | 32.93% | 26.11% | 80.00% | 56.90% | 26.85% | 23.70% |
| fwd_3m_return | all | rev_growth_30pct_model_top_half | 8 | 0.06833221831065119 | 30.76% | 26.11% | 62.50% | 60.02% | 26.35% | 22.40% |
| fwd_3m_return | all | rev_growth_30pct_positive_fcf | 12 | 0.030507711334479992 | 27.99% | 18.58% | 75.00% | 54.77% | 20.76% | 22.43% |
| fwd_3m_return | all | revenue_growth_30pct | 13 | 0.02919337245105362 | 25.00% | 14.75% | 69.23% | 52.93% | 17.47% | 20.02% |
| fwd_3m_return | all | positive_fcf_top_model | 35 | 0.07582994672840683 | 13.07% | 4.94% | 62.86% | 21.60% | 21.41% | 21.12% |
| fwd_3m_return | all | model_top_quintile | 37 | 0.07519638394826249 | 11.90% | 3.42% | 59.46% | 20.97% | 21.03% | 19.72% |
| fwd_3m_return | large_cap | rev_growth_30pct_model_top_half | 7 | 0.08793780200630437 | 36.92% | 22.42% | 85.71% | 67.35% | 32.74% | 25.22% |
| fwd_3m_return | large_cap | rev_growth_30pct_positive_margin | 7 | 0.08482777821063846 | 34.57% | 22.42% | 71.43% | 65.64% | 32.22% | 26.12% |
| fwd_3m_return | large_cap | revenue_growth_30pct | 8 | 0.0780034960316259 | 31.50% | 18.58% | 75.00% | 64.43% | 32.22% | 26.77% |
| fwd_3m_return | large_cap | rev_growth_30pct_positive_fcf | 8 | 0.0780034960316259 | 31.50% | 18.58% | 75.00% | 64.43% | 32.22% | 26.77% |
| fwd_3m_return | large_cap | model_top_quintile | 29 | 0.07429293257023846 | 14.96% | 11.98% | 68.97% | 22.38% | 29.46% | 20.99% |
| fwd_3m_return | large_cap | positive_fcf_top_model | 29 | 0.07429293257023846 | 14.96% | 11.98% | 68.97% | 22.38% | 29.46% | 20.99% |
| fwd_3m_return | low_cap | revenue_growth_30pct | 5 | -0.059978855462808366 | 14.59% | 8.68% | 60.00% | 34.54% | -3.19% | 9.21% |
| fwd_3m_return | low_cap | model_top_quintile | 5 | 0.07415120438032943 | 1.32% | 0.88% | 60.00% | 14.04% | -1.33% | 9.65% |
| fwd_6m_return | all | rev_growth_30pct_positive_margin | 9 | 0.07950225888311008 | 33.62% | 12.73% | 77.78% | 57.79% | 26.92% | 25.84% |
| fwd_6m_return | all | rev_growth_30pct_model_top_half | 6 | 0.12383989846468817 | 31.87% | 27.91% | 66.67% | 55.35% | 28.92% | 22.17% |
| fwd_6m_return | all | rev_growth_30pct_positive_fcf | 11 | 0.06776418815110047 | 28.91% | 12.73% | 72.73% | 55.31% | 20.21% | 24.06% |
| fwd_6m_return | all | revenue_growth_30pct | 12 | 0.05703050197924895 | 26.59% | 12.25% | 75.00% | 53.27% | 16.67% | 21.31% |
| fwd_6m_return | all | positive_fcf_top_model | 31 | 0.16341191616291756 | 10.46% | 4.34% | 67.74% | 12.08% | 31.03% | 21.56% |
| fwd_6m_return | all | model_top_quintile | 34 | 0.1608355546299866 | 10.46% | 4.37% | 67.65% | 10.68% | 30.14% | 18.58% |
| fwd_6m_return | large_cap | rev_growth_30pct_model_top_half | 6 | 0.22227030274466245 | 55.58% | 59.56% | 100.00% | 72.44% | 32.29% | 34.80% |
| fwd_6m_return | large_cap | rev_growth_30pct_positive_margin | 6 | 0.20857226313483657 | 50.40% | 59.56% | 83.33% | 68.43% | 33.22% | 29.73% |
| fwd_6m_return | large_cap | revenue_growth_30pct | 7 | 0.20094116474275922 | 46.60% | 43.10% | 85.71% | 66.66% | 33.22% | 29.96% |
| fwd_6m_return | large_cap | rev_growth_30pct_positive_fcf | 7 | 0.20094116474275922 | 46.60% | 43.10% | 85.71% | 66.66% | 33.22% | 29.96% |
| fwd_6m_return | large_cap | model_top_quintile | 26 | 0.1843651820732564 | 26.56% | 13.57% | 80.77% | 23.29% | 27.54% | 19.73% |
| fwd_6m_return | large_cap | positive_fcf_top_model | 26 | 0.1843651820732564 | 26.56% | 13.57% | 80.77% | 23.29% | 27.54% | 19.73% |
| fwd_6m_return | low_cap | model_top_quintile | 5 | 0.2071004958047645 | 202.05% | 128.68% | 80.00% | -15.04% | -49.16% | -39.64% |
| fwd_6m_return | low_cap | revenue_growth_30pct | 5 | -0.07681429136815292 | -1.42% | 1.11% | 60.00% | 34.54% | -3.19% | 9.21% |

## Average standardized coefficients

| forward_window | regression_bucket | feature | mean | median | count |
| --- | --- | --- | --- | --- | --- |
| fwd_12m_return | all | asset_turnover | 4.77% | 4.77% | 1 |
| fwd_12m_return | all | net_margin | 2.89% | 2.89% | 1 |
| fwd_12m_return | all | cfo_margin | 2.89% | 2.89% | 1 |
| fwd_12m_return | all | revenue_growth_yoy | 2.34% | 2.34% | 1 |
| fwd_12m_return | all | fcf_margin | 2.19% | 2.19% | 1 |
| fwd_12m_return | all | ebitda_margin | 1.93% | 1.93% | 1 |
| fwd_12m_return | all | operating_margin | 1.12% | 1.12% | 1 |
| fwd_12m_return | all | current_ratio | 0.99% | 0.99% | 1 |
| fwd_12m_return | all | debt_to_equity | 0.70% | 0.70% | 1 |
| fwd_12m_return | all | rd_to_revenue | 0.56% | 0.56% | 1 |
| fwd_12m_return | all | revenue_growth_accel | 0.00% | 0.00% | 1 |
| fwd_12m_return | all | capex_to_revenue | -0.82% | -0.82% | 1 |
| fwd_12m_return | all | debt_to_assets | -3.17% | -3.17% | 1 |
| fwd_12m_return | all | cash_to_assets | -5.07% | -5.07% | 1 |
| fwd_12m_return | all | sga_to_revenue | -5.50% | -5.50% | 1 |
| fwd_12m_return | all | gross_margin | -6.22% | -6.22% | 1 |
| fwd_12m_return | large_cap | revenue_growth_yoy | 5.69% | 5.69% | 1 |
| fwd_12m_return | large_cap | rd_to_revenue | 4.37% | 4.37% | 1 |
| fwd_12m_return | large_cap | asset_turnover | 3.76% | 3.76% | 1 |
| fwd_12m_return | large_cap | net_margin | 1.68% | 1.68% | 1 |
| fwd_12m_return | large_cap | ebitda_margin | 1.25% | 1.25% | 1 |
| fwd_12m_return | large_cap | fcf_margin | 0.92% | 0.92% | 1 |
| fwd_12m_return | large_cap | cfo_margin | 0.64% | 0.64% | 1 |
| fwd_12m_return | large_cap | revenue_growth_accel | 0.00% | 0.00% | 1 |
| fwd_12m_return | large_cap | capex_to_revenue | -0.41% | -0.41% | 1 |
| fwd_12m_return | large_cap | current_ratio | -0.98% | -0.98% | 1 |
| fwd_12m_return | large_cap | debt_to_equity | -1.12% | -1.12% | 1 |
| fwd_12m_return | large_cap | operating_margin | -2.02% | -2.02% | 1 |
| fwd_12m_return | large_cap | cash_to_assets | -2.12% | -2.12% | 1 |
| fwd_12m_return | large_cap | gross_margin | -3.08% | -3.08% | 1 |
| fwd_12m_return | large_cap | debt_to_assets | -3.16% | -3.16% | 1 |
| fwd_12m_return | large_cap | sga_to_revenue | -3.71% | -3.71% | 1 |
| fwd_3m_return | all | revenue_growth_yoy | 2.92% | 2.92% | 2 |
| fwd_3m_return | all | rd_to_revenue | 2.83% | 2.83% | 2 |
| fwd_3m_return | all | asset_turnover | 1.64% | 1.64% | 2 |
| fwd_3m_return | all | operating_margin | 1.41% | 1.41% | 2 |
| fwd_3m_return | all | net_margin | 1.40% | 1.40% | 2 |
| fwd_3m_return | all | cash_to_assets | 1.29% | 1.29% | 2 |
| fwd_3m_return | all | capex_to_revenue | 0.91% | 0.91% | 2 |
| fwd_3m_return | all | sga_to_revenue | 0.66% | 0.66% | 2 |
| fwd_3m_return | all | ebitda_margin | 0.49% | 0.49% | 2 |
| fwd_3m_return | all | debt_to_equity | -0.02% | -0.02% | 2 |
| fwd_3m_return | all | cfo_margin | -0.03% | -0.03% | 2 |
| fwd_3m_return | all | fcf_margin | -0.20% | -0.20% | 2 |
| fwd_3m_return | all | revenue_growth_accel | -0.53% | -0.53% | 2 |
| fwd_3m_return | all | debt_to_assets | -0.69% | -0.69% | 2 |
| fwd_3m_return | all | gross_margin | -2.02% | -2.02% | 2 |
| fwd_3m_return | all | current_ratio | -3.03% | -3.03% | 2 |
| fwd_3m_return | large_cap | revenue_growth_yoy | 2.92% | 2.92% | 2 |
| fwd_3m_return | large_cap | net_margin | 2.00% | 2.00% | 2 |
| fwd_3m_return | large_cap | asset_turnover | 1.45% | 1.45% | 2 |
| fwd_3m_return | large_cap | cash_to_assets | 1.30% | 1.30% | 2 |
| fwd_3m_return | large_cap | current_ratio | 0.99% | 0.99% | 2 |
| fwd_3m_return | large_cap | capex_to_revenue | 0.89% | 0.89% | 2 |
| fwd_3m_return | large_cap | sga_to_revenue | 0.78% | 0.78% | 2 |
| fwd_3m_return | large_cap | rd_to_revenue | 0.60% | 0.60% | 2 |
| fwd_3m_return | large_cap | ebitda_margin | 0.48% | 0.48% | 2 |
| fwd_3m_return | large_cap | revenue_growth_accel | 0.25% | 0.25% | 2 |
| fwd_3m_return | large_cap | debt_to_equity | 0.16% | 0.16% | 2 |
| fwd_3m_return | large_cap | operating_margin | -0.07% | -0.07% | 2 |

## Interpretation

- Out-of-sample R2 is generally weak or negative, so annual financial ratios alone are not a strong standalone return model in this sample.
- The most useful output is ranking/conditioning: top predicted quintiles and 30%+ revenue growth combined with positive margins or FCF can identify better pockets than raw revenue growth alone.
- Predictive value differs by market-cap bucket; small samples in low/mid caps make those results fragile.
- A deployable version would need point-in-time filings, point-in-time market caps, more history, and transaction-cost-aware portfolio construction.
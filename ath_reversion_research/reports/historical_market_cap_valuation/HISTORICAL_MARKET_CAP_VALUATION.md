# Historical market-cap valuation feature test

This test replaces current market-cap valuation ratios with ratios computed from `shares_outstanding * trade_close` at the trade date. Average shares is used only when ending shares outstanding is missing.

## No-leakage setup

- Fundamental rows are annual Dolt rows already shifted to their reporting/trade date.
- Market cap uses the trade-date close and annual share count from the same fiscal row.
- Walk-forward training only uses observations whose 12-month forward window ended before the test year.
- The calculation remains approximate if Dolt share counts and Yahoo adjusted closes differ in split convention.

Row-level event/prediction CSVs are skipped by default to avoid large generated files. Set `SAVE_ROW_LEVEL_HIST_VALUATION=1` when running the script to export them locally.

## Feature coverage

| field                           | coverage | median           |
| ------------------------------- | -------- | ---------------- |
| historical_market_cap           | 1.0000   | 21330580223.0835 |
| shares_outstanding              | 0.9935   | 299200000.0000   |
| average_shares                  | 1.0000   | 309000000.0000   |
| hist_sales_to_market_cap        | 0.9998   | 0.4646           |
| hist_earnings_yield             | 1.0000   | 0.0485           |
| hist_fcf_yield                  | 0.9906   | 0.0524           |
| hist_gross_profit_to_market_cap | 1.0000   | 0.1913           |
| hist_ebitda_to_market_cap       | 1.0000   | 0.1017           |
| hist_growth_to_sales_multiple   | 0.9998   | 0.0208           |

## Performance summary

| experiment                       | target_type          | regression_bucket | rows | years | spearman | target_spearman | top_quintile_mean_return | top_quintile_hit_rate | mean_yearly_top_minus_bottom | positive_yearly_spread_rate |
| -------------------------------- | -------------------- | ----------------- | ---- | ----- | -------- | --------------- | ------------------------ | --------------------- | ---------------------------- | --------------------------- |
| hist_and_current_valuation       | raw_return           | all               | 4605 | 10    | 0.3087   | 0.3087          | 0.4325                   | 0.7755                | 0.2809                       | 1.0000                      |
| current_cap_valuation            | raw_return           | all               | 4605 | 10    | 0.2880   | 0.2880          | 0.3890                   | 0.7473                | 0.2236                       | 1.0000                      |
| hist_cap_valuation               | raw_return           | all               | 4605 | 10    | 0.2344   | 0.2344          | 0.4092                   | 0.7646                | 0.1872                       | 0.8000                      |
| baseline_no_valuation            | raw_return           | all               | 4605 | 10    | 0.2295   | 0.2295          | 0.3693                   | 0.7430                | 0.1581                       | 0.7000                      |
| hist_cap_valuation_sector_excess | sector_excess_return | all               | 4561 | 10    | 0.1974   | 0.0828          | 0.3268                   | 0.7415                | 0.1722                       | 0.8000                      |
| hist_and_current_valuation       | raw_return           | large_cap         | 2101 | 10    | 0.2935   | 0.2935          | 0.5013                   | 0.7933                | 0.2993                       | 0.9000                      |
| current_cap_valuation            | raw_return           | large_cap         | 2101 | 10    | 0.2854   | 0.2854          | 0.4939                   | 0.7767                | 0.2610                       | 0.9000                      |
| baseline_no_valuation            | raw_return           | large_cap         | 2101 | 10    | 0.2499   | 0.2499          | 0.4631                   | 0.7672                | 0.2013                       | 0.9000                      |
| hist_cap_valuation               | raw_return           | large_cap         | 2101 | 10    | 0.2449   | 0.2449          | 0.4579                   | 0.7838                | 0.2171                       | 0.9000                      |
| hist_cap_valuation_sector_excess | sector_excess_return | large_cap         | 2076 | 10    | 0.2385   | 0.0423          | 0.4303                   | 0.7764                | 0.2043                       | 0.8000                      |
| hist_and_current_valuation       | raw_return           | mid_cap           | 2302 | 10    | 0.2724   | 0.2724          | 0.3363                   | 0.7505                | 0.1751                       | 1.0000                      |
| current_cap_valuation            | raw_return           | mid_cap           | 2302 | 10    | 0.2445   | 0.2445          | 0.2800                   | 0.7158                | 0.1348                       | 0.8000                      |
| hist_cap_valuation               | raw_return           | mid_cap           | 2302 | 10    | 0.2030   | 0.2030          | 0.2743                   | 0.7137                | 0.0943                       | 0.5000                      |
| baseline_no_valuation            | raw_return           | mid_cap           | 2302 | 10    | 0.1953   | 0.1953          | 0.2582                   | 0.7093                | 0.0741                       | 0.5000                      |
| hist_cap_valuation_sector_excess | sector_excess_return | mid_cap           | 2284 | 10    | 0.1187   | 0.0832          | 0.1999                   | 0.6761                | 0.0661                       | 0.6000                      |

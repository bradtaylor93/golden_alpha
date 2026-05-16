# Yahoo vs Dolt Feature Reconciliation

Compares matched rows by `symbol + fiscal_period_end` to determine whether Yahoo and Dolt fundamentals are equivalent.

## Coverage

| metric | value |
| --- | --- |
| yahoo_rows | 1507 |
| dolt_rows | 5871 |
| matched_rows | 1411 |
| yahoo_symbols | 503 |
| dolt_symbols | 496 |
| matched_symbols | 496 |
| matched_min_fiscal_year | 2023 |
| matched_max_fiscal_year | 2026 |

## Feature comparison

| feature | status | observations | pearson_corr | spearman_corr | median_abs_pct_diff | p90_abs_pct_diff | median_signed_pct_diff | same_sign_rate | yahoo_median | dolt_median | yahoo_column | dolt_column |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trade_close | ok | 1411 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 122.77999877929688 | 122.77999877929688 | trade_close_yahoo | trade_close_dolt |
| fwd_12m_return | ok | 982 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 0.10843872887492931 | 0.10843872887492931 | fwd_12m_return_yahoo | fwd_12m_return_dolt |
| net_income | ok | 1411 | 100.00% | 99.96% | 0.00% | 0.07% | 0.00% | 99.93% | 1509120000.0 | 1509000000.0 | net_income_yahoo | net_income_dolt |
| operating_cash_flow | ok | 1401 | 99.99% | 99.60% | 0.00% | 0.00% | 0.00% | 100.00% | 2407200000.0 | 2393200000.0 | operating_cash_flow | net_cash_from_operating_activities |
| total_assets | ok | 1399 | 99.95% | 99.77% | 0.00% | 0.00% | 0.00% | 100.00% | 29076181000.0 | 29264000000.0 | total_assets_yahoo | total_assets_dolt |
| asset_turnover | ok | 1399 | 99.82% | 99.47% | 0.00% | 1.90% | 0.00% | 100.00% | 0.5158396253735037 | 0.5191149147007489 | asset_turnover_yahoo | asset_turnover_dolt |
| equity | ok | 1397 | 99.79% | 99.27% | 0.05% | 10.21% | 0.03% | 99.57% | 9324000000.0 | 9665000000.0 | stockholders_equity | total_equity |
| ebitda | ok | 1287 | 99.61% | 97.28% | 3.76% | 22.87% | -1.72% | 98.99% | 2814000000.0 | 2773000000.0 | ebitda_yahoo | ebitda_dolt |
| operating_income | ok | 1285 | 99.59% | 96.33% | 1.69% | 32.15% | -0.02% | 97.90% | 2015000000.0 | 1871000000.0 | operating_income_yahoo | operating_income_dolt |
| revenue | ok | 1411 | 99.46% | 99.40% | 0.00% | 1.97% | 0.00% | 100.00% | 13627000000.0 | 13702000000.0 | total_revenue | sales |
| net_margin | ok | 1411 | 98.93% | 98.53% | 0.01% | 2.92% | 0.00% | 99.93% | 0.126575198236436 | 0.1244935170178282 | net_margin_yahoo | net_margin_dolt |
| debt | ok | 1388 | 98.01% | 95.44% | 13.64% | 63.49% | -13.30% | 93.30% | 8439612000.0 | 6973000000.0 | total_debt | long_term_debt |
| gross_profit | ok | 1265 | 97.06% | 96.97% | 0.00% | 36.85% | 0.00% | 100.00% | 4895000000.0 | 5383000000.0 | gross_profit_yahoo | gross_profit_dolt |
| cfo_margin | ok | 1401 | 96.07% | 98.81% | 0.00% | 2.43% | 0.00% | 100.00% | 0.2038985717842144 | 0.2014097968936678 | cfo_margin_yahoo | cfo_margin_dolt |
| ebitda_margin | ok | 1287 | 95.66% | 96.58% | 3.79% | 22.31% | -1.93% | 98.99% | 0.2428597550385865 | 0.2362405416168857 | ebitda_margin_yahoo | ebitda_margin_dolt |
| operating_margin | ok | 1285 | 94.00% | 95.34% | 1.78% | 30.56% | -0.03% | 97.90% | 0.1820868328667289 | 0.1728428455504463 | operating_margin_yahoo | operating_margin_dolt |
| debt_to_assets | ok | 1388 | 93.80% | 88.72% | 13.66% | 61.89% | -13.34% | 93.30% | 0.30367241418871016 | 0.2408622584849647 | debt_to_assets_yahoo | debt_to_assets_dolt |
| cash | ok | 1399 | 90.79% | 94.75% | 0.46% | 128.44% | 0.33% | 100.00% | 1497785000.0 | 1845000000.0 | cash | cash_and_equivalents |
| fcf_margin | ok | 1404 | 89.84% | 95.12% | 0.34% | 34.11% | 0.00% | 98.58% | 0.1337548330781333 | 0.136897531604764 | fcf_margin_yahoo | fcf_margin_dolt |
| gross_margin | ok | 1265 | 87.19% | 88.25% | 0.00% | 31.03% | 0.00% | 100.00% | 0.4402260491401062 | 0.4623718643220268 | gross_margin_yahoo | gross_margin_dolt |
| current_assets | ok | 1287 | 86.84% | 98.11% | 0.00% | 0.08% | 0.00% | 100.00% | 5900200000.0 | 5777000000.0 | current_assets | total_current_assets |
| capex_to_revenue | ok | 1334 | 85.58% | 83.63% | 0.56% | 48.78% | -0.12% | 96.03% | 0.03814588900011185 | 0.0328637884549082 | capex_to_revenue_yahoo | capex_to_revenue_dolt |
| debt_to_equity | ok | 1386 | 81.83% | 92.32% | 16.40% | 71.99% | -15.68% | 93.29% | 0.7940128632732066 | 0.6463309407577653 | debt_to_equity_yahoo | debt_to_equity_dolt |
| cash_to_assets | ok | 1399 | 79.16% | 91.66% | 0.46% | 128.44% | 0.33% | 100.00% | 0.0614940924737184 | 0.0696958756683342 | cash_to_assets_yahoo | cash_to_assets_dolt |
| current_liabilities | ok | 1287 | 78.90% | 98.15% | 0.00% | 1.70% | 0.00% | 99.84% | 4476000000.0 | 4422000000.0 | current_liabilities | total_current_liabilities |
| current_ratio | ok | 1285 | 67.86% | 92.86% | 0.00% | 2.48% | 0.00% | 100.00% | 1.2335683394611838 | 1.2550711021907162 | current_ratio_yahoo | current_ratio_dolt |
| revenue_growth_yoy | ok | 1411 | 57.67% | 93.41% | 0.02% | 28.64% | 0.00% | 96.46% | 0.0563743085173769 | 0.0571643474990597 | revenue_growth_yoy_yahoo | revenue_growth_yoy_dolt |

## Fields requiring attention

| feature | status | observations | pearson_corr | spearman_corr | median_abs_pct_diff | p90_abs_pct_diff | median_signed_pct_diff | same_sign_rate | yahoo_median | dolt_median | yahoo_column | dolt_column |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| debt | ok | 1388 | 98.01% | 95.44% | 13.64% | 63.49% | -13.30% | 93.30% | 8439612000.0 | 6973000000.0 | total_debt | long_term_debt |
| operating_margin | ok | 1285 | 94.00% | 95.34% | 1.78% | 30.56% | -0.03% | 97.90% | 0.1820868328667289 | 0.1728428455504463 | operating_margin_yahoo | operating_margin_dolt |
| debt_to_assets | ok | 1388 | 93.80% | 88.72% | 13.66% | 61.89% | -13.34% | 93.30% | 0.30367241418871016 | 0.2408622584849647 | debt_to_assets_yahoo | debt_to_assets_dolt |
| cash | ok | 1399 | 90.79% | 94.75% | 0.46% | 128.44% | 0.33% | 100.00% | 1497785000.0 | 1845000000.0 | cash | cash_and_equivalents |
| fcf_margin | ok | 1404 | 89.84% | 95.12% | 0.34% | 34.11% | 0.00% | 98.58% | 0.1337548330781333 | 0.136897531604764 | fcf_margin_yahoo | fcf_margin_dolt |
| gross_margin | ok | 1265 | 87.19% | 88.25% | 0.00% | 31.03% | 0.00% | 100.00% | 0.4402260491401062 | 0.4623718643220268 | gross_margin_yahoo | gross_margin_dolt |
| current_assets | ok | 1287 | 86.84% | 98.11% | 0.00% | 0.08% | 0.00% | 100.00% | 5900200000.0 | 5777000000.0 | current_assets | total_current_assets |
| capex_to_revenue | ok | 1334 | 85.58% | 83.63% | 0.56% | 48.78% | -0.12% | 96.03% | 0.03814588900011185 | 0.0328637884549082 | capex_to_revenue_yahoo | capex_to_revenue_dolt |
| debt_to_equity | ok | 1386 | 81.83% | 92.32% | 16.40% | 71.99% | -15.68% | 93.29% | 0.7940128632732066 | 0.6463309407577653 | debt_to_equity_yahoo | debt_to_equity_dolt |
| cash_to_assets | ok | 1399 | 79.16% | 91.66% | 0.46% | 128.44% | 0.33% | 100.00% | 0.0614940924737184 | 0.0696958756683342 | cash_to_assets_yahoo | cash_to_assets_dolt |
| current_liabilities | ok | 1287 | 78.90% | 98.15% | 0.00% | 1.70% | 0.00% | 99.84% | 4476000000.0 | 4422000000.0 | current_liabilities | total_current_liabilities |
| current_ratio | ok | 1285 | 67.86% | 92.86% | 0.00% | 2.48% | 0.00% | 100.00% | 1.2335683394611838 | 1.2550711021907162 | current_ratio_yahoo | current_ratio_dolt |
| revenue_growth_yoy | ok | 1411 | 57.67% | 93.41% | 0.02% | 28.64% | 0.00% | 96.46% | 0.0563743085173769 | 0.0571643474990597 | revenue_growth_yoy_yahoo | revenue_growth_yoy_dolt |

## Largest discrepancies

| feature | symbol | fiscal_period_end | yahoo_value | dolt_value | abs_pct_diff |
| --- | --- | --- | --- | --- | --- |
| revenue | BK | 2024-12-31 00:00:00 | 18258000000.0 | 39914000000.0 | 118.61% |
| revenue | C | 2024-12-31 00:00:00 | 80672000000.0 | 170757000000.0 | 111.67% |
| revenue | KEY | 2024-12-31 00:00:00 | 4396000000.0 | 9236000000.0 | 110.10% |
| revenue | BK | 2025-12-31 00:00:00 | 19759000000.0 | 40762000000.0 | 106.30% |
| revenue | WDC | 2024-06-30 00:00:00 | 6317000000.0 | 13003000000.0 | 105.84% |
| revenue | C | 2023-12-31 00:00:00 | 78090000000.0 | 156820000000.0 | 100.82% |
| revenue | C | 2025-12-31 00:00:00 | 85213000000.0 | 168297000000.0 | 97.50% |
| revenue | WDC | 2023-06-30 00:00:00 | 6255000000.0 | 12318000000.0 | 96.93% |
| revenue | BK | 2023-12-31 00:00:00 | 17344000000.0 | 33805000000.0 | 94.91% |
| revenue | GE | 2023-12-31 00:00:00 | 35348000000.0 | 67954000000.0 | 92.24% |
| gross_profit | F | 2025-12-31 00:00:00 | 1680000000.0 | 12801000000.0 | 661.96% |
| gross_profit | EQT | 2023-12-31 00:00:00 | 941579000.0 | 4494000000.0 | 377.28% |
| gross_profit | UPS | 2025-12-31 00:00:00 | 16030000000.0 | 70650000000.0 | 340.74% |
| gross_profit | UPS | 2024-12-31 00:00:00 | 16356000000.0 | 70175000000.0 | 329.05% |
| gross_profit | UPS | 2023-12-31 00:00:00 | 17238000000.0 | 69704000000.0 | 304.36% |
| gross_profit | EQT | 2024-12-31 00:00:00 | 767219000.0 | 2868000000.0 | 273.82% |
| gross_profit | JBHT | 2024-12-31 00:00:00 | 1567783000.0 | 5561000000.0 | 254.70% |
| gross_profit | JBHT | 2025-12-31 00:00:00 | 1596624000.0 | 5548000000.0 | 247.48% |
| gross_profit | JBHT | 2023-12-31 00:00:00 | 1700785000.0 | 5696000000.0 | 234.90% |
| gross_profit | FDX | 2025-05-31 00:00:00 | 18995000000.0 | 62383000000.0 | 228.42% |
| operating_income | WBD | 2024-12-31 00:00:00 | 18000000.0 | -10032000000.0 | 55833.33% |
| operating_income | INTC | 2025-12-31 00:00:00 | -23000000.0 | -2214000000.0 | 9526.09% |
| operating_income | CNC | 2025-12-31 00:00:00 | -312000000.0 | -7623000000.0 | 2343.27% |
| operating_income | IP | 2025-12-31 00:00:00 | -10000000.0 | 200000000.0 | 2100.00% |
| operating_income | DOW | 2025-12-31 00:00:00 | 158000000.0 | -1698000000.0 | 1174.68% |
| operating_income | HAS | 2023-12-31 00:00:00 | 191400000.0 | -1539000000.0 | 904.08% |
| operating_income | TKO | 2024-12-31 00:00:00 | 30946000.0 | 283000000.0 | 814.50% |
| operating_income | STX | 2023-06-30 00:00:00 | 60000000.0 | -342000000.0 | 670.00% |
| operating_income | ALB | 2025-12-31 00:00:00 | 67285000.0 | -367000000.0 | 645.44% |
| operating_income | TTWO | 2025-03-31 00:00:00 | -739400000.0 | -4391000000.0 | 493.86% |
| net_income | CRL | 2024-12-31 00:00:00 | 10297000.0 | 22000000.0 | 113.65% |
| net_income | SW | 2023-12-31 00:00:00 | 825000000.0 | -22000000.0 | 102.67% |
| net_income | MCHP | 2025-03-31 00:00:00 | -500000.0 | -1000000.0 | 100.00% |
| net_income | DOC | 2023-12-31 00:00:00 | 306009000.0 | 42000000.0 | 86.27% |
| net_income | LYV | 2023-12-31 00:00:00 | 556893000.0 | 316000000.0 | 43.26% |
| net_income | TRGP | 2023-12-31 00:00:00 | 1345900000.0 | 828000000.0 | 38.48% |
| net_income | CRWD | 2025-01-31 00:00:00 | -15241000.0 | -19000000.0 | 24.66% |
| net_income | CRWD | 2024-01-31 00:00:00 | 72181000.0 | 89000000.0 | 23.30% |
| net_income | CAH | 2023-06-30 00:00:00 | 330000000.0 | 261000000.0 | 20.91% |
| net_income | MPWR | 2024-12-31 00:00:00 | 1592058000.0 | 1787000000.0 | 12.24% |
| ebitda | IP | 2025-12-31 00:00:00 | 65000000.0 | 3082000000.0 | 4641.54% |
| ebitda | CVNA | 2025-12-31 00:00:00 | -110000000.0 | 2166000000.0 | 2069.09% |
| ebitda | SATS | 2023-12-31 00:00:00 | -243404000.0 | 1320000000.0 | 642.31% |
| ebitda | ARE | 2025-12-31 00:00:00 | 360450000.0 | 1969000000.0 | 446.26% |
| ebitda | STZ | 2025-02-28 00:00:00 | 781300000.0 | 3609000000.0 | 361.92% |
| ebitda | DD | 2023-12-31 00:00:00 | 697000000.0 | 2800000000.0 | 301.72% |
| ebitda | KHC | 2025-12-31 00:00:00 | -3530000000.0 | 5605000000.0 | 258.78% |
| ebitda | WDC | 2023-06-30 00:00:00 | 289000000.0 | -444000000.0 | 253.63% |
| ebitda | TAP | 2025-12-31 00:00:00 | -1544400000.0 | 2040000000.0 | 232.09% |
| ebitda | XYZ | 2023-12-31 00:00:00 | 790384000.0 | -722000000.0 | 191.35% |
| operating_cash_flow | DD | 2024-12-31 00:00:00 | 765000000.0 | 2321000000.0 | 203.40% |
| operating_cash_flow | DD | 2023-12-31 00:00:00 | 845000000.0 | 2191000000.0 | 159.29% |
| operating_cash_flow | DELL | 2025-01-31 00:00:00 | 4521000000.0 | 4520000.0 | 99.90% |
| operating_cash_flow | JCI | 2025-09-30 00:00:00 | 1399000000.0 | 2554000000.0 | 82.56% |
| operating_cash_flow | SW | 2023-12-31 00:00:00 | 1559000000.0 | 275000000.0 | 82.36% |
| operating_cash_flow | TKO | 2023-12-31 00:00:00 | 266159000.0 | 468380000.0 | 75.98% |
| operating_cash_flow | DOC | 2023-12-31 00:00:00 | 956242000.0 | 273430000.0 | 71.41% |
| operating_cash_flow | DOV | 2024-12-31 00:00:00 | 748379000.0 | 1087830000.0 | 45.36% |
| operating_cash_flow | NWSA | 2023-06-30 00:00:00 | 777000000.0 | 1092000000.0 | 40.54% |
| operating_cash_flow | NWS | 2023-06-30 00:00:00 | 777000000.0 | 1092000000.0 | 40.54% |
| total_assets | COIN | 2023-12-31 00:00:00 | 14753901000.0 | 206983000000.0 | 1302.90% |
| total_assets | SW | 2023-12-31 00:00:00 | 14051000000.0 | 26746000000.0 | 90.35% |
| total_assets | HOOD | 2023-12-31 00:00:00 | 17624000000.0 | 32332000000.0 | 83.45% |
| total_assets | DOC | 2023-12-31 00:00:00 | 15698850000.0 | 5156000000.0 | 67.16% |
| total_assets | TKO | 2024-12-31 00:00:00 | 15111782000.0 | 12700000000.0 | 15.96% |
| total_assets | GE | 2023-12-31 00:00:00 | 173300000000.0 | 163045000000.0 | 5.92% |
| total_assets | LII | 2024-12-31 00:00:00 | 3620000000.0 | 3472000000.0 | 4.09% |
| total_assets | PYPL | 2024-12-31 00:00:00 | 78725000000.0 | 81611000000.0 | 3.67% |
| total_assets | XYZ | 2023-12-31 00:00:00 | 33031308000.0 | 34070000000.0 | 3.14% |
| total_assets | MPWR | 2024-12-31 00:00:00 | 3515822000.0 | 3617000000.0 | 2.88% |
| cash | UDR | 2025-12-31 00:00:00 | 1222000.0 | 923000000.0 | 75431.91% |
| cash | UDR | 2024-12-31 00:00:00 | 1326000.0 | 953000000.0 | 71770.29% |
| cash | UDR | 2023-12-31 00:00:00 | 2922000.0 | 988000000.0 | 33712.46% |
| cash | HIG | 2025-12-31 00:00:00 | 133000000.0 | 4530000000.0 | 3306.02% |
| cash | HIG | 2023-12-31 00:00:00 | 126000000.0 | 4039000000.0 | 3105.56% |
| cash | CNP | 2024-12-31 00:00:00 | 24000000.0 | 585000000.0 | 2337.50% |
| cash | HIG | 2024-12-31 00:00:00 | 183000000.0 | 4302000000.0 | 2250.82% |
| cash | PGR | 2023-12-31 00:00:00 | 85000000.0 | 1890000000.0 | 2123.53% |
| cash | IBKR | 2024-12-31 00:00:00 | 3633000000.0 | 80023000000.0 | 2102.67% |
| cash | IBKR | 2023-12-31 00:00:00 | 3753000000.0 | 79318000000.0 | 2013.46% |

## Interpretation

- Revenue/sales and many margin features should be highly correlated if definitions match.
- Debt, capex, FCF, and cash-flow features are expected to differ more because Dolt and Yahoo expose different line-item definitions/sign conventions.
- If price/return fields differ, model comparisons are not apples-to-apples even with matched fundamentals.
- Features with low correlation or large median absolute percentage difference should not be mixed without remapping definitions.
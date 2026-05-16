# DoltHub Temporal Stability Diagnostics

Tests whether older training years hurt fundamentals by comparing expanding, 3-year rolling, and 5-year rolling training windows.

## Overall performance

| observations | test_years | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean | training_mode | feature_set |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2101 | 10 | 6.21% | 25.89% | 25.45% | 50.20% | 34.12% | 79.57% | 10.14% | 0.40058411885290973 | expanding | price_only |
| 2101 | 10 | 5.76% | 26.15% | 24.72% | 46.14% | 31.46% | 76.72% | 9.68% | 0.36464196782950076 | expanding | financial_plus_price |
| 2101 | 10 | 6.20% | 26.87% | 23.90% | 52.10% | 36.54% | 80.76% | 12.13% | 0.399663304395958 | rolling_5y | price_only |
| 2101 | 10 | 5.49% | 27.32% | 23.43% | 48.29% | 34.12% | 77.43% | 9.47% | 0.38821765817886356 | rolling_5y | financial_plus_price |
| 2101 | 10 | 0.64% | 17.15% | 11.80% | 34.76% | 21.29% | 75.06% | 14.36% | 0.203975105429194 | expanding | financial_only |
| 2101 | 10 | 0.79% | 15.90% | 8.96% | 33.81% | 20.22% | 72.92% | 14.80% | 0.19012015340381247 | rolling_5y | financial_only |
| 2101 | 10 | -5.15% | 13.31% | 8.59% | 30.14% | 18.72% | 75.06% | 16.17% | 0.13974365605993935 | rolling_3y | financial_only |
| 2101 | 10 | -3.93% | 16.43% | 6.25% | 32.55% | 21.51% | 71.73% | 20.15% | 0.12396505801889035 | rolling_3y | financial_plus_price |
| 2101 | 10 | -3.48% | 12.91% | 3.21% | 30.18% | 20.69% | 70.78% | 20.76% | 0.09424785904948538 | rolling_3y | price_only |

## Year-by-year performance

| observations | test_years | oos_r2 | pearson_corr | spearman_corr | top_quintile_mean_return | top_quintile_median_return | top_quintile_hit_rate | bottom_quintile_mean_return | top_minus_bottom_mean | training_mode | feature_set | test_year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 201 | 1 | -12.27% | 7.28% | 4.46% | 38.76% | 28.02% | 87.80% | 26.47% | 0.12284610956430347 | expanding | financial_only | 2016 |
| 201 | 1 | -7.36% | 16.55% | 7.49% | 39.50% | 25.16% | 82.93% | 26.98% | 0.1252540940688452 | expanding | financial_plus_price | 2016 |
| 201 | 1 | 1.26% | 29.00% | 19.39% | 47.75% | 30.31% | 90.24% | 19.99% | 0.2775898466916177 | expanding | price_only | 2016 |
| 201 | 1 | -12.27% | 7.28% | 4.46% | 38.76% | 28.02% | 87.80% | 26.47% | 0.12284610956430347 | rolling_3y | financial_only | 2016 |
| 201 | 1 | -7.36% | 16.55% | 7.49% | 39.50% | 25.16% | 82.93% | 26.98% | 0.1252540940688452 | rolling_3y | financial_plus_price | 2016 |
| 201 | 1 | 1.26% | 29.00% | 19.39% | 47.75% | 30.31% | 90.24% | 19.99% | 0.2775898466916177 | rolling_3y | price_only | 2016 |
| 201 | 1 | -12.27% | 7.28% | 4.46% | 38.76% | 28.02% | 87.80% | 26.47% | 0.12284610956430347 | rolling_5y | financial_only | 2016 |
| 201 | 1 | -7.36% | 16.55% | 7.49% | 39.50% | 25.16% | 82.93% | 26.98% | 0.1252540940688452 | rolling_5y | financial_plus_price | 2016 |
| 201 | 1 | 1.26% | 29.00% | 19.39% | 47.75% | 30.31% | 90.24% | 19.99% | 0.2775898466916177 | rolling_5y | price_only | 2016 |
| 204 | 1 | -4.81% | 17.95% | 14.86% | 24.91% | 22.67% | 73.17% | 12.86% | 0.12047698194754303 | expanding | financial_only | 2017 |
| 204 | 1 | 2.42% | 24.47% | 22.39% | 26.97% | 23.02% | 82.93% | 10.23% | 0.1673645471107853 | expanding | financial_plus_price | 2017 |
| 204 | 1 | -2.72% | 14.43% | 14.63% | 23.12% | 21.52% | 75.61% | 11.07% | 0.1205468953609186 | expanding | price_only | 2017 |
| 204 | 1 | -4.81% | 17.95% | 14.86% | 24.91% | 22.67% | 73.17% | 12.86% | 0.12047698194754303 | rolling_3y | financial_only | 2017 |
| 204 | 1 | 2.42% | 24.47% | 22.39% | 26.97% | 23.02% | 82.93% | 10.23% | 0.1673645471107853 | rolling_3y | financial_plus_price | 2017 |
| 204 | 1 | -2.72% | 14.43% | 14.63% | 23.12% | 21.52% | 75.61% | 11.07% | 0.1205468953609186 | rolling_3y | price_only | 2017 |
| 204 | 1 | -4.81% | 17.95% | 14.86% | 24.91% | 22.67% | 73.17% | 12.86% | 0.12047698194754303 | rolling_5y | financial_only | 2017 |
| 204 | 1 | 2.42% | 24.47% | 22.39% | 26.97% | 23.02% | 82.93% | 10.23% | 0.1673645471107853 | rolling_5y | financial_plus_price | 2017 |
| 204 | 1 | -2.72% | 14.43% | 14.63% | 23.12% | 21.52% | 75.61% | 11.07% | 0.1205468953609186 | rolling_5y | price_only | 2017 |
| 207 | 1 | -10.91% | 4.93% | 0.51% | 16.87% | 11.37% | 69.05% | 15.12% | 0.017459774287887697 | expanding | financial_only | 2018 |
| 207 | 1 | -17.16% | 0.71% | -8.85% | 17.07% | 8.23% | 57.14% | 16.74% | 0.0033445891683446527 | expanding | financial_plus_price | 2018 |
| 207 | 1 | -15.46% | -1.33% | -6.96% | 20.34% | 12.94% | 66.67% | 19.48% | 0.008582585536141718 | expanding | price_only | 2018 |
| 207 | 1 | -12.43% | 4.89% | 0.93% | 16.30% | 12.23% | 66.67% | 10.20% | 0.060969104048727824 | rolling_3y | financial_only | 2018 |
| 207 | 1 | -27.27% | 3.49% | -4.63% | 20.39% | 14.19% | 64.29% | 18.23% | 0.021579710205789693 | rolling_3y | financial_plus_price | 2018 |
| 207 | 1 | -25.85% | 1.73% | -4.91% | 23.80% | 22.52% | 71.43% | 19.44% | 0.0436487873036053 | rolling_3y | price_only | 2018 |
| 207 | 1 | -10.91% | 4.93% | 0.51% | 16.87% | 11.37% | 69.05% | 15.12% | 0.017459774287887697 | rolling_5y | financial_only | 2018 |
| 207 | 1 | -17.16% | 0.71% | -8.85% | 17.07% | 8.23% | 57.14% | 16.74% | 0.0033445891683446527 | rolling_5y | financial_plus_price | 2018 |
| 207 | 1 | -15.46% | -1.33% | -6.96% | 20.34% | 12.94% | 66.67% | 19.48% | 0.008582585536141718 | rolling_5y | price_only | 2018 |
| 211 | 1 | -32.50% | 27.55% | 28.09% | 8.07% | 14.04% | 62.79% | -12.81% | 0.20880860393510992 | expanding | financial_only | 2019 |
| 211 | 1 | -13.15% | 35.09% | 35.22% | 9.86% | 14.04% | 60.47% | -22.06% | 0.3191657523068489 | expanding | financial_plus_price | 2019 |
| 211 | 1 | -24.65% | 26.05% | 21.46% | 13.37% | 16.42% | 65.12% | -8.40% | 0.21761465584259843 | expanding | price_only | 2019 |
| 211 | 1 | -67.55% | 20.60% | 24.37% | 8.95% | 12.96% | 60.47% | -12.86% | 0.21803260019338905 | rolling_3y | financial_only | 2019 |
| 211 | 1 | -53.20% | 31.41% | 31.48% | 8.44% | 16.42% | 60.47% | -15.41% | 0.23846537467709272 | rolling_3y | financial_plus_price | 2019 |
| 211 | 1 | -66.02% | 24.24% | 18.25% | 6.40% | 7.01% | 58.14% | -7.32% | 0.1372474547216448 | rolling_3y | price_only | 2019 |
| 211 | 1 | -32.50% | 27.55% | 28.09% | 8.07% | 14.04% | 62.79% | -12.81% | 0.20880860393510992 | rolling_5y | financial_only | 2019 |
| 211 | 1 | -13.15% | 35.09% | 35.22% | 9.86% | 14.04% | 60.47% | -22.06% | 0.3191657523068489 | rolling_5y | financial_plus_price | 2019 |
| 211 | 1 | -24.65% | 26.05% | 21.46% | 13.37% | 16.42% | 65.12% | -8.40% | 0.21761465584259843 | rolling_5y | price_only | 2019 |
| 212 | 1 | -54.64% | 9.88% | 14.73% | 71.43% | 58.80% | 93.02% | 56.30% | 0.151384476129386 | expanding | financial_only | 2020 |
| 212 | 1 | -31.11% | 10.74% | 17.36% | 63.69% | 48.68% | 88.37% | 55.54% | 0.08148132187132606 | expanding | financial_plus_price | 2020 |
| 212 | 1 | -28.63% | 8.52% | 11.35% | 87.50% | 58.80% | 88.37% | 64.16% | 0.23342047865156879 | expanding | price_only | 2020 |
| 212 | 1 | -53.20% | 7.54% | 7.89% | 77.69% | 58.80% | 95.35% | 71.25% | 0.06441049091864648 | rolling_3y | financial_only | 2020 |
| 212 | 1 | -56.27% | 1.16% | 5.26% | 82.84% | 70.25% | 95.35% | 85.41% | -0.025680724657364595 | rolling_3y | financial_plus_price | 2020 |
| 212 | 1 | -54.72% | -11.76% | -17.02% | 65.46% | 45.27% | 90.70% | 80.96% | -0.15502781425374856 | rolling_3y | price_only | 2020 |
| 212 | 1 | -53.69% | 11.33% | 13.06% | 72.96% | 58.95% | 93.02% | 60.84% | 0.12123558404842527 | rolling_5y | financial_only | 2020 |
| 212 | 1 | -27.29% | 9.49% | 14.32% | 64.13% | 49.52% | 88.37% | 47.29% | 0.16841157090258135 | rolling_5y | financial_plus_price | 2020 |
| 212 | 1 | -25.59% | 6.09% | 7.92% | 67.97% | 50.99% | 90.70% | 64.07% | 0.03898779890922399 | rolling_5y | price_only | 2020 |
| 211 | 1 | -4.42% | 3.12% | -5.84% | 26.95% | 15.84% | 65.12% | 18.90% | 0.08049888888080392 | expanding | financial_only | 2021 |
| 211 | 1 | -9.88% | 8.69% | 8.03% | 21.78% | 19.13% | 65.12% | 12.79% | 0.08991646640068704 | expanding | financial_plus_price | 2021 |
| 211 | 1 | 1.45% | 23.91% | 24.51% | 30.86% | 26.13% | 67.44% | 7.84% | 0.2301678774094379 | expanding | price_only | 2021 |
| 211 | 1 | -20.77% | -15.70% | -27.90% | 14.17% | 2.50% | 58.14% | 31.77% | -0.1759778937460645 | rolling_3y | financial_only | 2021 |
| 211 | 1 | -27.74% | -16.95% | -20.14% | 8.64% | 2.65% | 58.14% | 25.05% | -0.16405329093451837 | rolling_3y | financial_plus_price | 2021 |
| 211 | 1 | -19.86% | -15.49% | -6.06% | 15.12% | 14.77% | 72.09% | 27.35% | -0.12224114158150295 | rolling_3y | price_only | 2021 |
| 211 | 1 | -7.14% | 3.23% | -5.02% | 27.94% | 21.28% | 67.44% | 18.24% | 0.09699914912181395 | rolling_5y | financial_only | 2021 |
| 211 | 1 | -20.35% | 0.99% | 0.53% | 16.02% | 12.38% | 65.12% | 17.27% | -0.012468957920328205 | rolling_5y | financial_plus_price | 2021 |
| 211 | 1 | -6.12% | 15.25% | 19.44% | 26.42% | 19.21% | 67.44% | 7.72% | 0.18707901339057842 | rolling_5y | price_only | 2021 |
| 218 | 1 | -65.90% | -9.21% | -3.14% | -3.67% | -2.70% | 38.64% | -0.25% | -0.03417349747542663 | expanding | financial_only | 2022 |
| 218 | 1 | -212.17% | -29.16% | -21.52% | -7.42% | -7.66% | 29.55% | 6.95% | -0.14370660600209684 | expanding | financial_plus_price | 2022 |
| 218 | 1 | -224.26% | -30.91% | -19.41% | -6.31% | -6.78% | 36.36% | 2.12% | -0.08426672459551059 | expanding | price_only | 2022 |
| 218 | 1 | -159.98% | -9.06% | -1.40% | -2.16% | -2.21% | 40.91% | 2.05% | -0.04207334086094408 | rolling_3y | financial_only | 2022 |
| 218 | 1 | -301.54% | -26.11% | -15.35% | -9.44% | -7.72% | 29.55% | 5.51% | -0.14952399330237542 | rolling_3y | financial_plus_price | 2022 |
| 218 | 1 | -300.20% | -26.59% | -15.00% | -7.10% | -7.32% | 36.36% | 4.94% | -0.12041862525369917 | rolling_3y | price_only | 2022 |
| 218 | 1 | -100.85% | -10.29% | -5.02% | -1.06% | -2.21% | 43.18% | 2.13% | -0.03190524069805989 | rolling_5y | financial_only | 2022 |
| 218 | 1 | -250.73% | -30.22% | -21.57% | -8.89% | -7.08% | 29.55% | 5.92% | -0.1480664953210591 | rolling_5y | financial_plus_price | 2022 |
| 218 | 1 | -256.07% | -30.60% | -18.12% | -6.53% | -6.50% | 36.36% | 4.08% | -0.1060352510300758 | rolling_5y | price_only | 2022 |
| 224 | 1 | 0.80% | 34.03% | 16.91% | 83.05% | 48.99% | 95.56% | 31.73% | 0.5132500951415293 | expanding | financial_only | 2023 |
| 224 | 1 | 7.61% | 35.56% | 34.93% | 93.33% | 50.08% | 95.56% | 21.06% | 0.7226467408784816 | expanding | financial_plus_price | 2023 |
| 224 | 1 | 5.85% | 33.48% | 31.97% | 93.05% | 50.08% | 95.56% | 25.70% | 0.6735370475627935 | expanding | price_only | 2023 |
| 224 | 1 | -0.36% | 18.82% | 9.02% | 60.73% | 28.27% | 86.67% | 43.31% | 0.1742418871738109 | rolling_3y | financial_only | 2023 |
| 224 | 1 | 21.43% | 47.15% | 23.16% | 92.45% | 49.35% | 88.89% | 28.64% | 0.6381095379995235 | rolling_3y | financial_plus_price | 2023 |
| 224 | 1 | 20.37% | 50.62% | 24.96% | 85.61% | 48.70% | 88.89% | 26.50% | 0.5910827291202767 | rolling_3y | price_only | 2023 |
| 224 | 1 | 4.15% | 31.97% | 10.29% | 77.24% | 38.97% | 93.33% | 30.59% | 0.46646264153662725 | rolling_5y | financial_only | 2023 |
| 224 | 1 | 12.31% | 41.51% | 29.79% | 91.85% | 50.08% | 93.33% | 24.69% | 0.6716344925072458 | rolling_5y | financial_plus_price | 2023 |
| 224 | 1 | 9.61% | 40.72% | 29.10% | 90.79% | 48.99% | 95.56% | 27.96% | 0.6282572196577344 | rolling_5y | price_only | 2023 |
| 224 | 1 | 2.68% | 17.89% | -3.52% | 28.04% | 11.78% | 57.78% | 17.28% | 0.10767476276015028 | expanding | financial_only | 2024 |
| 224 | 1 | 0.57% | 19.78% | -3.23% | 37.28% | 13.67% | 62.22% | 19.47% | 0.178162993865046 | expanding | financial_plus_price | 2024 |
| 224 | 1 | -0.42% | 16.79% | 2.86% | 34.39% | 12.22% | 62.22% | 17.05% | 0.17338168438206863 | expanding | price_only | 2024 |
| 224 | 1 | -9.24% | 4.36% | -14.55% | 21.34% | 11.78% | 64.44% | 25.88% | -0.04530817974299031 | rolling_3y | financial_only | 2024 |
| 224 | 1 | -10.54% | -0.46% | -8.31% | 22.54% | 10.13% | 68.89% | 24.95% | -0.024153548230857774 | rolling_3y | financial_plus_price | 2024 |
| 224 | 1 | -8.66% | -0.11% | -2.54% | 25.28% | 16.66% | 73.33% | 23.70% | 0.015732102269421594 | rolling_3y | price_only | 2024 |
| 224 | 1 | -0.28% | 10.83% | -13.09% | 22.50% | 5.35% | 55.56% | 19.56% | 0.02939817258117558 | rolling_5y | financial_only | 2024 |
| 224 | 1 | -7.44% | 18.98% | -6.34% | 36.13% | 11.78% | 57.78% | 22.41% | 0.1372064949570668 | rolling_5y | financial_plus_price | 2024 |
| 224 | 1 | -5.53% | 18.98% | 2.39% | 38.87% | 16.72% | 66.67% | 17.41% | 0.21460323984613514 | rolling_5y | price_only | 2024 |
| 189 | 1 | -4.10% | 6.95% | 5.84% | 35.27% | 22.00% | 81.58% | 32.14% | 0.03133723673251554 | expanding | financial_only | 2025 |
| 189 | 1 | 13.39% | 42.81% | 36.98% | 71.29% | 46.83% | 92.11% | 10.17% | 0.6112151654682337 | expanding | financial_plus_price | 2025 |
| 189 | 1 | 16.23% | 46.13% | 40.52% | 71.75% | 46.74% | 92.11% | 11.25% | 0.6050108171845214 | expanding | price_only | 2025 |
| 189 | 1 | -6.50% | 10.45% | 9.95% | 41.82% | 21.14% | 81.58% | 24.65% | 0.1717411763722001 | rolling_3y | financial_only | 2025 |
| 189 | 1 | -6.64% | 27.35% | 21.30% | 46.03% | 35.80% | 81.58% | 16.26% | 0.2976580217023498 | rolling_3y | financial_plus_price | 2025 |
| 189 | 1 | -6.92% | 23.03% | 15.90% | 50.52% | 35.80% | 81.58% | 19.15% | 0.31368081186113983 | rolling_3y | price_only | 2025 |
| 189 | 1 | -0.18% | 11.99% | 14.72% | 43.44% | 30.25% | 86.84% | 20.33% | 0.2310466571048178 | rolling_5y | financial_only | 2025 |
| 189 | 1 | 13.21% | 45.15% | 35.73% | 72.58% | 46.83% | 94.74% | 14.84% | 0.5773857062584717 | rolling_5y | financial_plus_price | 2025 |
| 189 | 1 | 16.86% | 48.23% | 40.08% | 76.11% | 48.37% | 94.74% | 4.98% | 0.7113313913332949 | rolling_5y | price_only | 2025 |

## Coefficient stability

| training_mode | feature_set | feature | mean_coef | median_coef | sign_consistency | folds |
| --- | --- | --- | --- | --- | --- | --- |
| expanding | financial_only | capex_to_revenue | -2.03% | -1.82% | 100.00% | 10 |
| expanding | financial_only | cash_to_assets | 4.73% | 4.79% | 100.00% | 10 |
| expanding | financial_only | debt_to_assets | 1.72% | 1.65% | 100.00% | 10 |
| expanding | financial_only | ebitda_margin | 3.54% | 3.58% | 100.00% | 10 |
| expanding | financial_only | net_margin | -2.46% | -2.66% | 100.00% | 10 |
| expanding | financial_only | operating_margin | -3.87% | -3.85% | 100.00% | 10 |
| expanding | financial_only | revenue_growth_accel | -4.18% | -3.83% | 100.00% | 6 |
| expanding | financial_only | revenue_growth_yoy | -1.44% | -1.43% | 80.00% | 10 |
| expanding | financial_only | cfo_margin | -0.62% | -1.70% | 70.00% | 10 |
| expanding | financial_only | current_ratio | -0.49% | -0.76% | 70.00% | 10 |
| expanding | financial_only | fcf_margin | 0.07% | 1.19% | 70.00% | 10 |
| expanding | financial_only | gross_margin | 0.96% | 0.27% | 60.00% | 10 |
| expanding | financial_only | asset_turnover | 0.33% | 0.07% | 50.00% | 10 |
| expanding | financial_only | debt_to_equity | -0.42% | -0.03% | 50.00% | 10 |
| rolling_3y | financial_only | debt_to_assets | 2.01% | 1.48% | 90.00% | 10 |
| rolling_3y | financial_only | revenue_growth_accel | -7.70% | -6.09% | 83.33% | 6 |
| rolling_3y | financial_only | capex_to_revenue | -0.98% | -1.22% | 80.00% | 10 |
| rolling_3y | financial_only | net_margin | -4.02% | -4.31% | 80.00% | 10 |
| rolling_3y | financial_only | operating_margin | -1.65% | -1.99% | 80.00% | 10 |
| rolling_3y | financial_only | cash_to_assets | 2.39% | 4.02% | 70.00% | 10 |
| rolling_3y | financial_only | gross_margin | 1.53% | 1.73% | 70.00% | 10 |
| rolling_3y | financial_only | asset_turnover | 0.30% | 1.43% | 60.00% | 10 |
| rolling_3y | financial_only | ebitda_margin | 0.04% | 1.55% | 60.00% | 10 |
| rolling_3y | financial_only | current_ratio | 0.97% | -0.05% | 50.00% | 10 |
| rolling_3y | financial_only | debt_to_equity | -0.23% | -0.48% | 50.00% | 10 |
| rolling_3y | financial_only | fcf_margin | -0.52% | 1.07% | 40.00% | 10 |
| rolling_3y | financial_only | revenue_growth_yoy | 0.68% | -1.43% | 40.00% | 10 |
| rolling_3y | financial_only | cfo_margin | 0.50% | -1.66% | 20.00% | 10 |
| rolling_5y | financial_only | debt_to_assets | 1.95% | 2.09% | 100.00% | 10 |
| rolling_5y | financial_only | revenue_growth_accel | -5.19% | -3.83% | 100.00% | 6 |
| rolling_5y | financial_only | net_margin | -3.47% | -2.66% | 90.00% | 10 |
| rolling_5y | financial_only | cash_to_assets | 3.34% | 3.78% | 80.00% | 10 |
| rolling_5y | financial_only | operating_margin | -2.55% | -3.41% | 80.00% | 10 |
| rolling_5y | financial_only | revenue_growth_yoy | -1.31% | -1.43% | 80.00% | 10 |
| rolling_5y | financial_only | capex_to_revenue | -1.02% | -1.10% | 70.00% | 10 |
| rolling_5y | financial_only | gross_margin | 1.53% | 1.08% | 70.00% | 10 |
| rolling_5y | financial_only | debt_to_equity | -0.60% | -0.38% | 60.00% | 10 |
| rolling_5y | financial_only | ebitda_margin | 0.99% | 3.15% | 60.00% | 10 |
| rolling_5y | financial_only | asset_turnover | 0.53% | 0.25% | 50.00% | 10 |
| rolling_5y | financial_only | current_ratio | 0.40% | -0.68% | 40.00% | 10 |
| rolling_5y | financial_only | cfo_margin | 1.08% | -1.66% | 30.00% | 10 |
| rolling_5y | financial_only | fcf_margin | -1.05% | 1.11% | 30.00% | 10 |
| expanding | financial_plus_price | capex_to_revenue | -2.55% | -2.30% | 100.00% | 10 |
| expanding | financial_plus_price | cash_to_assets | 4.45% | 4.87% | 100.00% | 10 |
| expanding | financial_plus_price | debt_to_assets | 1.74% | 1.85% | 100.00% | 10 |
| expanding | financial_plus_price | revenue_growth_accel | -2.59% | -2.14% | 100.00% | 6 |
| expanding | financial_plus_price | trailing_6m_return | 13.03% | 12.85% | 100.00% | 10 |
| expanding | financial_plus_price | debt_to_equity | -0.86% | -0.55% | 90.00% | 10 |
| expanding | financial_plus_price | operating_margin | -2.25% | -2.06% | 90.00% | 10 |
| expanding | financial_plus_price | trailing_12m_return | -3.87% | -4.49% | 90.00% | 10 |
| expanding | financial_plus_price | drawdown_12m | -1.89% | -1.83% | 80.00% | 10 |
| expanding | financial_plus_price | ebitda_margin | 2.23% | 2.43% | 80.00% | 10 |
| expanding | financial_plus_price | realized_vol_3m | 9.45% | 7.91% | 80.00% | 10 |
| expanding | financial_plus_price | trailing_3m_return | -4.95% | -3.22% | 80.00% | 10 |
| expanding | financial_plus_price | asset_turnover | 0.59% | 0.85% | 70.00% | 10 |
| expanding | financial_plus_price | cfo_margin | -0.61% | -1.82% | 70.00% | 10 |
| expanding | financial_plus_price | current_ratio | -1.04% | -1.37% | 70.00% | 10 |
| expanding | financial_plus_price | relative_6m_vs_spy | -3.26% | -3.19% | 70.00% | 10 |
| expanding | financial_plus_price | revenue_growth_yoy | -1.29% | -0.93% | 70.00% | 10 |
| expanding | financial_plus_price | fcf_margin | 0.07% | 0.48% | 60.00% | 10 |
| expanding | financial_plus_price | growth_x_fcf_margin | -0.80% | -0.60% | 60.00% | 10 |
| expanding | financial_plus_price | growth_x_trailing_6m | -0.41% | -0.56% | 60.00% | 10 |
| expanding | financial_plus_price | net_margin | -0.63% | -0.23% | 50.00% | 10 |
| expanding | financial_plus_price | gross_margin | 0.30% | -0.19% | 40.00% | 10 |
| rolling_3y | financial_plus_price | debt_to_assets | 1.99% | 1.43% | 90.00% | 10 |
| rolling_3y | financial_plus_price | revenue_growth_accel | -5.09% | -3.72% | 83.33% | 6 |
| rolling_3y | financial_plus_price | trailing_6m_return | 9.74% | 7.58% | 80.00% | 10 |
| rolling_3y | financial_plus_price | asset_turnover | 0.73% | 1.45% | 70.00% | 10 |
| rolling_3y | financial_plus_price | capex_to_revenue | -1.46% | -1.33% | 70.00% | 10 |
| rolling_3y | financial_plus_price | cash_to_assets | 2.31% | 4.12% | 70.00% | 10 |
| rolling_3y | financial_plus_price | drawdown_12m | -5.46% | -4.73% | 70.00% | 10 |
| rolling_3y | financial_plus_price | fcf_margin | 0.32% | 1.06% | 70.00% | 10 |
| rolling_3y | financial_plus_price | gross_margin | 0.78% | 0.83% | 70.00% | 10 |
| rolling_3y | financial_plus_price | trailing_12m_return | -2.16% | -1.30% | 70.00% | 10 |
| rolling_3y | financial_plus_price | debt_to_equity | -0.28% | -0.82% | 60.00% | 10 |
| rolling_3y | financial_plus_price | realized_vol_3m | 6.92% | 6.81% | 60.00% | 10 |
| rolling_3y | financial_plus_price | current_ratio | 0.31% | -0.07% | 50.00% | 10 |
| rolling_3y | financial_plus_price | growth_x_fcf_margin | -0.86% | -0.54% | 50.00% | 10 |
| rolling_3y | financial_plus_price | growth_x_trailing_6m | 0.19% | -0.59% | 50.00% | 10 |
| rolling_3y | financial_plus_price | net_margin | -0.28% | 0.21% | 50.00% | 10 |

## Interpretation

- If rolling windows beat expanding windows, the signal is non-stationary and older years may hurt.
- If financial-only coefficients have poor sign consistency, fundamentals are unstable as direct predictors.
- If price-only is stable while financial-plus-price is not, fundamentals are better as filters than direct model inputs.
# Deployable Multi-Strategy Portfolio Research

This report converts the exploratory strategy results into a conservative deployment candidate.
It is still a research result, not investment advice or a production trading approval.

## Selected sleeves

| sleeve | selected_weight | annual_return | annual_std | sharpe | max_drawdown |
| --- | --- | --- | --- | --- | --- |
| large_cap_rs_126d | 38.21% | 27.24% | 20.09% | 1.36 | -34.17% |
| large_cap_bear_short_weak | 22.20% | -9.85% | 15.98% | -0.62 | -75.85% |
| ath_dip_recovery | 20.31% | 15.98% | 16.08% | 0.99 | -34.53% |
| large_cap_mom_12_1 | 17.08% | 23.12% | 20.33% | 1.14 | -36.81% |
| ma200_wider_stop | 1.66% | 10.32% | 19.75% | 0.52 | -36.70% |
| sector_defensive_rotation | 0.54% | 0.61% | 12.70% | 0.05 | -44.64% |

## Portfolio comparison

| portfolio | annual_return | annual_std | sharpe | max_drawdown | hit_rate |
| --- | --- | --- | --- | --- | --- |
| deployable_guarded | 15.30% | 11.68% | 1.31 | -20.43% | 57.51% |
| aggressive_vol_target_16 | 17.78% | 13.68% | 1.30 | -23.10% | 57.51% |
| optimized_vol_target_12 | 15.68% | 12.11% | 1.29 | -19.09% | 57.51% |
| optimized_core | 15.70% | 12.42% | 1.26 | -24.91% | 57.51% |
| regime_switched_guarded | 12.93% | 12.17% | 1.06 | -21.87% | 56.34% |
| equal_weight_core | 11.47% | 11.67% | 0.98 | -24.12% | 56.28% |
| regime_switched_core | 13.87% | 14.53% | 0.95 | -27.50% | 56.34% |

## 2022-2026 holdout check

| portfolio | annual_return | annual_std | sharpe | max_drawdown | hit_rate |
| --- | --- | --- | --- | --- | --- |
| optimized_core | 12.47% | 11.78% | 1.06 | -18.50% | 54.79% |
| optimized_vol_target_12 | 12.79% | 12.19% | 1.05 | -18.50% | 54.79% |
| aggressive_vol_target_16 | 14.46% | 14.04% | 1.03 | -23.10% | 54.79% |
| deployable_guarded | 11.84% | 11.77% | 1.01 | -19.85% | 54.79% |
| regime_switched_guarded | 8.79% | 11.93% | 0.74 | -20.87% | 54.24% |
| equal_weight_core | 8.07% | 11.12% | 0.73 | -20.33% | 54.33% |
| regime_switched_core | 8.90% | 14.24% | 0.63 | -26.66% | 54.24% |

## Estimated next-year return distribution

| portfolio | expected_return | median_return | p05_return | p25_return | p75_return | p95_return | loss_probability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aggressive_vol_target_16 | 15.16% | 14.70% | -10.84% | 4.16% | 25.61% | 42.55% | 17.57% |
| optimized_core | 14.06% | 13.96% | -10.38% | 4.09% | 24.08% | 38.53% | 17.04% |
| optimized_vol_target_12 | 12.80% | 12.29% | -9.51% | 3.36% | 21.93% | 36.49% | 17.63% |
| deployable_guarded | 12.37% | 11.87% | -8.80% | 3.09% | 21.09% | 34.95% | 17.70% |
| regime_switched_core | 11.13% | 10.16% | -15.32% | -0.71% | 21.92% | 41.17% | 26.52% |
| equal_weight_core | 10.11% | 10.23% | -12.70% | 1.11% | 19.25% | 32.76% | 22.50% |
| regime_switched_guarded | 8.68% | 7.88% | -13.02% | -0.99% | 17.65% | 32.70% | 27.69% |

## Recommended deployment candidate

`deployable_guarded` is the preferred candidate because it keeps most of the optimized core's Sharpe while reducing realized volatility with a causal 12% vol target and a trailing-drawdown brake.

- Full-sample annual return: 15.30%.
- Full-sample annual std: 11.68%.
- Full-sample Sharpe: 1.31.
- Full-sample max drawdown: -20.43%.
- Holdout annual return: 11.84%.
- Holdout Sharpe: 1.01.
- Estimated next-year mean return: 12.37%.
- Estimated next-year 5th/95th percentile: -8.80% / 34.95%.
- Estimated probability of a negative next year: 17.7%.

## Safety controls before live use

- Trade liquid large-cap/ETF sleeves first; keep lower-cap and hourly sleeves out of the production portfolio until validated on a better intraday data source.
- Enforce max gross exposure, max single-sleeve weight, borrow availability for short sleeves, and daily loss limits.
- Recompute signals after market close; execute with limit/VWAP-aware orders rather than assuming close-to-close fills.
- Re-run this report on survivorship-free data before sizing real capital.

Training objective selected weights on 2013-2021 data: 1.420.
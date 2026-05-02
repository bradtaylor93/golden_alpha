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
| max_return_35_guarded | 47.51% | 33.20% | 1.43 | -41.69% | 57.09% |
| max_return_35_target | 50.07% | 35.45% | 1.41 | -43.69% | 57.09% |
| growth_30_target | 44.12% | 31.33% | 1.41 | -38.53% | 57.09% |
| growth_30_guarded | 41.79% | 29.71% | 1.41 | -35.94% | 57.09% |
| max_return_plus_asset_overlay_guarded | 47.86% | 34.22% | 1.40 | -34.96% | 57.30% |
| max_return_plus_asset_overlay | 50.42% | 36.36% | 1.39 | -44.02% | 57.30% |
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
| max_return_plus_asset_overlay | 51.20% | 36.20% | 1.41 | -36.40% | 55.80% |
| max_return_plus_asset_overlay_guarded | 45.86% | 34.17% | 1.34 | -29.75% | 55.80% |
| growth_30_target | 35.47% | 31.23% | 1.14 | -34.92% | 54.33% |
| max_return_35_target | 40.70% | 36.07% | 1.13 | -39.59% | 54.33% |
| max_return_35_guarded | 37.50% | 33.65% | 1.11 | -40.55% | 54.33% |
| growth_30_guarded | 32.64% | 29.51% | 1.11 | -34.83% | 54.33% |
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
| max_return_plus_asset_overlay | 52.52% | 45.36% | -22.66% | 13.24% | 83.21% | 151.75% | 15.58% |
| max_return_plus_asset_overlay_guarded | 48.39% | 40.70% | -20.29% | 11.88% | 76.37% | 143.15% | 15.86% |
| max_return_35_target | 47.99% | 40.98% | -23.34% | 11.03% | 76.76% | 143.54% | 16.75% |
| max_return_35_guarded | 44.58% | 37.39% | -22.29% | 9.58% | 71.35% | 135.73% | 17.29% |
| growth_30_target | 40.08% | 34.84% | -21.02% | 8.83% | 65.03% | 119.42% | 17.15% |
| growth_30_guarded | 37.41% | 31.89% | -19.99% | 7.93% | 60.75% | 112.80% | 17.39% |
| aggressive_vol_target_16 | 15.16% | 14.70% | -10.84% | 4.16% | 25.61% | 42.55% | 17.57% |
| optimized_core | 14.06% | 13.96% | -10.38% | 4.09% | 24.08% | 38.53% | 17.04% |
| optimized_vol_target_12 | 12.80% | 12.29% | -9.51% | 3.36% | 21.93% | 36.49% | 17.63% |
| deployable_guarded | 12.37% | 11.87% | -8.80% | 3.09% | 21.09% | 34.95% | 17.70% |
| regime_switched_core | 11.13% | 10.16% | -15.32% | -0.71% | 21.92% | 41.17% | 26.52% |
| equal_weight_core | 10.11% | 10.23% | -12.70% | 1.11% | 19.25% | 32.76% | 22.50% |
| regime_switched_guarded | 8.68% | 7.88% | -13.02% | -0.99% | 17.65% | 32.70% | 27.69% |

## High-return deployment candidate

`max_return_plus_asset_overlay` is now the high-return candidate. It keeps the 70% large-cap relative strength / 20% large-cap 12-1 momentum / 10% ATH dip recovery core, overlays 25% asset-class absolute momentum, and re-targets the combined sleeve to 35% annualized volatility.

- Full-sample annual return: 50.42%.
- Full-sample annual std: 36.36%.
- Full-sample Sharpe: 1.39.
- Full-sample max drawdown: -44.02%.
- Holdout annual return: 51.20%.
- Holdout Sharpe: 1.41.
- Estimated next-year mean return: 52.52%.
- Estimated next-year 5th/95th percentile: -22.66% / 151.75%.
- Estimated probability of a negative next year: 15.6%.

`max_return_plus_asset_overlay_guarded` is a safer high-return variant using a trailing drawdown brake. It reduces annual return to 47.86% and max drawdown to -34.96%.

`max_return_35_target` remains the stock-only high-return baseline: 50.07% annual return, 1.41 Sharpe, and -43.69% max drawdown.

`deployable_guarded` remains the conservative version: 15.30% annual return, 1.31 Sharpe, and -20.43% max drawdown.

## Safety controls before live use

- Treat `max_return_plus_asset_overlay` as aggressive: it targets much higher return by accepting 35%+ annualized volatility and 40% drawdown risk.
- Trade liquid large-cap/ETF sleeves first; keep lower-cap and hourly sleeves out of the production portfolio until validated on a better intraday data source.
- Enforce max gross exposure, max single-sleeve weight, borrow availability for short sleeves, and daily loss limits.
- Recompute signals after market close; execute with limit/VWAP-aware orders rather than assuming close-to-close fills.
- Re-run this report on survivorship-free data before sizing real capital.

Training objective selected weights on 2013-2021 data: 1.420.
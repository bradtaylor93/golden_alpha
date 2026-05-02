# Alternative Direction Results

Downloaded 35 usable ETFs from an intended alternative universe of 35.
Strategy families include cross-asset trend, defensive credit/stress rotation, sector mean reversion, inverse equity, and volatility ETF regimes.
Candidates and combinations are selected using pre-2022 data only; 2022-2026 is validation.

## Top pre-2022 alternative candidates

| strategy | train_annual_return | train_sharpe | train_max_drawdown | validation_annual_return | validation_sharpe | validation_max_drawdown | full_annual_return | full_sharpe | full_max_drawdown | train_objective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| credit_stress_rotation | 5.63% | 0.57 | -17.52% | -3.57% | -0.37 | -25.81% | 3.12% | 0.32 | -30.06% | 0.41 |
| xasset_abs_mom_252_top3 | 2.57% | 0.22 | -30.46% | 12.41% | 0.93 | -20.63% | 5.09% | 0.42 | -30.46% | -0.08 |
| etf_short_term_reversal | 0.91% | 0.08 | -31.83% | -6.65% | -0.43 | -28.73% | -1.14% | -0.09 | -31.83% | -0.24 |
| xasset_abs_mom_126_top2 | 1.23% | 0.09 | -36.82% | 13.25% | 0.87 | -15.45% | 4.28% | 0.31 | -36.82% | -0.27 |
| xasset_carry_proxy_defensive | -1.88% | -0.32 | -31.42% | -0.57% | -0.09 | -13.05% | -1.54% | -0.25 | -37.18% | -0.64 |
| vol_regime_svxy_vixy | 4.02% | 0.10 | -76.49% | 0.74% | 0.03 | -31.66% | 3.14% | 0.09 | -76.49% | -0.65 |
| risk_off_inverse_equity | -4.91% | -0.46 | -48.03% | -8.81% | -0.75 | -34.60% | -5.96% | -0.54 | -65.07% | -0.95 |
| sector_market_neutral_trend | -5.68% | -0.75 | -51.81% | -6.84% | -0.69 | -42.33% | -5.99% | -0.72 | -64.17% | -1.28 |
| sector_mean_reversion_pairs | -8.68% | -1.19 | -66.49% | -12.05% | -1.41 | -43.98% | -9.59% | -1.26 | -80.81% | -1.88 |

## 2022-2026 validation for selected candidates

| strategy | validation_observations | validation_total_return | validation_annual_return | validation_annual_std | validation_sharpe | validation_average_daily_return | validation_average_daily_turnover | validation_max_drawdown | validation_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| credit_stress_rotation | 1086 | -14.51% | -3.57% | 0.09544902062105919 | -0.37 | -0.01% | 0.0 | -25.81% | 0.4953959484346225 |
| xasset_abs_mom_252_top3 | 1086 | 65.55% | 12.41% | 0.13291043206079137 | 0.93 | 0.05% | 0.0 | -20.63% | 0.5478821362799263 |
| etf_short_term_reversal | 1086 | -25.65% | -6.65% | 0.1541677689467231 | -0.43 | -0.02% | 0.0 | -28.73% | 0.20534069981583794 |

## Combination tests with current return/Sharpe candidate

| combo | full_annual_return | full_sharpe | full_max_drawdown | validation_annual_return | validation_sharpe | validation_max_drawdown |
| --- | --- | --- | --- | --- | --- | --- |
| asset_overlay_regime_70_xasset_abs_mom_252_top3_30 | 44.73% | 1.54 | -27.46% | 38.13% | 1.36 | -24.36% |
| asset_overlay_regime_80_xasset_abs_mom_252_top3_20 | 49.87% | 1.56 | -29.32% | 41.73% | 1.35 | -26.51% |
| asset_overlay_regime_90_xasset_abs_mom_252_top3_10 | 55.02% | 1.58 | -31.14% | 45.29% | 1.35 | -28.62% |
| asset_overlay_regime_90_credit_stress_rotation_10 | 54.07% | 1.58 | -30.26% | 43.26% | 1.30 | -28.12% |
| asset_overlay_regime_90_etf_short_term_reversal_10 | 53.46% | 1.54 | -30.87% | 42.74% | 1.27 | -28.54% |
| asset_overlay_regime_80_credit_stress_rotation_20 | 47.97% | 1.57 | -27.51% | 37.77% | 1.27 | -26.67% |
| asset_overlay_regime_70_credit_stress_rotation_30 | 41.90% | 1.55 | -26.06% | 32.32% | 1.22 | -25.86% |
| asset_overlay_regime_80_etf_short_term_reversal_20 | 46.84% | 1.48 | -28.76% | 36.77% | 1.20 | -26.92% |
| asset_overlay_regime_70_etf_short_term_reversal_30 | 40.33% | 1.41 | -26.60% | 30.90% | 1.11 | -25.28% |

## Conclusion

- This different direction did not produce a massive validated improvement.
- Cross-asset and defensive sleeves can reduce equity specificity, but validation Sharpe was not better than the current fixed candidates.
- The useful takeaway is to keep asset-class absolute momentum as a modest overlay, not to replace the current return engine.
# Sharpe Improvement Results

This report optimizes risk-adjusted performance rather than raw return.

## Full-sample comparison

| portfolio | annual_return | annual_std | sharpe | max_drawdown |
| --- | --- | --- | --- | --- |
| sharpe_guarded_ml_growth_25 | 42.03% | 25.70% | 1.64 | -22.72% |
| asset_overlay_regime_target35 | 60.20% | 37.77% | 1.59 | -32.93% |
| ml_balanced_target30_brake | 47.18% | 29.84% | 1.58 | -30.45% |
| ml_growth_expanded12_1_target30_brake | 47.47% | 30.09% | 1.58 | -28.31% |
| asset_overlay_target35 | 58.94% | 37.96% | 1.55 | -36.79% |
| ml_meta_scale_balanced | 56.85% | 38.07% | 1.49 | -46.15% |
| ml_meta_scale_growth | 59.54% | 40.10% | 1.48 | -53.62% |
| base_asset_overlay | 53.67% | 36.82% | 1.46 | -44.02% |

## 2022-2026 validation window

| portfolio | annual_return | annual_std | sharpe | max_drawdown |
| --- | --- | --- | --- | --- |
| base_asset_overlay | 51.20% | 36.20% | 1.41 | -36.40% |
| ml_meta_scale_balanced | 52.82% | 37.70% | 1.40 | -39.66% |
| ml_meta_scale_growth | 54.21% | 39.55% | 1.37 | -42.79% |
| asset_overlay_target35 | 50.92% | 37.42% | 1.36 | -35.52% |
| asset_overlay_regime_target35 | 48.80% | 36.51% | 1.34 | -30.69% |
| ml_balanced_target30_brake | 37.09% | 28.73% | 1.29 | -30.45% |
| sharpe_guarded_ml_growth_25 | 30.94% | 24.46% | 1.26 | -22.36% |
| ml_growth_expanded12_1_target30_brake | 34.87% | 28.63% | 1.22 | -28.17% |

## Estimated next-year distribution

| portfolio | expected_return | median_return | p05_return | p95_return | loss_probability |
| --- | --- | --- | --- | --- | --- |
| asset_overlay_regime_target35 | 59.45% | 49.53% | -19.90% | 172.29% | 14.24% |
| asset_overlay_target35 | 58.06% | 47.80% | -20.42% | 171.28% | 14.78% |
| ml_meta_scale_growth | 54.47% | 45.70% | -25.49% | 162.92% | 16.76% |
| base_asset_overlay | 52.97% | 45.22% | -22.33% | 154.11% | 15.44% |
| ml_meta_scale_balanced | 52.79% | 45.01% | -23.20% | 154.47% | 16.08% |
| ml_balanced_target30_brake | 41.86% | 35.71% | -15.68% | 119.75% | 14.11% |
| ml_growth_expanded12_1_target30_brake | 40.67% | 35.28% | -15.94% | 115.34% | 14.90% |
| sharpe_guarded_ml_growth_25 | 35.84% | 31.96% | -12.31% | 96.37% | 12.87% |

## Interpretation

`sharpe_guarded_ml_growth_25` is the max-Sharpe variant: it targets 25% volatility on the ML growth portfolio and applies a trailing drawdown brake.
The upstream ML portfolio is regenerated with a 21-trading-day purge before each annual test fold.

- Annual return: 42.03%.
- Annual std: 25.70%.
- Sharpe: 1.64.
- Max drawdown: -22.72%.
- Validation annual return: 30.94%.
- Validation Sharpe: 1.26.

`asset_overlay_target35` is the better return/Sharpe compromise when validation Sharpe matters more than full-sample Sharpe.
It has 58.94% annual return and 1.55 full-sample Sharpe.

The Sharpe improvement comes from lowering target volatility and cutting exposure after trailing drawdowns; it is not a new alpha source.
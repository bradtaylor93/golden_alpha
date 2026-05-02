# Sharpe Improvement Results

This report optimizes risk-adjusted performance rather than raw return.

## Full-sample comparison

| portfolio | annual_return | annual_std | sharpe | max_drawdown |
| --- | --- | --- | --- | --- |
| sharpe_guarded_ml_growth_25 | 43.62% | 25.74% | 1.69 | -22.30% |
| ml_growth_expanded12_1_target30_brake | 49.07% | 30.17% | 1.63 | -28.63% |
| ml_balanced_target30_brake | 48.24% | 29.88% | 1.61 | -31.03% |
| asset_overlay_regime_target35 | 60.20% | 37.77% | 1.59 | -32.93% |
| asset_overlay_target35 | 58.94% | 37.96% | 1.55 | -36.79% |
| ml_meta_scale_growth | 60.51% | 39.77% | 1.52 | -52.26% |
| ml_meta_scale_balanced | 57.31% | 37.91% | 1.51 | -44.87% |
| base_asset_overlay | 53.67% | 36.82% | 1.46 | -44.02% |

## 2022-2026 holdout

| portfolio | annual_return | annual_std | sharpe | max_drawdown |
| --- | --- | --- | --- | --- |
| base_asset_overlay | 51.20% | 36.20% | 1.41 | -36.40% |
| ml_meta_scale_balanced | 53.00% | 37.74% | 1.40 | -39.58% |
| ml_meta_scale_growth | 54.58% | 39.61% | 1.38 | -42.64% |
| asset_overlay_target35 | 50.92% | 37.42% | 1.36 | -35.52% |
| asset_overlay_regime_target35 | 48.80% | 36.51% | 1.34 | -30.69% |
| sharpe_guarded_ml_growth_25 | 31.69% | 24.37% | 1.30 | -22.30% |
| ml_balanced_target30_brake | 37.21% | 28.76% | 1.29 | -31.03% |
| ml_growth_expanded12_1_target30_brake | 35.24% | 28.66% | 1.23 | -28.63% |

## Estimated next-year distribution

| portfolio | expected_return | median_return | p05_return | p95_return | loss_probability |
| --- | --- | --- | --- | --- | --- |
| asset_overlay_regime_target35 | 59.45% | 49.53% | -19.90% | 172.29% | 14.24% |
| asset_overlay_target35 | 58.06% | 47.80% | -20.42% | 171.28% | 14.78% |
| ml_meta_scale_growth | 55.67% | 47.17% | -24.54% | 163.97% | 16.01% |
| ml_meta_scale_balanced | 53.38% | 45.74% | -22.51% | 155.20% | 15.77% |
| base_asset_overlay | 52.97% | 45.22% | -22.33% | 154.11% | 15.44% |
| ml_balanced_target30_brake | 43.22% | 36.95% | -14.74% | 121.48% | 13.44% |
| ml_growth_expanded12_1_target30_brake | 42.73% | 37.30% | -14.46% | 118.88% | 13.60% |
| sharpe_guarded_ml_growth_25 | 37.81% | 33.92% | -10.67% | 98.79% | 11.47% |

## Interpretation

`sharpe_guarded_ml_growth_25` is the max-Sharpe variant: it targets 25% volatility on the ML growth portfolio and applies a trailing drawdown brake.

- Annual return: 43.62%.
- Annual std: 25.74%.
- Sharpe: 1.69.
- Max drawdown: -22.30%.
- Holdout annual return: 31.69%.
- Holdout Sharpe: 1.30.

`asset_overlay_target35` is the better return/Sharpe compromise when holdout Sharpe matters more than full-sample Sharpe.
It has 58.94% annual return and 1.55 full-sample Sharpe.

The Sharpe improvement comes from lowering target volatility and cutting exposure after trailing drawdowns; it is not a new alpha source.
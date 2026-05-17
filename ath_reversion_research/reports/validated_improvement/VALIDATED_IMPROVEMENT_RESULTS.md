# Validated Improvement Results

This report attempts to improve the strategy using only choices available before 2022.
The 2022-2026 period is used only as validation for pre-selected rules.

## Pre-2022 selected fixed candidates

| portfolio | train_annual_return | train_sharpe | train_max_drawdown | max_sharpe_objective | return_sharpe_objective | selection_rule |
| --- | --- | --- | --- | --- | --- | --- |
| sharpe_guarded_ml_growth_25 | 52.34% | 1.96 | -22.72% | 1.84 | 78.61% | pre2022_max_sharpe_with_drawdown_penalty |
| asset_overlay_regime_target35 | 70.73% | 1.82 | -32.93% | 1.63 | 83.35% | pre2022_return_sharpe_with_drawdown_penalty |

## Validation metrics for pre-selected fixed candidates

| selection_rule | portfolio | full_annual_return | full_sharpe | full_max_drawdown | validation_annual_return | validation_sharpe | validation_max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pre2022_max_sharpe_with_drawdown_penalty | sharpe_guarded_ml_growth_25 | 42.03% | 1.64 | -22.72% | 30.94% | 1.26 | -22.36% |
| pre2022_return_sharpe_with_drawdown_penalty | asset_overlay_regime_target35 | 60.20% | 1.59 | -32.93% | 48.80% | 1.34 | -30.69% |

## Best causal monthly allocator selected on pre-2022 data

| lookback | top_n | target_vol | drawdown_penalty | train_objective | train_annual_return | train_sharpe | train_max_drawdown | validation_annual_return | validation_sharpe | validation_max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 63 | 5 | 20.00% | 100.00% | 0.81 | 24.83% | 1.05 | -28.97% | 24.19% | 1.12 | -16.80% |

## Conclusion

- No new live-style allocator produced a statistically cleaner massive improvement over the fixed rules.
- The allocator is a sanity check over previously generated candidates, not a new independent alpha source.
- The max-Sharpe rule selected `sharpe_guarded_ml_growth_25`, which remains the most defensible risk-adjusted candidate.
- The return/Sharpe rule selected `asset_overlay_regime_target35`, which remains the better high-return compromise.
- Further claimed improvements require new data or a new untouched validation period, otherwise they are likely data-mined.
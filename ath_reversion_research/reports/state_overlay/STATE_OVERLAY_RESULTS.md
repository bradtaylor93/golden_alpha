# State Overlay Improvement Results

This report extends the prior breadth overlay with a prior-day SPY volatility filter.
All thresholds and parameters are selected on pre-2022 data; 2022-2026 is validation only.

## Selected pre-2022 state overlay

| base | breadth_feature | low_quantile | high_quantile | low_threshold | high_threshold | vol_feature | vol_quantile | vol_threshold | low_scale | high_scale | vol_scale | target_vol | train_objective | train_annual_return | train_sharpe | train_max_drawdown | validation_annual_return | validation_sharpe | validation_max_drawdown | full_annual_return | full_sharpe | full_max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharpe_guarded_ml_growth_25 | adv_20 | 30.00% | 60.00% | 37.70% | 50.82% | spy_vol_21 | 70.00% | 15.46% | 70.00% | 145.00% | 70.00% | 40.00% | 2.20 | 100.02% | 2.29 | -28.63% | 57.80% | 1.41 | -33.01% | 79.22% | 1.87 | -33.01% |

## Validation versus baseline

| portfolio | full_annual_return | full_sharpe | full_max_drawdown | validation_annual_return | validation_sharpe | validation_max_drawdown |
| --- | --- | --- | --- | --- | --- | --- |
| selected_state_overlay | 79.22% | 1.87 | -33.01% | 57.80% | 1.41 | -33.01% |
| baseline_sharpe_guarded_ml_growth_25 | 42.03% | 1.64 | -22.72% | 30.94% | 1.26 | -22.36% |

## Estimated next-year distribution

| portfolio | expected_return | median_return | p05_return | p95_return | loss_probability |
| --- | --- | --- | --- | --- | --- |
| selected_state_overlay | 73.31% | 59.69% | -13.46% | 205.41% | 0.1043 |

## Interpretation

- Adding a high-volatility cut to the breadth overlay improved pre-2022 objective and validation return versus the prior guarded baseline.
- It improves return and Sharpe versus the prior breadth-only overlay, but accepts slightly higher validation drawdown.
- This remains a validation result, not an untouched final test.
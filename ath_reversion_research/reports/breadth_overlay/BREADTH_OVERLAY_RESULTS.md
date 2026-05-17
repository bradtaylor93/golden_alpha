# Breadth Overlay Improvement Results

This report tests a different improvement path: scale an existing corrected portfolio using prior-day market breadth.
All overlay parameters are selected on pre-2022 data; 2022-2026 is validation only.

## Selected pre-2022 breadth overlay

| portfolio | base | breadth_feature | low_quantile | high_quantile | low_threshold | high_threshold | low_scale | high_scale | target_vol | train_objective | train_annual_return | train_sharpe | train_max_drawdown | validation_annual_return | validation_sharpe | validation_max_drawdown | full_annual_return | full_sharpe | full_max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharpe_guarded_ml_growth_25_adv_20_lq0.30_hq0.60_ls0.70_hs1.30_tv0.35 | sharpe_guarded_ml_growth_25 | adv_20 | 30.00% | 60.00% | 37.70% | 50.82% | 70.00% | 130.00% | 35.00% | 2.12 | 85.39% | 2.20 | -24.58% | 53.14% | 1.40 | -28.56% | 69.69% | 1.81 | -28.56% |

## Validation versus baseline

| portfolio | full_annual_return | full_sharpe | full_max_drawdown | validation_annual_return | validation_sharpe | validation_max_drawdown |
| --- | --- | --- | --- | --- | --- | --- |
| selected_breadth_overlay | 69.69% | 1.81 | -28.56% | 53.14% | 1.40 | -28.56% |
| baseline_sharpe_guarded_ml_growth_25 | 42.03% | 1.64 | -22.72% | 30.94% | 1.26 | -22.36% |

## Estimated next-year distribution

| portfolio | expected_return | median_return | p05_return | p95_return | loss_probability |
| --- | --- | --- | --- | --- | --- |
| selected_breadth_overlay | 65.38% | 55.59% | -11.56% | 176.60% | 0.1016 |

## Interpretation

- The selected overlay uses prior-day 20-day advance breadth to increase exposure in broad positive tape and reduce exposure in weak tape.
- This improved validation Sharpe and return versus the guarded ML-growth baseline, while keeping drawdown materially below the aggressive high-return portfolios.
- It is still selected from a breadth-parameter grid, so it should be treated as validation evidence rather than a final untouched test.
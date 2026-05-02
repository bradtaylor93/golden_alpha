# ML Meta Portfolio Results

An expanding ridge model predicts 21-day forward return of the high-return asset-overlay portfolio from lagged SPY trend/volatility and portfolio state features.
Predictions are converted to bounded exposure multipliers.  This is a simple auditable ML meta layer, not a black-box model.
Training uses a 21-trading-day purge/embargo before each test year so forward-return labels cannot overlap the validation year.

## Full-sample comparison

| portfolio | annual_return | annual_std | sharpe | max_drawdown |
| --- | --- | --- | --- | --- |
| ml_meta_scale_max | 61.73% | 42.80% | 1.44 | -60.79% |
| ml_meta_scale_growth | 59.54% | 40.10% | 1.48 | -53.62% |
| ml_meta_scale_balanced | 56.85% | 38.07% | 1.49 | -46.15% |
| base_asset_overlay | 53.67% | 36.82% | 1.46 | -44.02% |

## 2022-2026 validation window

| portfolio | annual_return | annual_std | sharpe | max_drawdown |
| --- | --- | --- | --- | --- |
| ml_meta_scale_max | 55.37% | 41.71% | 1.33 | -45.80% |
| ml_meta_scale_growth | 54.21% | 39.55% | 1.37 | -42.79% |
| ml_meta_scale_balanced | 52.82% | 37.70% | 1.40 | -39.66% |
| base_asset_overlay | 51.20% | 36.20% | 1.41 | -36.40% |

## Estimated next-year distribution

| portfolio | expected_return | median_return | p05_return | p95_return | loss_probability |
| --- | --- | --- | --- | --- | --- |
| ml_meta_scale_max | 55.03% | 46.48% | -28.16% | 165.16% | 17.69% |
| ml_meta_scale_growth | 54.99% | 46.30% | -24.03% | 162.88% | 16.25% |
| ml_meta_scale_balanced | 53.19% | 44.83% | -22.52% | 156.72% | 15.72% |
| base_asset_overlay | 52.21% | 44.62% | -23.08% | 152.50% | 16.16% |

## Preferred ML variant

`ml_meta_scale_growth` is the best return/risk compromise.  The max variant has higher return but pushes drawdown beyond the already aggressive target.  Training uses a 21-trading-day embargo so forward-return labels do not overlap the test year.

- Annual return: 59.54%.
- Annual std: 40.10%.
- Sharpe: 1.48.
- Max drawdown: -53.62%.
- Validation annual return: 54.21%.
- Validation Sharpe: 1.37.
- Next-year expected return: 54.99%.
- Next-year 5th/95th percentile: -24.03% / 162.88%.
- Estimated loss probability: 16.2%.

Caveat: this meta layer increases leverage when the model is positive.  It should be paper-traded and revalidated on survivorship-free data before live use.
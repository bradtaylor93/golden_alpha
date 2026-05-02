# ML Meta Portfolio Results

An expanding ridge model predicts 21-day forward return of the high-return asset-overlay portfolio from lagged SPY trend/volatility and portfolio state features.
Predictions are converted to bounded exposure multipliers.  This is a simple auditable ML meta layer, not a black-box model.

## Full-sample comparison

| portfolio | annual_return | annual_std | sharpe | max_drawdown |
| --- | --- | --- | --- | --- |
| ml_meta_scale_max | 63.25% | 42.30% | 1.50 | -59.02% |
| ml_meta_scale_growth | 60.51% | 39.77% | 1.52 | -52.26% |
| ml_meta_scale_balanced | 57.31% | 37.91% | 1.51 | -44.87% |
| base_asset_overlay | 53.67% | 36.82% | 1.46 | -44.02% |

## 2022-2026 holdout

| portfolio | annual_return | annual_std | sharpe | max_drawdown |
| --- | --- | --- | --- | --- |
| ml_meta_scale_max | 55.94% | 41.76% | 1.34 | -45.58% |
| ml_meta_scale_growth | 54.58% | 39.61% | 1.38 | -42.64% |
| ml_meta_scale_balanced | 53.00% | 37.74% | 1.40 | -39.58% |
| base_asset_overlay | 51.20% | 36.20% | 1.41 | -36.40% |

## Estimated next-year distribution

| portfolio | expected_return | median_return | p05_return | p95_return | loss_probability |
| --- | --- | --- | --- | --- | --- |
| ml_meta_scale_max | 56.81% | 48.30% | -26.50% | 166.99% | 16.54% |
| ml_meta_scale_growth | 56.16% | 47.41% | -23.23% | 164.56% | 15.58% |
| ml_meta_scale_balanced | 53.81% | 45.51% | -22.09% | 157.50% | 15.40% |
| base_asset_overlay | 52.21% | 44.62% | -23.08% | 152.50% | 16.16% |

## Preferred ML variant

`ml_meta_scale_growth` is the best return/risk compromise.  The max variant has higher return but pushes drawdown beyond the already aggressive target.

- Annual return: 60.51%.
- Annual std: 39.77%.
- Sharpe: 1.52.
- Max drawdown: -52.26%.
- Holdout annual return: 54.58%.
- Holdout Sharpe: 1.38.
- Next-year expected return: 56.16%.
- Next-year 5th/95th percentile: -23.23% / 164.56%.
- Estimated loss probability: 15.6%.

Caveat: this meta layer increases leverage when the model is positive.  It should be paper-traded and revalidated on survivorship-free data before live use.
# Champion Edge Recipe (Ambitious, No Breaking Changes)

`ChampionEdgeRecipe` is an intentionally ambitious recipe designed to maximize predictive edge using only current engine capabilities.

File: `trading_research/recipes/champion_edge.py`

## Core idea

Blend complementary alpha models, risk/context models, and explicit model-health signals:

1. **Diverse base alpha models**
   - LOESS (`ret_loess_base`) for local nonlinear structure
   - Kernel ridge (`ret_kernel_rich`) for smooth global nonlinearity
   - Ridge (`ret_ridge_stable`) for stable linear signal
   - Persistence baseline (`ret_persistence_baseline`) as a hard benchmark

2. **Risk/context side-channel**
   - Realized volatility model (`vol_kernel_risk`)
   - Direction probability classifier (`dir_logistic_prob`)

3. **Explicit diagnostics and causal projection**
   - `DiagnosticNode` for deviance gap, feature drift, residual dispersion, calibration gap
   - `ProjectionNode(mode="prev_fold_to_test")` to turn fold health into row-level causal features

4. **Derived targets for model weakness structure**
   - abs error targets
   - signed residual target
   - disagreement target across return models
   - calibration-gap target from classification diagnostics

5. **Error/disagreement forecasters**
   - dedicated models trained on derived targets
   - outputs fed into final alpha model as confidence/fragility context

6. **Champion meta model**
   - `champion_meta_return` consumes:
     - rich + regime + projected diagnostic features
     - base alpha predictions
     - risk/context predictions
     - error/disagreement forecasts
     - diagnostic inputs

7. **Selection layer**
   - `SelectionNode` chooses best recent performer among strong candidates.

## Why this can produce edge

- **Bias diversification**: local smoother + nonlinear kernel + stable linear.
- **Regime adaptation**: explicit regime and drift signals are projected with leakage-safe lag.
- **Confidence awareness**: error/disagreement forecasters provide uncertainty-aware context.
- **Risk-adjusted alpha**: direction and volatility channels reduce blind spots in pure return modeling.

## Run example

```bash
PYTHONPATH=/workspace python3 trading_research/examples/11_champion_edge_recipe.py
```

Outputs:
- `trading_research/examples/_output/11_champion_edge/reports/candidate_oos_performance.csv`
- `trading_research/examples/_output/11_champion_edge/reports/selector_decision.csv`


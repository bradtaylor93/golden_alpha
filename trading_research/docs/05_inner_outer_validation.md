# Inner vs Outer Validation

Outer validation estimates generalization on unseen time windows.
Inner validation selects hyperparameters safely inside each outer train window.

## Why nested?

Without nesting, tuning can "peek" at evaluation windows and inflate results.

## Engine behavior

- `FoldPlanNode(level="outer")` defines walk-forward evaluation windows.
- `TrainingRecipe.selection_validation` optionally defines inner walk-forward splits.
- Hyperopt (`HyperoptSpec`) is run on inner folds only.
- Model can then refit on full outer-train window.

## Modes

- `pooled`: all assets in one sample matrix.
- `per_asset`: split logic tracks each asset separately (scaffold in v1; extensible).

## Leakage controls

- Time-ordered folds.
- Embargo/gap between train end and test start.
- Train-fitted transforms only use train rows.

# vol_shape_regimes

Research module for volatility-shape embeddings, state clustering, and regime transition modeling.

## What it does

For each time `t`, it builds a trailing volatility-window object (shape), embeds it into a latent space,
clusters those shapes into interpretable states, and models `P(S_{t+1} | S_t, ...)`.

Outputs include:

- cluster diagnostics and interpretable summaries
- transition matrices
- dwell-time and persistence stats
- out-of-sample next-state prediction metrics
- baseline comparisons
- plots and markdown report

## Quickstart

```bash
cd /workspace/vol_shape_regimes
PYTHONPATH=/workspace/vol_shape_regimes/src python3 -m vol_shape_regimes.pipeline \
  --config /workspace/vol_shape_regimes/configs/experiment_default.yaml
```

Outputs are saved under:

`/workspace/vol_shape_regimes/outputs/<experiment_name>/`

## Tests

```bash
cd /workspace/vol_shape_regimes
PYTHONPATH=/workspace/vol_shape_regimes/src python3 -m pytest -q tests
```

## Notes

- Strict chronological splits are enforced.
- All transforms (scaler/PCA/cluster fit) are fit on train only.
- The pipeline starts with a robust single-asset daily setup and is structured for extension to multi-asset later.

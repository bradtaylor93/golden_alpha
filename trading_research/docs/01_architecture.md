# 01 Architecture

```
Recipe API (user-facing)
   |
   v compile()
Typed Workflow Graph
   |
   v run()
Node Executor ------> Artifact Store + Index + Manifest
   |
   v
Inspection API
```

## Two-layer model

1. **Recipe layer** keeps user workflows concise.
2. **Typed graph layer** enforces explicit artifact contracts and dependencies.

## Key choices

- Artifact-first persistence.
- Typed nodes instead of generic "op" nodes.
- Explicit diagnostics and projections for leakage safety.
- Fold-scoped artifacts using `ArtifactScope`.
- Pluggable real-data vendors (e.g., Polygon) behind a stable bars schema contract.
- Model registry that separates algorithm implementation from workflow orchestration.

## Execution model

- Runner resolves graph topologically.
- Every node writes one or more typed artifacts.
- Every artifact is indexed with lineage metadata.

## Data vendor layer

The engine includes two vendor modes:

1. `MockMarketDataVendor`: deterministic synthetic bars for examples/tests.
2. `PolygonMarketDataVendor`: real OHLCV ingestion from Polygon aggregates API.

Both vendors emit canonical columns:

`timestamp, asset, open, high, low, close, volume`

So downstream targets/features/validation are vendor-agnostic.

## Expanded model layer

In addition to ridge/logistic, the model registry now supports:

- `loess` (locally weighted regression with Gaussian kernel)
- `kernel_ridge` (RBF kernel ridge regression)

These are plug-compatible with existing `ModelSpec` and `ModelNode` flow.


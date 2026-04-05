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

## Execution model

- Runner resolves graph topologically.
- Every node writes one or more typed artifacts.
- Every artifact is indexed with lineage metadata.


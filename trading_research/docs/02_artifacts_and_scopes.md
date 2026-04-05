# Artifacts and Fold Scopes

The engine stores explicit typed artifacts for every major workflow stage.

## ArtifactScope

Each artifact has:

- `cv_level`: `global` / `outer_fold` / `inner_fold`
- `split_role`: `full` / `train` / `test` / `oof` / `validation`
- `outer_fold_id`: optional integer
- `inner_fold_id`: optional integer

This avoids ambiguity and leakage.

## Artifact Types

- `BarsArtifact`
- `TargetTableArtifact`
- `FeatureTableArtifact`
- `FoldPlanArtifact`
- `PredictionArtifact`
- `DiagnosticArtifact`
- `SelectionArtifact`
- `ModelStateArtifact`
- `TrainingDatasetSnapshotArtifact`

## On-Disk Layout

```
runs/<run_id>/
  config/
    run_config.json
    workflow_graph.json
  manifest.json
  artifact_index.parquet
  global/
    bars/...
    targets/...
    folds/...
  outer_fold_000/
    model_ret20/...
    diag_ret20/...
```

## Lineage

Each artifact stores `upstream_ids`, enabling graph traversal from any output artifact back to source inputs.

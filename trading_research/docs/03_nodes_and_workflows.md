# Nodes and Workflows

The engine uses explicit node types to avoid ambiguity.

## Node catalog

- `DataNode` -> `BarsArtifact`
- `TargetNode` -> `TargetTableArtifact`
- `FoldPlanNode` -> `FoldPlanArtifact`
- `FeatureNode` -> `FeatureTableArtifact`
- `TransformNode` -> `FeatureTableArtifact` (fit-on-train transforms)
- `ModelNode` -> `PredictionArtifact`, `ModelStateArtifact`, diagnostics, optional snapshots
- `DiagnosticNode` -> `DiagnosticArtifact`
- `DerivedFeatureNode` -> `FeatureTableArtifact`
- `ProjectionNode` -> `FeatureTableArtifact` from fold diagnostics
- `SelectionNode` -> `SelectionArtifact`

## Typed inputs

Each node has an explicit input dataclass:

- `ModelInputs`
- `DiagnosticNodeInputs`
- `DerivedFeatureNodeInputs`
- `ProjectionNodeInputs`
- `SelectionNodeInputs`

No node relies on free-form untyped arguments.

## Example graph

```
bars -> targets -> model_a -> diag_a -> projection -> meta_model
   \-> folds ---^           \-> model_b -> diag_b -/
   \-> features -----------/
```

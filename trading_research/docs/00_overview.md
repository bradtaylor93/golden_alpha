# Trading Research Engine Overview

This system is a **two-layer design**:

1. **Recipe API (user-facing)** for concise experiment declarations.
2. **Typed execution graph** for explicit, auditable artifact execution.

```
Recipe -> Compiler -> WorkflowGraph -> Runner -> Run Directory + Artifact Index
```

Core guarantees:

- Artifact-first persistence
- Fold-aware scope metadata
- Explicit diagnostics and projection nodes
- Reproducible runs with graph/config snapshots and lineage index

## Minimal usage

```python
graph = SingleModelRecipe(universe_name="u1").compile()
run_id = WorkflowRunner(runs_root=Path("runs"), catalog=catalog).run(graph, {"exp": "quick"})
```

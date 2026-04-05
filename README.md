# Trading Research Engine

`trading_research` is a typed, artifact-first engine for systematic alpha research.

It implements the requested two-layer architecture:

- Layer A: concise user-facing recipe API
- Layer B: explicit typed execution graph with persistent fold-scoped artifacts

Start here:

- Overview: `trading_research/docs/00_overview.md`
- Architecture: `trading_research/docs/01_architecture.md`
- Examples: `trading_research/examples/`

Quick start:

```bash
python3 trading_research/examples/01_single_model_run.py
python3 trading_research/examples/07_advanced_cross_target_meta_run.py
```

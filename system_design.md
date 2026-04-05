Build a production-quality Python trading research engine designed for systematic alpha research, predictive modeling, stacked/meta models, and deeply auditable workflow execution.

This is not just a backtesting library or a simple ML pipeline. It must support:
- persistent market data storage
- extensible feature engineering
- flexible prediction task definitions
- regression and classification models
- walk-forward validation
- inner and outer fold optimization
- diagnostics and drift analysis
- stacked/meta models
- model selection / blending
- explicit artifact persistence and lineage
- deep run inspection

The codebase must be modular, typed, clean, and practical.

==================================================
1. CORE DESIGN DECISION
==================================================

The engine must use a TWO-LAYER DESIGN:

LAYER A: USER-FACING RECIPE API
- This is what researchers mostly write.
- It should be concise and readable.
- Examples:
  - single model run
  - feature comparison run
  - nested tuning run
  - stacked model run
  - diagnostic-aware meta model run
  - selector / ensemble run
- Recipes compile into a workflow graph internally.

LAYER B: TYPED EXECUTION GRAPH
- This is the actual execution engine.
- It must be artifact-first, fold-scoped, typed, and auditable.
- It must support complex, custom dependent workflows.

DO NOT make the primary user API a generic low-level DAG only.
DO NOT make everything one giant integrated pipeline.
Use recipes on top of a typed graph.

==================================================
2. KEY ARCHITECTURAL PRINCIPLES
==================================================

A. ARTIFACT-FIRST
Everything important should be persisted as an explicit artifact:
- bars
- target tables
- feature tables
- fold plans
- model fits or model state summaries
- predictions
- diagnostics
- selections
- projected meta features
- training dataset snapshots for complex model nodes

B. FOLD-SCOPED ARTIFACTS
Every artifact must know:
- cv_level: global / outer_fold / inner_fold
- split_role: full / train / test / oof / validation
- outer_fold_id
- inner_fold_id

This is essential for leakage safety and auditability.

C. TYPED NODES
Do NOT use vague generic node definitions with implicit semantics.
Use explicit node types:
- DataNode
- TargetNode
- FoldPlanNode
- FeatureNode
- TransformNode
- ModelNode
- DiagnosticNode
- DerivedFeatureNode
- ProjectionNode
- SelectionNode

D. SEPARATE ESTIMATOR VS TRAINING PROCEDURE
ModelSpec defines:
- algorithm
- params
- task compatibility

TrainingRecipe defines:
- inner validation
- hyperparameter search
- refit policy
- calibration
- OOF prediction generation
- training snapshot persistence

E. DIAGNOSTICS AS FIRST-CLASS
Diagnostics must not be hidden side effects inside model nodes.
They should be explicit artifacts and often explicit nodes.

F. PROJECTION DISCIPLINE
Fold-level diagnostics should NOT be silently joined into row-level data.
If diagnostics are to become model inputs, they must pass through an explicit ProjectionNode that creates safe, lagged, row-level features.

G. REPRODUCIBILITY
Every run must be reconstructable from:
- config snapshot
- compiled workflow graph
- artifact index
- manifest
- code version / parameter hash metadata

==================================================
3. MAIN CAPABILITIES REQUIRED
==================================================

The system must support:

1. Asset universe definition
2. Historical data retrieval and local persistence
3. Feature family registry and feature set composition
4. Prediction task registry
5. Model registry
6. Walk-forward validation
7. Nested inner-fold selection / tuning
8. Multi-asset pooled and per-asset modes
9. Diagnostics and drift analysis
10. Stacking/meta models using prior model artifacts
11. Selection / blending / model governance
12. Hyperparameter optimization
13. Strong run auditability and inspectability

==================================================
4. PROJECT STRUCTURE
==================================================

Use a structure similar to:

trading_research/
    configs/
        universes/
        tasks/
        models/
        validations/
        recipes/

    data/
        vendors/
        storage/
        catalog.py
        schemas.py

    features/
        registry.py
        baseline.py
        field_adapter.py
        cross_asset.py
        regime.py

    targets/
        registry.py
        returns.py
        volatility.py
        classification.py
        extremes.py

    models/
        registry.py
        preprocessing.py
        regression.py
        classification.py
        calibration.py

    validation/
        splits.py
        embargo.py

    workflow/
        artifacts.py
        nodes.py
        graph.py
        compiler.py
        runner.py
        manifest.py
        inspectors.py

    recipes/
        base.py
        single_model.py
        comparison.py
        nested_tuning.py
        stacking.py
        diagnostics_meta.py
        selector.py

    diagnostics/
        metrics.py
        drift.py
        residuals.py
        stability.py
        calibration.py

    analysis/
        performance.py
        residual_analysis.py
        subgroup_analysis.py
        comparison.py
        plots.py

    optimization/
        search.py
        objective.py

    utils/
        io.py
        hashing.py
        logging.py
        typing.py
        paths.py

    examples/
        01_single_model_run.py
        02_feature_comparison_run.py
        03_nested_tuning_run.py
        04_stacked_meta_run.py
        05_diagnostic_aware_meta_run.py
        06_selector_run.py

    docs/
        00_overview.md
        01_architecture.md
        02_artifacts_and_scopes.md
        03_nodes_and_workflows.md
        04_recipes.md
        05_inner_outer_validation.md
        06_diagnostics_and_projection.md
        07_run_audit_and_inspection.md
        tutorials/
            tutorial_01_run_first_experiment.md
            tutorial_02_add_feature_family.md
            tutorial_03_add_prediction_task.md
            tutorial_04_add_model.md
            tutorial_05_build_stacked_workflow.md

==================================================
5. ARTIFACT MODEL
==================================================

Implement a strong artifact abstraction.

Need dataclasses roughly like:

- ArtifactScope
    - cv_level: "global" | "outer_fold" | "inner_fold"
    - split_role: "full" | "train" | "test" | "oof" | "validation"
    - outer_fold_id: int | None
    - inner_fold_id: int | None

- ArtifactRef
    - artifact_id
    - artifact_type
    - node_name
    - path
    - scope
    - schema
    - metadata
    - upstream_ids

Artifact types should include at least:
- BarsArtifact
- TargetTableArtifact
- FeatureTableArtifact
- FoldPlanArtifact
- PredictionArtifact
- DiagnosticArtifact
- SelectionArtifact
- ModelStateArtifact
- TrainingDatasetSnapshotArtifact

Implement:
- artifact index persistence
- lineage tracking
- artifact load helpers

==================================================
6. NODE TYPES
==================================================

Implement explicit typed node classes.

A. DataNode
- produces BarsArtifact

B. TargetNode
- produces TargetTableArtifact
- defines task_name and horizon or references a PredictionTask

C. FoldPlanNode
- produces FoldPlanArtifact
- supports outer and inner levels

D. FeatureNode
- produces FeatureTableArtifact
- references feature family name + params
- should be used for stateless causal features

E. TransformNode
- produces FeatureTableArtifact
- for train-fitted transforms like:
  - scaling
  - PCA
  - feature selection
  - clipping rules if fit-dependent

F. ModelNode
- produces:
  - PredictionArtifact
  - ModelStateArtifact
  - maybe DiagnosticArtifact summary
  - optionally SelectionArtifact if embedded tuning is used
- consumes:
  - features
  - targets
  - folds
  - predictions
  - diagnostics
  - selections
- uses ModelInputs + TrainingRecipe

G. DiagnosticNode
- produces DiagnosticArtifact
- examples:
  - train_test_deviance_gap
  - feature_drift
  - residual_std_by_asset
  - prediction_disagreement
  - coefficient_stability
  - calibration_gap

H. DerivedFeatureNode
- produces FeatureTableArtifact
- examples:
  - residual std by asset
  - model disagreement score
  - rolling model health

I. ProjectionNode
- produces FeatureTableArtifact
- converts fold-level diagnostics into row-level lagged meta-features
- must be explicit to avoid silent leakage

J. SelectionNode
- produces SelectionArtifact
- examples:
  - best recent model
  - weighted blend
  - regime-conditioned selection

==================================================
7. INPUT SPECS
==================================================

Implement typed input specs for nodes, especially ModelNode.

Need dataclass like:

ModelInputs:
- bars: list[str]
- fold_plans: list[str]
- feature_tables: list[str]
- target_tables: list[str]
- prediction_tables: list[str]
- diagnostic_tables: list[str]
- selection_tables: list[str]
- use_base_features: bool

Similarly create typed inputs for:
- DiagnosticNode
- DerivedFeatureNode
- ProjectionNode
- SelectionNode

Do NOT use vague free-form input semantics.

==================================================
8. MODEL / TRAINING SPECS
==================================================

Implement:

ModelSpec:
- name
- algorithm
- params

TrainingRecipe:
- selection_validation: ValidationSpec | None
- hyperopt: HyperoptSpec | None
- refit_on_full_outer_train: bool
- generate_train_oof_predictions: bool
- save_training_snapshot: bool

HyperoptSpec:
- method: grid or random
- objective
- param_grid or random search space

ValidationSpec:
- split_type: expanding | rolling
- min_train_size
- test_size
- embargo
- rolling_train_size
- mode: pooled | per_asset

==================================================
9. RECIPE LAYER
==================================================

Implement high-level recipe classes that compile to the graph.

At minimum:
- SingleModelRecipe
- FeatureComparisonRecipe
- NestedTuningRecipe
- StackedModelRecipe
- DiagnosticAwareMetaRecipe
- SelectorRecipe

Recipes should produce a compiled workflow graph internally.

This is critical:
most users should not manually author low-level graph specs unless needed.

==================================================
10. FEATURE SYSTEM
==================================================

Implement a feature registry with family abstraction.

Need:
- FeatureFamily base interface
- FeatureRegistry
- parameter hashing/versioning
- feature persistence cache

Implement at least:
1. baseline feature family
2. field_adapter feature family
3. cross_asset feature family
4. regime feature family

Important:
The field adapter should define the interface but not hardcode the user’s proprietary field implementation. Make it easy to plug in an external package.

==================================================
11. TARGET / TASK SYSTEM
==================================================

Implement PredictionTask abstraction and TaskRegistry.

PredictionTask should define:
- name
- task_type: regression or classification
- horizon
- build_target_fn
- metric names
- embargo

Implement at least:
- forward return
- forward realized volatility
- up/down classification
- max upside
- max drawdown

==================================================
12. DATA / STORAGE SYSTEM
==================================================

Implement:
- local data catalog
- raw vs processed storage separation
- Parquet as primary storage format
- CSV export support

Data requirements:
- define universe
- retrieve historical bars
- normalize schema
- persist locally
- load quickly
- track metadata/versioning

Even if vendor support is initially simple/scaffolded, the storage layer must be robust.

==================================================
13. VALIDATION REQUIREMENTS
==================================================

Must support:
- expanding walk-forward
- rolling walk-forward
- embargo/gap handling
- pooled mode
- per-asset mode

Must ensure:
- train-only transforms are fit only on train
- tuning is nested correctly
- OOF predictions are valid
- no same-row target leakage

==================================================
14. DIAGNOSTICS / DRIFT REQUIREMENTS
==================================================

Implement diagnostics as explicit artifacts.

Need support for:
- train-test deviance gap
- feature drift
- residual std by asset
- prediction disagreement
- coefficient stability
- calibration gap

Feature drift examples can include:
- mean/std shifts
- quantile shifts
- PSI-like summary
- simple correlation structure changes

These diagnostics should be generated via DiagnosticNode, not hidden in model internals.

==================================================
15. META-MODEL SUPPORT
==================================================

This system MUST support downstream models trained on:
- prior model predictions
- diagnostics from prior models
- projected lagged fold-health features
- models trained on different targets

Important:
This means a later model can train on:
- base features
- predictions from a return model
- predictions from a volatility model
- lagged train/test deviance
- lagged feature drift
while itself predicting another target entirely.

This is a required capability.

Implement this by:
- explicit PredictionArtifact outputs
- explicit DiagnosticArtifact outputs
- ProjectionNode for lagged row-level diagnostic features
- ModelNode with typed mixed-task inputs

==================================================
16. SELECTION / ENSEMBLE SUPPORT
==================================================

Implement SelectionNode and selection strategies.

Needed strategies:
- best recent model by metric
- weighted blend by recent performance
- diagnostic-aware selection scaffold

Outputs should include:
- selection decisions
- comparison tables
- chosen prediction artifact
- or blend weights

==================================================
17. RUN MANAGEMENT AND AUDITABILITY
==================================================

Every run must persist:

1. config snapshot
2. compiled workflow graph
3. manifest
4. artifact index
5. fold-level outputs
6. metrics

Implement:
- run_id generation
- run directory layout
- artifact index (Parquet or JSON + Parquet)
- run manifest

Filesystem layout should resemble:

runs/<run_id>/
    config/
        run_config.json
        workflow_graph.json
    manifest.json
    artifact_index.parquet

    global/
        node_bars/
        node_targets/
        node_features/
        node_folds/

    outer_fold_000/
        node_x/
        node_y/

Each complex model node should optionally save:
- training_dataset_snapshot.parquet

This is important for debugging meta-models.

==================================================
18. INSPECTION API
==================================================

Implement a run inspection API.

Examples:
- load_run(run_id)
- list_nodes()
- list_artifacts()
- load_artifact(node_name=..., outer_fold_id=..., artifact_type=..., split_role=...)
- show_lineage(node_name=..., outer_fold_id=...)
- load_predictions(node_name=..., outer_fold_id=...)
- load_inner_validation(node_name=..., outer_fold_id=...)
- load_training_snapshot(node_name=..., outer_fold_id=...)

This is a core requirement.

==================================================
19. EXAMPLE WORKFLOWS TO IMPLEMENT
==================================================

Implement end-to-end examples demonstrating the design from simple to complex.

Example 1: simplest possible run
- single asset
- baseline features
- one target
- one ridge model
- expanding outer CV

Example 2: feature comparison run
- baseline vs field vs combined
- compare performance cleanly

Example 3: nested tuning run
- inner expanding validation
- alpha search for ridge
- outer test evaluation

Example 4: stacked meta run
- base model
- meta model using base OOF predictions

Example 5: diagnostic-aware meta run
- base return model
- base volatility model
- diagnostics:
  - train_test_deviance_gap
  - feature_drift
- projection to lagged row-level model-health features
- final downstream model on a different target

Example 6: selector run
- multiple candidate models
- selection based on recent fold performance

==================================================
20. CRITICAL COMPLEX WORKFLOW EXAMPLE
==================================================

Implement one advanced example proving the architecture supports the hardest requirement:

A model trained on:
- baseline features
- predictions from previous models trained on DIFFERENT targets
- lagged projected diagnostics such as:
  - train/test deviance gap
  - feature drift
  - residual std by asset

And the downstream model itself predicts a different target.

Concretely, support a workflow like:
- ret20_model predicts forward return 20
- vol20_model predicts forward realized volatility 20
- diagnostics are computed from both
- diagnostics are projected into lagged row-level health features
- final meta model predicts forward return 40 or max_upside_40 using:
  - baseline features
  - ret20 predictions
  - vol20 predictions
  - lagged health/drift features

This example is REQUIRED.

==================================================
21. QUALITY REQUIREMENTS
==================================================

Code must be:
- typed
- dataclass-based where appropriate
- modular
- practical
- documented
- testable
- not overengineered nonsense

Add:
- docstrings
- comments where design intent matters
- TODOs for future extension
- clear naming
- no giant god files

==================================================
22. DOCS REQUIREMENT
==================================================

Create deep docs, not shallow README-style summaries.

Need markdown docs explaining:
- system overview
- architecture
- artifacts and scopes
- node types
- recipe layer
- inner vs outer validation
- diagnostics and projection
- run audit and inspection

Include code examples and ASCII diagrams.

Also create tutorials:
- run first experiment
- add feature family
- add target
- add model
- build stacked workflow
- inspect a complex run

==================================================
23. IMPLEMENTATION STRATEGY
==================================================

Build in stages:

STAGE 1
- project structure
- core dataclasses
- artifact model
- node types
- run manifest/index
- baseline data/feature/task/model pipeline

STAGE 2
- recipe compiler
- validation engine
- model runner
- feature persistence
- example simple runs

STAGE 3
- diagnostics
- projection
- meta-model support
- selector support
- advanced example workflows

STAGE 4
- docs
- polish
- inspection helpers

==================================================
24. IMPORTANT DESIGN NON-GOALS
==================================================

Do NOT:
- collapse everything into one monolithic pipeline
- hide inner-CV outputs entirely inside black-box model code
- rely on notebooks as the primary orchestration layer
- make workflows impossible to audit
- use ambiguous node semantics
- silently leak diagnostics into row-level features

==================================================
25. FINAL GOAL
==================================================

The final system should make it possible to say:

“Given a universe, data source, feature set, prediction task, model spec, and validation scheme, I can run a fully reproducible research workflow, including nested tuning, stacked/meta models, and diagnostic-aware downstream models, while preserving fold-safe artifacts, deep run lineage, and easy inspection of every stage.”

Now implement this codebase.
Start by creating the project structure and core dataclasses.
Then implement the minimal working engine.
Then add the example workflows.
Then add docs.

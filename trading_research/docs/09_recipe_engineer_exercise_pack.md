# Recipe Engineer Exercise Pack (20 Exercises)

This pack is for engineers who need deep, practical mastery of the trading research engine.

If a learner can complete all 20 exercises **unassisted**, they should be able to design, implement, debug, audit, and extend production-quality recipes in this system.

---

## How to use this pack

- Work in a feature branch and keep each exercise in separate commits.
- Do not skip artifacts, metadata, or audit outputs.
- For every exercise, capture:
  - graph snippet or code diff,
  - run ID,
  - key artifacts produced,
  - short retrospective on what failed first and how it was fixed.

---

## Evidence requirements (for every exercise)

To mark an exercise complete, learner must provide:

1. **Recipe/runner code change** (or explicit note: "no code changes needed" if analysis-only).
2. **Successful run output** with run ID.
3. **Artifact proof** (paths or `RunInspector` evidence).
4. **Why this is causally safe** (no leakage, proper folds/scopes).
5. **One failure mode** and guardrail.

---

## Exercise 1 - Compile and inspect a minimal recipe

**Goal:** Build and run a minimal graph (`DataNode -> TargetNode -> FoldPlanNode -> FeatureNode -> ModelNode`).

**Requirements:**
- Use one target task and one feature family.
- Persist run config and graph.
- Show topological order.

**Pass criteria:**
- Run succeeds.
- Prediction, model state, and diagnostics artifacts exist.
- Learner can explain how node dependencies become execution order.

---

## Exercise 2 - Artifact scope literacy

**Goal:** Prove understanding of `ArtifactScope` fields.

**Requirements:**
- For one run, list artifacts by `cv_level`, `split_role`, and fold IDs.
- Explain why each major artifact has its current scope.

**Pass criteria:**
- Correctly identifies at least one global artifact and one fold-scoped artifact.
- No confusion between train/test/oof/validation roles.

---

## Exercise 3 - Add structured node labels

**Goal:** Use `WorkflowNode.labels` for machine-readable semantics.

**Requirements:**
- Add labels for `task_family`, `horizon`, and `role` to at least 3 node types.
- Verify labels appear in artifact metadata.

**Pass criteria:**
- Metadata contains `node_labels`.
- Post-run analysis can group by labels without parsing node name strings.

---

## Exercise 4 - Create a two-model comparison recipe

**Goal:** Compare two model specs on same target/folds.

**Requirements:**
- Build two `ModelNode`s with different algorithms or params.
- Produce fold-level and overall comparison table.

**Pass criteria:**
- Both model branches run successfully.
- Comparison is based on out-of-sample test rows only.

---

## Exercise 5 - Implement and use `DerivedTargetNode` (abs error)

**Goal:** Train a meta-model on derived absolute error target.

**Requirements:**
- Add a `DerivedTargetNode` with `target_kind="abs_error"`.
- Feed derived target into downstream `ModelNode`.

**Pass criteria:**
- Derived target artifact exists and has correct metadata.
- Learner explains why derived target is valid for error-model training.

---

## Exercise 6 - `DerivedTargetNode` variants

**Goal:** Exercise all derived target modes.

**Requirements:**
- Build 3 derived targets:
  - `signed_residual`
  - `calibration_gap`
  - `disagreement` (must use >=2 prediction tables)
- Run at least one downstream model using one of them.

**Pass criteria:**
- All 3 target tables materialize.
- Disagreement target logic verified on multi-model predictions.

---

## Exercise 7 - Explicit reusable diagnostics

**Goal:** Build reusable `DiagnosticNode`s decoupled from model internals.

**Requirements:**
- Create a standalone diagnostic node using prediction inputs.
- Include at least 3 diagnostics (e.g., deviance gap, residual std, feature drift).

**Pass criteria:**
- Diagnostic artifact created with requested `diagnostic_types`.
- Diagnostic outputs reused by at least one other node.

---

## Exercise 8 - Fold-safe `ProjectionNode`

**Goal:** Convert fold-level diagnostics into row-level features safely.

**Requirements:**
- Use `ProjectionNode` with fold plan and `mode="prev_fold_to_test"`.
- Demonstrate lag behavior and test-interval projection.

**Pass criteria:**
- Projected features are zero/default before fold history exists.
- Learner can prove no same-fold leakage.

---

## Exercise 9 - Build a diagnostic-aware meta-model

**Goal:** Train meta-model using:
- base features
- projected diagnostics
- upstream predictions

**Requirements:**
- At least one base model and one meta model.
- Meta model depends on projected diagnostics artifact.

**Pass criteria:**
- Meta model runs and produces oof predictions.
- Inputs are auditable via upstream lineage.

---

## Exercise 10 - Inner/outer fold geometry challenge

**Goal:** Tune fold geometry and justify tradeoffs.

**Requirements:**
- Run two configurations of outer/inner specs.
- Compare stability vs data efficiency.

**Pass criteria:**
- Learner articulates impact of `min_train_size`, `test_size`, embargo.
- Presents fold-level variance differences with evidence.

---

## Exercise 11 - Failure injection: unknown task handling

**Goal:** Trigger and then fix a task resolution failure.

**Requirements:**
- Intentionally reference an invalid task once.
- Add a robust fallback or clear error path in recipe/runner usage.

**Pass criteria:**
- Failure reproduced and explained.
- Final solution avoids silent misclassification of task type.

---

## Exercise 12 - Failure injection: empty fold/test slice

**Goal:** Prove robustness on sparse or edge-case fold boundaries.

**Requirements:**
- Create a configuration likely to produce empty train/test in some folds.
- Ensure run remains auditable and non-crashing.

**Pass criteria:**
- Engine emits artifacts with explicit skipped/empty behavior.
- Learner explains how downstream nodes remain stable.

---

## Exercise 13 - Artifact lineage deep dive

**Goal:** Trace one final prediction artifact back to root data.

**Requirements:**
- Use `RunInspector` and index lineage.
- Produce a lineage report with all upstream IDs/types.

**Pass criteria:**
- Lineage chain is complete and ordered.
- Learner identifies exactly where each feature/diagnostic entered pipeline.

---

## Exercise 14 - Metadata-first analytics (no string parsing)

**Goal:** Build a run summary grouped by metadata labels.

**Requirements:**
- Group predictions by `task_family` and `horizon` from metadata/labels.
- Keep fallback only for legacy runs.

**Pass criteria:**
- Summary works without relying on node-name token parsing.
- Learner can explain backward compatibility strategy.

---

## Exercise 15 - Selection strategy extension

**Goal:** Extend `SelectionNode` strategy.

**Requirements:**
- Add one strategy beyond baseline best-MSE (e.g., weighted blend with stability penalty).
- Persist strategy output as `SelectionArtifact`.

**Pass criteria:**
- Strategy artifact includes chosen model(s) and rationale fields.
- Reproducible selection given same artifacts.

---

## Exercise 16 - Confidence interval calibration workflow

**Goal:** Build and evaluate 95% interval coverage pipeline.

**Requirements:**
- Use error-model outputs for interval width (or proxy).
- Report empirical coverage and average interval width per fold.

**Pass criteria:**
- Coverage report generated and attached to run outputs.
- Learner explains over/under-coverage diagnostics.

---

## Exercise 17 - Multi-horizon expansion

**Goal:** Scale one recipe across multiple horizons cleanly.

**Requirements:**
- Run at least 4 horizons with shared recipe template.
- Avoid copy-paste fragility where possible.

**Pass criteria:**
- All horizon branches complete.
- Metadata labels correctly annotate each horizon branch.

---

## Exercise 18 - Add one new diagnostic end-to-end

**Goal:** Extend diagnostics subsystem and consume it.

**Requirements:**
- Implement one new diagnostic function.
- Wire through `DiagnosticNode` and project it via `ProjectionNode`.
- Use projected signal in a downstream model.

**Pass criteria:**
- New metric appears in diagnostic artifact and projected feature table.
- Learner validates expected directionality on at least one fold.

---

## Exercise 19 - Recipe quality hardening

**Goal:** Improve recipe maintainability and debuggability.

**Requirements:**
- Refactor one complex recipe for readability (naming, helper builders, labels).
- Add concise comments only where complexity is non-obvious.

**Pass criteria:**
- No behavioral regression.
- New engineer can follow graph intent from code + metadata.

---

## Exercise 20 - Capstone: design and defend a new production recipe

**Goal:** Create a novel recipe from scratch with:
- multiple targets or horizons,
- derived targets,
- explicit diagnostics,
- safe projections,
- meta model and selection strategy,
- metadata-first reporting.

**Requirements:**
- Full run + artifact audit + summary notebook/report.
- Include "what I would ship next" roadmap.

**Pass criteria:**
- Independent reviewer can reproduce run and verify lineage.
- Learner answers architecture-level questions without help.

---

## Mastery rubric (unassisted)

Learner is considered system-proficient if they can:

- Author new recipes without breaking type/artifact contracts.
- Diagnose and fix runner/graph integration failures quickly.
- Demonstrate fold-causal correctness (no leakage) with evidence.
- Use metadata-first analysis over brittle naming conventions.
- Explain how design choices impact auditability and production trust.

---

## Suggested scoring

- 0 = not attempted
- 1 = attempted, incomplete
- 2 = complete with guidance
- 3 = complete unassisted

**Passing bar:**  
`>= 55 / 60 total` and **Exercise 20 must be a 3**.


"""Workflow runner executing typed graph nodes with persisted artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.data.catalog import DataCatalog
from trading_research.diagnostics import (
    calibration_gap,
    coefficient_stability,
    feature_drift_summary,
    prediction_disagreement,
    residual_std_by_asset,
    train_test_deviance_gap,
)
from trading_research.features import (
    BaselineFeatureFamily,
    CrossAssetFeatureFamily,
    FeatureRegistry,
    FeatureSpec,
    FieldAdapterFeatureFamily,
    RegimeFeatureFamily,
)
from trading_research.features.research_pack import ResearchFeaturePackFamily
from trading_research.models.registry import ModelRegistry, default_model_registry
from trading_research.targets import TaskRegistry, default_task_registry
from trading_research.utils.hashing import stable_hash
from trading_research.utils.io import read_json, write_json
from trading_research.utils.paths import ensure_dir, run_root
from trading_research.validation.splits import build_walk_forward_splits
from trading_research.workflow.artifacts import (
    ArtifactIndex,
    ArtifactRef,
    ArtifactScope,
    ArtifactStore,
)
from trading_research.workflow.graph import WorkflowGraph
from trading_research.workflow.manifest import RunManifest, generate_run_id
from trading_research.workflow.nodes import (
    DataNode,
    DerivedTargetInputs,
    DerivedTargetNode,
    DerivedFeatureInputs,
    DerivedFeatureNode,
    DiagnosticInputs,
    DiagnosticNode,
    FeatureNode,
    FoldPlanNode,
    ModelInputs,
    ModelNode,
    ProjectionInputs,
    ProjectionNode,
    SelectionInputs,
    SelectionNode,
    TargetNode,
    WorkflowNode,
)


def default_feature_registry(cache_dir: Path | None = None) -> FeatureRegistry:
    registry = FeatureRegistry(cache_dir=cache_dir)
    registry.register(BaselineFeatureFamily())
    registry.register(FieldAdapterFeatureFamily())
    registry.register(CrossAssetFeatureFamily())
    registry.register(RegimeFeatureFamily())
    registry.register(ResearchFeaturePackFamily())
    return registry


class WorkflowRunner:
    """Minimal working execution engine with artifact persistence and lineage."""

    def __init__(
        self,
        runs_dir: Path | str = "runs",
        data_catalog: DataCatalog | None = None,
        task_registry: TaskRegistry | None = None,
        feature_registry: FeatureRegistry | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self.runs_dir = Path(runs_dir)
        ensure_dir(self.runs_dir)
        self.data_catalog = data_catalog or DataCatalog("trading_research/data/storage")
        self.task_registry = task_registry or default_task_registry()
        self.feature_registry = feature_registry or default_feature_registry(
            cache_dir=self.runs_dir / "_feature_cache"
        )
        self.model_registry = model_registry or default_model_registry()

    def run(
        self,
        graph: WorkflowGraph,
        *,
        run_config: dict[str, Any],
        run_id: str | None = None,
    ) -> str:
        rid = run_id or generate_run_id()
        run_dir = run_root(self.runs_dir, rid)
        ensure_dir(run_dir)
        config_dir = ensure_dir(run_dir / "config")
        write_json(config_dir / "run_config.json", run_config)
        graph.save(config_dir / "workflow_graph.json")

        store = ArtifactStore(run_dir)
        index = ArtifactIndex()
        produced: dict[str, list[ArtifactRef]] = {}

        ordered = graph.topological()
        for node in ordered:
            self._execute_node(node=node, store=store, index=index, produced=produced)
        store.persist_index(index)

        manifest = RunManifest(
            run_id=rid,
            created_at_utc=generate_run_id(prefix="ts").removeprefix("ts_"),
            code_version=stable_hash({"engine": "trading_research", "version": "0.1.0"})[:12],
            config_hash=stable_hash(run_config)[:16],
            notes=f"nodes={len(ordered)}",
        )
        manifest.save(run_dir / "manifest.json")
        return rid

    @staticmethod
    def _artifact_type_name(ref: ArtifactRef) -> str:
        return str(ref.artifact_type.value if hasattr(ref.artifact_type, "value") else ref.artifact_type)

    def _resolve_refs(
        self,
        names: list[str],
        produced: dict[str, list[ArtifactRef]],
        *,
        artifact_type: str | None = None,
    ) -> list[ArtifactRef]:
        refs: list[ArtifactRef] = []
        for name in names:
            if name not in produced or not produced[name]:
                raise KeyError(f"No produced artifacts found for node name '{name}'")
            candidates = produced[name]
            if artifact_type is not None:
                matches = [r for r in candidates if self._artifact_type_name(r) == artifact_type]
                if not matches:
                    raise KeyError(
                        f"No artifact of type {artifact_type} found for node '{name}'. "
                        f"Available: {[self._artifact_type_name(r) for r in candidates]}"
                    )
                refs.append(matches[-1])
            else:
                refs.append(candidates[-1])
        return refs

    def _write_artifact(
        self,
        *,
        store: ArtifactStore,
        index: ArtifactIndex,
        produced: dict[str, list[ArtifactRef]],
        node_name: str,
        artifact_type: str,
        data: pd.DataFrame,
        scope: ArtifactScope,
        schema: dict[str, Any],
        metadata: dict[str, Any],
        upstream_refs: list[ArtifactRef],
        node_type: str | None = None,
        suffix: str = "",
    ) -> ArtifactRef:
        full_meta = dict(metadata)
        if node_type is not None:
            full_meta["node_type"] = node_type
        ref = store.write(
            artifact_type=artifact_type,
            node_name=node_name,
            scope=scope,
            data=data,
            schema=schema,
            metadata=full_meta,
            upstream_ids=[u.artifact_id for u in upstream_refs],
            suffix=suffix,
        )
        index.add(ref)
        produced.setdefault(node_name, []).append(ref)
        return ref

    @staticmethod
    def _common_meta(node: WorkflowNode) -> dict[str, Any]:
        """Structured metadata to avoid downstream string parsing."""
        return {
            "node_type": node.__class__.__name__,
            "node_labels": dict(node.labels),
        }

    @staticmethod
    def _diag_metric(
        diagnostic_type: str,
        *,
        pred_fold: pd.DataFrame,
        pred_all: pd.DataFrame,
        feature_frame: pd.DataFrame | None,
    ) -> float:
        train = pred_fold[pred_fold["split_role"] == "train"]
        test = pred_fold[pred_fold["split_role"] == "test"]
        if diagnostic_type == "train_test_deviance_gap":
            if train.empty or test.empty:
                return float("nan")
            return train_test_deviance_gap(
                train["target"].to_numpy(dtype=float),
                train["prediction"].to_numpy(dtype=float),
                test["target"].to_numpy(dtype=float),
                test["prediction"].to_numpy(dtype=float),
            )
        if diagnostic_type == "residual_std_by_asset":
            if test.empty:
                return 0.0
            rs = residual_std_by_asset(test)
            return float(rs["residual_std_by_asset"].mean()) if not rs.empty else 0.0
        if diagnostic_type == "calibration_gap":
            if test.empty:
                return 0.0
            return calibration_gap(
                test["target"].to_numpy(dtype=float),
                test["prediction"].to_numpy(dtype=float),
            )
        if diagnostic_type == "feature_drift":
            if feature_frame is None or test.empty:
                return 0.0
            merged = (
                test[["timestamp", "asset"]]
                .merge(feature_frame, on=["timestamp", "asset"], how="left")
                .drop(columns=["timestamp", "asset"], errors="ignore")
            )
            if merged.empty:
                return 0.0
            split = max(1, int(len(merged) * 0.7))
            left = merged.iloc[:split]
            right = merged.iloc[split:]
            cols = [c for c in merged.columns if pd.api.types.is_numeric_dtype(merged[c])]
            if left.empty or right.empty or not cols:
                return 0.0
            return float(feature_drift_summary(left[cols], right[cols], cols)["mean_std_shift"])
        raise ValueError(f"Unsupported diagnostic type: {diagnostic_type}")

    @staticmethod
    def _project_diagnostics(
        *,
        bars: pd.DataFrame,
        folds: pd.DataFrame,
        diagnostics: pd.DataFrame,
        lag: int,
    ) -> pd.DataFrame:
        out = bars[["timestamp", "asset"]].reset_index(drop=True).copy()
        for col in diagnostics.columns:
            if col == "outer_fold_id" or not pd.api.types.is_numeric_dtype(diagnostics[col]):
                continue
            out[f"proj_{col}"] = 0.0
        if diagnostics.empty or folds.empty:
            return out
        for _, fold_row in folds.iterrows():
            fold_id = int(fold_row["outer_fold_id"])
            prev_id = fold_id - lag
            prev_diag = diagnostics[diagnostics["outer_fold_id"] == prev_id]
            if prev_diag.empty:
                continue
            # project onto this fold's test interval only; keeps causality explicit.
            ts = int(fold_row["test_start_idx"])
            te = int(fold_row["test_end_idx"])
            if ts >= len(out):
                continue
            te = min(te, len(out) - 1)
            for col in diagnostics.columns:
                if col == "outer_fold_id" or not pd.api.types.is_numeric_dtype(diagnostics[col]):
                    continue
                out_col = f"proj_{col}"
                val = float(pd.to_numeric(prev_diag[col], errors="coerce").fillna(0.0).mean())
                out.loc[ts:te, out_col] = val
        return out

    def _execute_node(
        self,
        *,
        node: WorkflowNode,
        store: ArtifactStore,
        index: ArtifactIndex,
        produced: dict[str, list[ArtifactRef]],
    ) -> None:
        if isinstance(node, DataNode):
            self._run_data_node(node=node, store=store, index=index, produced=produced)
            return
        if isinstance(node, TargetNode):
            self._run_target_node(node=node, store=store, index=index, produced=produced)
            return
        if isinstance(node, DerivedTargetNode):
            self._run_derived_target_node(node=node, store=store, index=index, produced=produced)
            return
        if isinstance(node, FoldPlanNode):
            self._run_fold_plan_node(node=node, store=store, index=index, produced=produced)
            return
        if isinstance(node, FeatureNode):
            self._run_feature_node(node=node, store=store, index=index, produced=produced)
            return
        if isinstance(node, ModelNode):
            self._run_model_node(node=node, store=store, index=index, produced=produced)
            return
        if isinstance(node, DiagnosticNode):
            self._run_diagnostic_node(node=node, store=store, index=index, produced=produced)
            return
        if isinstance(node, DerivedFeatureNode):
            self._run_derived_feature_node(node=node, store=store, index=index, produced=produced)
            return
        if isinstance(node, ProjectionNode):
            self._run_projection_node(node=node, store=store, index=index, produced=produced)
            return
        if isinstance(node, SelectionNode):
            self._run_selection_node(node=node, store=store, index=index, produced=produced)
            return
        raise TypeError(f"Unsupported node type: {type(node)!r}")

    def _run_data_node(
        self,
        *,
        node: DataNode,
        store: ArtifactStore,
        index: ArtifactIndex,
        produced: dict[str, list[ArtifactRef]],
    ) -> None:
        bars = self.data_catalog.load_bars(node.dataset_name, raw=False)
        self._write_artifact(
            store=store,
            index=index,
            produced=produced,
            node_name=node.name,
            artifact_type="BarsArtifact",
            data=bars,
            scope=ArtifactScope(cv_level="global", split_role="full"),
            schema={"columns": list(bars.columns)},
            metadata={
                "dataset_name": node.dataset_name,
                "universe": list(node.universe),
                **self._common_meta(node),
            },
            upstream_refs=[],
        )

    def _run_target_node(
        self,
        *,
        node: TargetNode,
        store: ArtifactStore,
        index: ArtifactIndex,
        produced: dict[str, list[ArtifactRef]],
    ) -> None:
        bars_ref = self._resolve_refs(list(node.depends_on), produced)[0]
        bars = store.load(bars_ref.artifact_id)
        task = self.task_registry.get(node.task_name)
        targets = self.task_registry.build_target_table(bars, task_name=task.name)
        self._write_artifact(
            store=store,
            index=index,
            produced=produced,
            node_name=node.name,
            artifact_type="TargetTableArtifact",
            data=targets,
            scope=ArtifactScope(cv_level="global", split_role="full"),
            schema={"columns": list(targets.columns)},
            metadata={
                "task": task.name,
                "horizon": task.horizon,
                "task_type": task.task_type,
                **self._common_meta(node),
            },
            upstream_refs=[bars_ref],
        )

    def _run_derived_target_node(
        self,
        *,
        node: DerivedTargetNode,
        store: ArtifactStore,
        index: ArtifactIndex,
        produced: dict[str, list[ArtifactRef]],
    ) -> None:
        inputs: DerivedTargetInputs = node.inputs
        pred_refs = (
            self._resolve_refs(inputs.prediction_tables, produced, artifact_type="PredictionArtifact")
            if inputs.prediction_tables
            else []
        )
        diag_refs = (
            self._resolve_refs(inputs.diagnostic_tables, produced, artifact_type="DiagnosticArtifact")
            if inputs.diagnostic_tables
            else []
        )
        if not pred_refs:
            raise ValueError("DerivedTargetNode requires at least one prediction table input.")
        pred = store.load(pred_refs[0].artifact_id)
        if not {"target", "prediction", "timestamp", "asset"}.issubset(pred.columns):
            raise ValueError("PredictionArtifact must contain timestamp/asset/target/prediction columns.")

        out = pred[["timestamp", "asset"]].copy()
        out["task_name"] = node.task_name
        target_kind = node.target_kind
        if target_kind == "abs_error":
            out["target"] = (pred["target"] - pred["prediction"]).abs()
        elif target_kind == "signed_residual":
            out["target"] = pred["target"] - pred["prediction"]
        elif target_kind == "disagreement":
            if len(pred_refs) < 2:
                raise ValueError("disagreement target requires >=2 prediction tables.")
            merged = out.copy()
            pred_cols: list[str] = []
            for i, ref in enumerate(pred_refs):
                frame = store.load(ref.artifact_id)[["timestamp", "asset", "prediction"]].rename(
                    columns={"prediction": f"pred_{i}"}
                )
                pred_col = f"pred_{i}"
                pred_cols.append(pred_col)
                merged = merged.merge(frame, on=["timestamp", "asset"], how="left")
            out["target"] = merged[pred_cols].std(axis=1).fillna(0.0)
        elif target_kind == "calibration_gap":
            val = 0.0
            if diag_refs:
                diag = pd.concat([store.load(r.artifact_id) for r in diag_refs], ignore_index=True)
                if "calibration_gap" in diag.columns:
                    val = float(pd.to_numeric(diag["calibration_gap"], errors="coerce").fillna(0.0).mean())
            out["target"] = val
        else:
            raise ValueError(f"Unsupported derived target kind: {target_kind}")

        out = out.dropna().reset_index(drop=True)
        self._write_artifact(
            store=store,
            index=index,
            produced=produced,
            node_name=node.name,
            artifact_type="TargetTableArtifact",
            data=out,
            scope=ArtifactScope(cv_level="outer_fold", split_role="validation"),
            schema={"columns": list(out.columns)},
            metadata={
                "task": node.task_name,
                "task_type": "regression",
                "derived_target_kind": node.target_kind,
                "source_prediction_node": node.source_prediction_node,
                **self._common_meta(node),
            },
            upstream_refs=pred_refs + diag_refs,
        )

    def _run_fold_plan_node(
        self,
        *,
        node: FoldPlanNode,
        store: ArtifactStore,
        index: ArtifactIndex,
        produced: dict[str, list[ArtifactRef]],
    ) -> None:
        bars_ref = self._resolve_refs(list(node.depends_on), produced)[0]
        bars = store.load(bars_ref.artifact_id)
        folds = build_walk_forward_splits(
            bars=bars,
            split_type=node.validation.split_type,
            min_train_size=node.validation.min_train_size,
            test_size=node.validation.test_size,
            embargo=node.validation.embargo,
            rolling_train_size=node.validation.rolling_train_size,
            mode=node.validation.mode,
        )
        self._write_artifact(
            store=store,
            index=index,
            produced=produced,
            node_name=node.name,
            artifact_type="FoldPlanArtifact",
            data=folds,
            scope=ArtifactScope(cv_level="global", split_role="validation"),
            schema={"columns": list(folds.columns)},
            metadata={
                "validation": asdict(node.validation),
                "level": node.level,
                **self._common_meta(node),
            },
            upstream_refs=[bars_ref],
        )

    def _run_feature_node(
        self,
        *,
        node: FeatureNode,
        store: ArtifactStore,
        index: ArtifactIndex,
        produced: dict[str, list[ArtifactRef]],
    ) -> None:
        bars_ref = self._resolve_refs(list(node.depends_on), produced)[0]
        bars = store.load(bars_ref.artifact_id)
        features = self.feature_registry.build(
            bars,
            FeatureSpec(family_name=node.family_name, params=node.params),
        )
        self._write_artifact(
            store=store,
            index=index,
            produced=produced,
            node_name=node.name,
            artifact_type="FeatureTableArtifact",
            data=features,
            scope=ArtifactScope(cv_level="global", split_role="full"),
            schema={"columns": list(features.columns), "family": node.family_name},
            metadata={
                "family_name": node.family_name,
                "params": node.params,
                **self._common_meta(node),
            },
            upstream_refs=[bars_ref],
        )

    def _collect_feature_table(
        self,
        *,
        inputs: ModelInputs,
        store: ArtifactStore,
        produced: dict[str, list[ArtifactRef]],
    ) -> tuple[pd.DataFrame, list[ArtifactRef]]:
        feature_refs = self._resolve_refs(inputs.feature_tables, produced, artifact_type="FeatureTableArtifact")
        features = store.load(feature_refs[0].artifact_id)
        for ref in feature_refs[1:]:
            more = store.load(ref.artifact_id)
            cols = [c for c in more.columns if c not in {"timestamp", "asset"}]
            features = features.merge(more[["timestamp", "asset"] + cols], on=["timestamp", "asset"], how="left")
        features = features.sort_values(["timestamp", "asset"]).reset_index(drop=True)
        return features, feature_refs

    def _augment_with_prediction_inputs(
        self,
        *,
        features: pd.DataFrame,
        prediction_refs: list[ArtifactRef],
        store: ArtifactStore,
    ) -> pd.DataFrame:
        out = features.copy()
        for ref in prediction_refs:
            pred = store.load(ref.artifact_id)
            test = pred[pred["split_role"] == "test"][["timestamp", "asset", "prediction"]].copy()
            col = f"pred_{ref.node_name}"
            test = test.rename(columns={"prediction": col})
            out = out.merge(test, on=["timestamp", "asset"], how="left")
            out[col] = out.groupby("asset", observed=True)[col].ffill().fillna(0.0)
        return out

    def _augment_with_diagnostic_inputs(
        self,
        *,
        features: pd.DataFrame,
        diagnostic_refs: list[ArtifactRef],
        store: ArtifactStore,
    ) -> pd.DataFrame:
        out = features.copy()
        if not diagnostic_refs:
            return out
        diag_frames = [store.load(ref.artifact_id) for ref in diagnostic_refs]
        merged = pd.concat(diag_frames, ignore_index=True)
        numeric_cols = [c for c in merged.columns if c not in {"outer_fold_id"}]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(merged[col]):
                continue
            out[f"diag_{col}"] = float(merged[col].mean())
            out[f"diag_{col}"] = out.groupby("asset", observed=True)[f"diag_{col}"].shift(1).fillna(0.0)
        return out

    def _fit_model_fold(
        self,
        *,
        model_node: ModelNode,
        model_features: pd.DataFrame,
        targets: pd.DataFrame,
        fold_row: pd.Series,
        model_kind: str,
    ) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
        merged = model_features.merge(targets[["timestamp", "asset", "target"]], on=["timestamp", "asset"], how="inner")
        merged = merged.sort_values(["timestamp", "asset"]).reset_index(drop=True)
        tr = int(fold_row["train_end_idx"])
        ts = int(fold_row["test_start_idx"])
        te = int(fold_row["test_end_idx"])
        train_df = merged.iloc[: tr + 1].copy()
        test_df = merged.iloc[ts : te + 1].copy()

        feature_cols = [c for c in merged.columns if c not in {"timestamp", "asset", "target"}]
        if not feature_cols:
            raise ValueError("No feature columns available for model fit.")
        if train_df.empty or test_df.empty:
            empty_pred = pd.DataFrame(
                columns=["timestamp", "asset", "target", "prediction", "split_role", "outer_fold_id"]
            )
            model_state: dict[str, Any] = {
                "model_name": model_node.model_spec.name,
                "algorithm": model_node.model_spec.algorithm,
                "params": model_node.model_spec.params,
                "feature_cols": feature_cols,
                "outer_fold_id": int(fold_row["outer_fold_id"]),
                "skipped": True,
            }
            diag_df = pd.DataFrame(
                [
                    {
                        "outer_fold_id": int(fold_row["outer_fold_id"]),
                        "train_metric": float("nan"),
                        "test_metric": float("nan"),
                        "train_test_deviance_gap": float("nan"),
                        "calibration_gap": 0.0,
                        "feature_drift": 0.0,
                        "residual_std_by_asset": 0.0,
                    }
                ]
            )
            snapshot = merged[["timestamp", "asset"] + feature_cols + ["target"]].copy()
            snapshot["outer_fold_id"] = int(fold_row["outer_fold_id"])
            return empty_pred, model_state, diag_df, snapshot
        x_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["target"].to_numpy(dtype=float)
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        y_test = test_df["target"].to_numpy(dtype=float)

        model = self.model_registry.build(model_node.model_spec)
        model.fit(x_train, y_train)

        if model_kind == "classification":
            train_pred = model.predict_proba(x_train)
            test_pred = model.predict_proba(x_test)
        else:
            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)

        out_train = train_df[["timestamp", "asset", "target"]].copy()
        out_train["prediction"] = train_pred
        out_train["split_role"] = "train"
        out_test = test_df[["timestamp", "asset", "target"]].copy()
        out_test["prediction"] = test_pred
        out_test["split_role"] = "test"
        pred = pd.concat([out_train, out_test], ignore_index=True)
        pred["outer_fold_id"] = int(fold_row["outer_fold_id"])

        model_state: dict[str, Any] = {
            "model_name": model_node.model_spec.name,
            "algorithm": model_node.model_spec.algorithm,
            "params": model_node.model_spec.params,
            "feature_cols": feature_cols,
            "outer_fold_id": int(fold_row["outer_fold_id"]),
        }
        coef = getattr(model, "coef_", None)
        if coef is not None:
            coef_vals = list(map(float, coef.tolist())) if hasattr(coef, "tolist") else [float(v) for v in coef]
            model_state["coef"] = coef_vals
        if hasattr(model, "params_"):
            model_state["params_"] = getattr(model, "params_")

        train_metric = float(((y_train - train_pred) ** 2).mean()) if len(y_train) else float("nan")
        test_metric = float(((y_test - test_pred) ** 2).mean()) if len(y_test) else float("nan")
        fold_diag = {
            "outer_fold_id": int(fold_row["outer_fold_id"]),
            "train_metric": train_metric,
            "test_metric": test_metric,
            "train_test_deviance_gap": float(test_metric - train_metric),
            "calibration_gap": calibration_gap(y_test, test_pred) if model_kind == "classification" else 0.0,
            "feature_drift": feature_drift_summary(
                train_df[feature_cols],
                test_df[feature_cols],
                feature_cols,
            )["mean_std_shift"],
        }
        resid = residual_std_by_asset(out_test)
        fold_diag["residual_std_by_asset"] = (
            float(resid["residual_std_by_asset"].mean()) if not resid.empty else 0.0
        )
        diag_df = pd.DataFrame([fold_diag])

        snapshot = merged[["timestamp", "asset"] + feature_cols + ["target"]].copy()
        snapshot["outer_fold_id"] = int(fold_row["outer_fold_id"])
        return pred, model_state, diag_df, snapshot

    def _run_model_node(
        self,
        *,
        node: ModelNode,
        store: ArtifactStore,
        index: ArtifactIndex,
        produced: dict[str, list[ArtifactRef]],
    ) -> None:
        inputs = node.inputs
        features, feature_refs = self._collect_feature_table(inputs=inputs, store=store, produced=produced)
        target_ref = self._resolve_refs(inputs.target_tables, produced, artifact_type="TargetTableArtifact")[0]
        fold_ref = self._resolve_refs(inputs.fold_plans, produced, artifact_type="FoldPlanArtifact")[0]
        targets = store.load(target_ref.artifact_id)
        folds = store.load(fold_ref.artifact_id)

        prediction_refs = (
            self._resolve_refs(inputs.prediction_tables, produced, artifact_type="PredictionArtifact")
            if inputs.prediction_tables
            else []
        )
        diagnostic_refs = (
            self._resolve_refs(inputs.diagnostic_tables, produced, artifact_type="DiagnosticArtifact")
            if inputs.diagnostic_tables
            else []
        )
        selection_refs = (
            self._resolve_refs(inputs.selection_tables, produced, artifact_type="SelectionArtifact")
            if inputs.selection_tables
            else []
        )

        if prediction_refs:
            features = self._augment_with_prediction_inputs(
                features=features,
                prediction_refs=prediction_refs,
                store=store,
            )
        if diagnostic_refs:
            features = self._augment_with_diagnostic_inputs(
                features=features,
                diagnostic_refs=diagnostic_refs,
                store=store,
            )

        task_name = str(targets["task_name"].iloc[0]) if "task_name" in targets.columns else ""
        task_type_from_meta = str(target_ref.metadata.get("task_type", "")).lower()
        if task_type_from_meta in {"classification", "regression"}:
            model_kind = task_type_from_meta
        else:
            task = None
            if task_name:
                try:
                    task = self.task_registry.get(task_name)
                except KeyError:
                    task = None
            model_kind = "classification" if task and task.task_type == "classification" else "regression"

        all_pred: list[pd.DataFrame] = []
        all_diag: list[pd.DataFrame] = []
        states: list[dict[str, Any]] = []
        snapshots: list[pd.DataFrame] = []
        for _, fold_row in folds.iterrows():
            pred, state, diag, snap = self._fit_model_fold(
                model_node=node,
                model_features=features,
                targets=targets,
                fold_row=fold_row,
                model_kind=model_kind,
            )
            all_pred.append(pred)
            all_diag.append(diag)
            states.append(state)
            if node.training_recipe.save_training_snapshot:
                snapshots.append(snap)

        non_empty_pred = [p for p in all_pred if not p.empty]
        if non_empty_pred:
            pred_df = pd.concat(non_empty_pred, ignore_index=True)
        else:
            pred_df = pd.DataFrame(
                columns=["timestamp", "asset", "target", "prediction", "split_role", "outer_fold_id"]
            )
        diag_df = pd.concat(all_diag, ignore_index=True)
        state_df = pd.DataFrame(states)
        if len(all_pred) >= 2:
            pivot = []
            for frame in all_pred:
                if frame.empty:
                    continue
                fold_id = int(frame["outer_fold_id"].iloc[0])
                part = frame[frame["split_role"] == "test"][["timestamp", "asset", "prediction"]].copy()
                if part.empty:
                    continue
                part = part.groupby(["timestamp", "asset"], as_index=False)["prediction"].mean()
                part = part.rename(columns={"prediction": f"pred_{fold_id:03d}"})
                pivot.append(part.set_index(["timestamp", "asset"]))
            if len(pivot) >= 2:
                dis = pd.concat(pivot, axis=1).reset_index()
                pred_cols = [c for c in dis.columns if c.startswith("pred_")]
                if len(pred_cols) >= 2:
                    diag_df["prediction_disagreement"] = float(prediction_disagreement(dis[pred_cols]))
        coef_dicts = [c for c in state_df.get("coef_map", pd.Series(dtype=object)).tolist() if isinstance(c, dict)]
        diag_df["coefficient_stability"] = coefficient_stability(coef_dicts)

        upstream = feature_refs + [target_ref, fold_ref] + prediction_refs + diagnostic_refs + selection_refs
        pred_ref = self._write_artifact(
            store=store,
            index=index,
            produced=produced,
            node_name=node.name,
            artifact_type="PredictionArtifact",
            data=pred_df,
            scope=ArtifactScope(cv_level="outer_fold", split_role="oof"),
            schema={"columns": list(pred_df.columns)},
            metadata={
                "model_name": node.model_spec.name,
                "algorithm": node.model_spec.algorithm,
                **self._common_meta(node),
            },
            upstream_refs=upstream,
        )
        self._write_artifact(
            store=store,
            index=index,
            produced=produced,
            node_name=node.name,
            artifact_type="ModelStateArtifact",
            data=state_df,
            scope=ArtifactScope(cv_level="outer_fold", split_role="train"),
            schema={"columns": list(state_df.columns)},
            metadata={
                "model_name": node.model_spec.name,
                "algorithm": node.model_spec.algorithm,
                **self._common_meta(node),
            },
            upstream_refs=upstream,
            suffix="state",
        )
        self._write_artifact(
            store=store,
            index=index,
            produced=produced,
            node_name=node.name,
            artifact_type="DiagnosticArtifact",
            data=diag_df,
            scope=ArtifactScope(cv_level="outer_fold", split_role="validation"),
            schema={"columns": list(diag_df.columns)},
            metadata={
                "source_model": node.model_spec.name,
                **self._common_meta(node),
            },
            upstream_refs=[pred_ref],
            suffix="diag",
        )
        if node.training_recipe.save_training_snapshot and snapshots:
            snap_df = pd.concat(snapshots, ignore_index=True)
            self._write_artifact(
                store=store,
                index=index,
                produced=produced,
                node_name=node.name,
                artifact_type="TrainingDatasetSnapshotArtifact",
                data=snap_df,
                scope=ArtifactScope(cv_level="outer_fold", split_role="train"),
                schema={"columns": list(snap_df.columns)},
                metadata={
                    "source_model": node.model_spec.name,
                    **self._common_meta(node),
                },
                upstream_refs=upstream,
                suffix="snapshot",
            )

    def _run_diagnostic_node(
        self,
        *,
        node: DiagnosticNode,
        store: ArtifactStore,
        index: ArtifactIndex,
        produced: dict[str, list[ArtifactRef]],
    ) -> None:
        inputs: DiagnosticInputs = node.inputs
        pred_refs = (
            self._resolve_refs(inputs.prediction_tables, produced, artifact_type="PredictionArtifact")
            if inputs.prediction_tables
            else []
        )
        target_refs = (
            self._resolve_refs(inputs.target_tables, produced, artifact_type="TargetTableArtifact")
            if inputs.target_tables
            else []
        )
        feature_refs = (
            self._resolve_refs(inputs.feature_tables, produced, artifact_type="FeatureTableArtifact")
            if inputs.feature_tables
            else []
        )
        state_refs = (
            self._resolve_refs(inputs.model_state_tables, produced, artifact_type="ModelStateArtifact")
            if inputs.model_state_tables
            else []
        )
        upstream = pred_refs + target_refs + feature_refs + state_refs

        rows: list[dict[str, float]] = []
        if pred_refs:
            pred = store.load(pred_refs[0].artifact_id)
            feature_frame = store.load(feature_refs[0].artifact_id) if feature_refs else None
            for fold_id, fold_df in pred.groupby("outer_fold_id", observed=True):
                row: dict[str, float] = {"outer_fold_id": float(fold_id)}
                for diagnostic_type in node.diagnostic_types:
                    row[diagnostic_type] = self._diag_metric(
                        diagnostic_type,
                        pred_fold=fold_df,
                        pred_all=pred,
                        feature_frame=feature_frame,
                    )
                rows.append(row)

        out = pd.DataFrame(rows or [{"outer_fold_id": -1.0}])
        self._write_artifact(
            store=store,
            index=index,
            produced=produced,
            node_name=node.name,
            artifact_type="DiagnosticArtifact",
            data=out,
            scope=ArtifactScope(cv_level="outer_fold", split_role="validation"),
            schema={"columns": list(out.columns)},
            metadata={
                "diagnostic_types": list(node.diagnostic_types),
                **self._common_meta(node),
            },
            upstream_refs=upstream,
        )

    def _run_derived_feature_node(
        self,
        *,
        node: DerivedFeatureNode,
        store: ArtifactStore,
        index: ArtifactIndex,
        produced: dict[str, list[ArtifactRef]],
    ) -> None:
        inputs: DerivedFeatureInputs = node.inputs
        diag_refs = (
            self._resolve_refs(inputs.diagnostic_tables, produced, artifact_type="DiagnosticArtifact")
            if inputs.diagnostic_tables
            else []
        )
        pred_refs = (
            self._resolve_refs(inputs.prediction_tables, produced, artifact_type="PredictionArtifact")
            if inputs.prediction_tables
            else []
        )
        feat_refs = (
            self._resolve_refs(inputs.feature_tables, produced, artifact_type="FeatureTableArtifact")
            if inputs.feature_tables
            else []
        )
        upstream = diag_refs + pred_refs + feat_refs
        if not upstream:
            raise ValueError("DerivedFeatureNode requires at least one input artifact.")
        frames = [store.load(r.artifact_id) for r in upstream]
        merged = pd.concat(frames, ignore_index=True, sort=False)
        out = pd.DataFrame({
            "timestamp": pd.date_range("2000-01-01", periods=len(merged), freq="D", tz="UTC"),
            "asset": ["GLOBAL"] * len(merged),
        })
        numeric_cols = [c for c in merged.columns if pd.api.types.is_numeric_dtype(merged[c])]
        for col in numeric_cols:
            out[f"derived_{col}"] = merged[col].rolling(window=2, min_periods=1).mean().fillna(0.0)
        self._write_artifact(
            store=store,
            index=index,
            produced=produced,
            node_name=node.name,
            artifact_type="FeatureTableArtifact",
            data=out,
            scope=ArtifactScope(cv_level="outer_fold", split_role="validation"),
            schema={"columns": list(out.columns)},
            metadata={
                "derived_expressions": node.derived_expressions,
                **self._common_meta(node),
            },
            upstream_refs=upstream,
        )

    def _run_projection_node(
        self,
        *,
        node: ProjectionNode,
        store: ArtifactStore,
        index: ArtifactIndex,
        produced: dict[str, list[ArtifactRef]],
    ) -> None:
        inputs: ProjectionInputs = node.inputs
        bars_ref = self._resolve_refs(inputs.bars, produced, artifact_type="BarsArtifact")[0]
        diag_refs = self._resolve_refs(inputs.diagnostic_tables, produced, artifact_type="DiagnosticArtifact")
        fold_refs = (
            self._resolve_refs(inputs.fold_plans, produced, artifact_type="FoldPlanArtifact")
            if inputs.fold_plans
            else []
        )
        bars = store.load(bars_ref.artifact_id)
        diag = pd.concat([store.load(r.artifact_id) for r in diag_refs], ignore_index=True)
        folds = store.load(fold_refs[0].artifact_id) if fold_refs else pd.DataFrame()

        if inputs.mode == "prev_fold_to_test" and not folds.empty and "outer_fold_id" in diag.columns:
            out = self._project_diagnostics(bars=bars, folds=folds, diagnostics=diag, lag=inputs.lag)
        else:
            out = bars[["timestamp", "asset"]].copy()
            for col in diag.columns:
                if col == "outer_fold_id" or not pd.api.types.is_numeric_dtype(diag[col]):
                    continue
                fname = f"proj_{col}"
                out[fname] = float(pd.to_numeric(diag[col], errors="coerce").fillna(0.0).mean())
                out[fname] = out.groupby("asset", observed=True)[fname].shift(inputs.lag).fillna(0.0)

        self._write_artifact(
            store=store,
            index=index,
            produced=produced,
            node_name=node.name,
            artifact_type="FeatureTableArtifact",
            data=out,
            scope=ArtifactScope(cv_level="outer_fold", split_role="validation"),
            schema={"columns": list(out.columns)},
            metadata={
                "lag": inputs.lag,
                "mode": inputs.mode,
                **self._common_meta(node),
            },
            upstream_refs=[bars_ref, *diag_refs, *fold_refs],
        )

    def _run_selection_node(
        self,
        *,
        node: SelectionNode,
        store: ArtifactStore,
        index: ArtifactIndex,
        produced: dict[str, list[ArtifactRef]],
    ) -> None:
        inputs: SelectionInputs = node.inputs
        pred_refs = self._resolve_refs(inputs.prediction_tables, produced, artifact_type="PredictionArtifact")
        candidates: list[dict[str, Any]] = []
        for ref in pred_refs:
            pred = store.load(ref.artifact_id)
            test = pred[pred["split_role"] == "test"]
            mse = float(((test["target"] - test["prediction"]) ** 2).mean()) if not test.empty else float("inf")
            candidates.append({"artifact_id": ref.artifact_id, "node_name": ref.node_name, "metric_mse": mse})
        comp = pd.DataFrame(candidates).sort_values("metric_mse", ascending=True).reset_index(drop=True)

        if node.strategy_name == "best_recent_model":
            chosen = comp.iloc[0]
            out = pd.DataFrame([{
                "strategy": node.strategy_name,
                "chosen_prediction_artifact": chosen["artifact_id"],
                "chosen_node_name": chosen["node_name"],
                "metric_mse": float(chosen["metric_mse"]),
            }])
        elif node.strategy_name == "weighted_blend":
            inv = 1.0 / (comp["metric_mse"] + 1e-9)
            weights = inv / inv.sum()
            out = comp.copy()
            out["weight"] = weights
            out.insert(0, "strategy", node.strategy_name)
        else:
            out = pd.DataFrame([{"strategy": node.strategy_name, "note": "diagnostic-aware selection scaffold"}])

        self._write_artifact(
            store=store,
            index=index,
            produced=produced,
            node_name=node.name,
            artifact_type="SelectionArtifact",
            data=out,
            scope=ArtifactScope(cv_level="outer_fold", split_role="validation"),
            schema={"columns": list(out.columns)},
            metadata={"strategy": node.strategy_name, **self._common_meta(node)},
            upstream_refs=pred_refs,
        )


def run_from_recipe(recipe: Any, *, run_config: dict[str, Any], runs_dir: Path | str = "runs") -> str:
    graph = recipe.compile()
    runner = WorkflowRunner(runs_dir=runs_dir)
    return runner.run(graph, run_config=run_config)


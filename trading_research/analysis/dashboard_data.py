"""Run-level analytics tables for quant dashboard views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.workflow.artifacts import ArtifactRef
from trading_research.workflow.inspectors import RunInspector, load_run


@dataclass(frozen=True)
class RunAnalyticsBundle:
    run_id: str
    runs_root: Path
    nodes: list[str]
    artifacts: list[ArtifactRef]
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame
    selections: pd.DataFrame
    model_states: pd.DataFrame
    snapshots: pd.DataFrame
    flow_edges: pd.DataFrame
    quant_recommendations: pd.DataFrame
    handoff_payload: dict[str, Any]


def _concat_with_source(frames: list[tuple[str, str, pd.DataFrame]]) -> pd.DataFrame:
    out: list[pd.DataFrame] = []
    for node_name, artifact_type, frame in frames:
        if frame is None or frame.empty:
            continue
        copy = frame.copy()
        copy["node_name"] = node_name
        copy["artifact_type"] = artifact_type
        out.append(copy)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True, sort=False)


def _artifact_table(inspector: RunInspector, artifact_type: str) -> pd.DataFrame:
    refs = inspector.artifact_store.index.list_artifacts(artifact_type=artifact_type)
    rows: list[tuple[str, str, pd.DataFrame]] = []
    for ref in refs:
        try:
            frame = inspector.artifact_store.load(ref.artifact_id)
        except Exception:
            continue
        rows.append((ref.node_name, artifact_type, frame))
    return _concat_with_source(rows)


def _flow_edges(inspector: RunInspector) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for ref in inspector.list_artifacts():
        for upstream_id in ref.upstream_ids:
            upstream = next((r for r in inspector.list_artifacts() if r.artifact_id == upstream_id), None)
            if upstream is None:
                continue
            rows.append(
                {
                    "from_node": upstream.node_name,
                    "from_artifact": str(upstream.artifact_type),
                    "to_node": ref.node_name,
                    "to_artifact": str(ref.artifact_type),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["from_node", "from_artifact", "to_node", "to_artifact"])
    return pd.DataFrame(rows).drop_duplicates(ignore_index=True)


def _quant_recommendations(predictions: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    recs: list[dict[str, Any]] = []
    if not predictions.empty and {"target", "prediction"}.issubset(predictions.columns):
        err = (predictions["target"] - predictions["prediction"]).abs()
        recs.append(
            {
                "priority": "high",
                "signal": "overall_abs_error",
                "value": float(err.mean()),
                "recommendation": "Review features with high residual concentration and consider robust transforms.",
            }
        )
        if "asset" in predictions.columns:
            by_asset = (
                predictions.assign(abs_err=err)
                .groupby("asset", observed=True)["abs_err"]
                .mean()
                .sort_values(ascending=False)
            )
            if not by_asset.empty:
                top_asset = str(by_asset.index[0])
                recs.append(
                    {
                        "priority": "medium",
                        "signal": "asset_error_hotspot",
                        "value": float(by_asset.iloc[0]),
                        "recommendation": f"Investigate asset-level instability for {top_asset}; feature drift likely.",
                    }
                )

    if not diagnostics.empty:
        for col, threshold, msg in [
            ("train_test_deviance_gap", 0.01, "Potential overfit; increase regularization or simplify features."),
            ("feature_drift", 0.5, "Feature drift elevated; inspect regime-specific model behavior."),
            ("coefficient_stability", 0.25, "Coefficient instability; consider smoother model class."),
        ]:
            if col in diagnostics.columns:
                val = float(pd.to_numeric(diagnostics[col], errors="coerce").fillna(0.0).mean())
                if val > threshold:
                    recs.append(
                        {
                            "priority": "high",
                            "signal": col,
                            "value": val,
                            "recommendation": msg,
                        }
                    )

    if not recs:
        recs.append(
            {
                "priority": "low",
                "signal": "no_major_alert",
                "value": 0.0,
                "recommendation": "No major alerts detected. Iterate on alpha ideas and run targeted ablations.",
            }
        )
    return pd.DataFrame(recs)


def _handoff_payload(bundle: RunAnalyticsBundle) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": bundle.run_id,
        "nodes": bundle.nodes,
        "artifacts_count": len(bundle.artifacts),
        "top_recommendations": bundle.quant_recommendations.head(10).to_dict(orient="records"),
    }
    if not bundle.predictions.empty and {"target", "prediction"}.issubset(bundle.predictions.columns):
        payload["prediction_summary"] = {
            "rows": int(len(bundle.predictions)),
            "abs_error_mean": float((bundle.predictions["target"] - bundle.predictions["prediction"]).abs().mean()),
        }
    if not bundle.diagnostics.empty:
        payload["diagnostics_columns"] = [str(c) for c in bundle.diagnostics.columns]
    return payload


def load_run_analytics(run_id: str, runs_root: Path | str = "runs") -> RunAnalyticsBundle:
    root = Path(runs_root)
    inspector = load_run(run_id, root)
    predictions = _artifact_table(inspector, "PredictionArtifact")
    diagnostics = _artifact_table(inspector, "DiagnosticArtifact")
    selections = _artifact_table(inspector, "SelectionArtifact")
    states = _artifact_table(inspector, "ModelStateArtifact")
    snapshots = _artifact_table(inspector, "TrainingDatasetSnapshotArtifact")
    flow = _flow_edges(inspector)
    recs = _quant_recommendations(predictions, diagnostics)

    bundle = RunAnalyticsBundle(
        run_id=run_id,
        runs_root=root,
        nodes=inspector.list_nodes(),
        artifacts=inspector.list_artifacts(),
        predictions=predictions,
        diagnostics=diagnostics,
        selections=selections,
        model_states=states,
        snapshots=snapshots,
        flow_edges=flow,
        quant_recommendations=recs,
        handoff_payload={},
    )
    object.__setattr__(bundle, "handoff_payload", _handoff_payload(bundle))
    return bundle


def discover_runs(runs_root: Path | str = "runs") -> list[str]:
    root = Path(runs_root)
    if not root.exists():
        return []
    manifests = sorted(root.glob("**/manifest.json"))
    run_ids: list[str] = []
    for manifest in manifests:
        run_ids.append(manifest.parent.name)
    return sorted(set(run_ids))

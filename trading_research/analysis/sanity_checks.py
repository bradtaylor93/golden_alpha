"""Run-level leakage and split integrity sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.workflow.inspectors import load_run


@dataclass(frozen=True)
class SanityCheckResult:
    check_name: str
    node_name: str
    artifact_type: str
    severity: str
    passed: bool
    value: float
    details: str

    def to_row(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "node_name": self.node_name,
            "artifact_type": self.artifact_type,
            "severity": self.severity,
            "passed": self.passed,
            "value": self.value,
            "details": self.details,
        }


def _prediction_integrity_checks(node_name: str, frame: pd.DataFrame) -> list[SanityCheckResult]:
    results: list[SanityCheckResult] = []
    artifact_type = "PredictionArtifact"
    required_cols = {"timestamp", "asset", "split_role", "outer_fold_id", "target", "prediction"}
    missing = sorted(required_cols.difference(frame.columns))
    results.append(
        SanityCheckResult(
            check_name="prediction_required_columns",
            node_name=node_name,
            artifact_type=artifact_type,
            severity="high",
            passed=not missing,
            value=float(len(missing)),
            details=f"missing={missing}",
        )
    )
    if missing:
        return results

    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["asset"] = df["asset"].astype(str)

    dup_cols = ["timestamp", "asset", "split_role", "outer_fold_id"]
    duplicate_count = int(df.duplicated(subset=dup_cols).sum())
    results.append(
        SanityCheckResult(
            check_name="prediction_duplicate_rows",
            node_name=node_name,
            artifact_type=artifact_type,
            severity="high",
            passed=duplicate_count == 0,
            value=float(duplicate_count),
            details=f"dedupe_key={dup_cols}",
        )
    )

    overlap_count = 0
    temporal_violations = 0
    per_fold_ranges: list[str] = []
    grouped = df.groupby("outer_fold_id", observed=True)
    for fold_id, fold_df in grouped:
        train = fold_df[fold_df["split_role"] == "train"]
        test = fold_df[fold_df["split_role"] == "test"]
        train_keys = set(zip(train["timestamp"], train["asset"], strict=False))
        test_keys = set(zip(test["timestamp"], test["asset"], strict=False))
        overlap_count += len(train_keys.intersection(test_keys))

        if not train.empty and not test.empty:
            train_max = train["timestamp"].max()
            test_min = test["timestamp"].min()
            per_fold_ranges.append(f"fold={int(fold_id)} train_max={train_max} test_min={test_min}")
            if pd.notna(train_max) and pd.notna(test_min) and train_max >= test_min:
                temporal_violations += 1

    results.append(
        SanityCheckResult(
            check_name="prediction_train_test_key_overlap",
            node_name=node_name,
            artifact_type=artifact_type,
            severity="critical",
            passed=overlap_count == 0,
            value=float(overlap_count),
            details="overlap on (timestamp, asset) within fold",
        )
    )
    results.append(
        SanityCheckResult(
            check_name="prediction_fold_temporal_order",
            node_name=node_name,
            artifact_type=artifact_type,
            severity="critical",
            passed=temporal_violations == 0,
            value=float(temporal_violations),
            details="; ".join(per_fold_ranges[:8]),
        )
    )

    test_df = df[df["split_role"] == "test"].sort_values(["outer_fold_id", "timestamp", "asset"])
    if not test_df.empty:
        per_fold_test_start = (
            test_df.groupby("outer_fold_id", observed=True)["timestamp"].min().sort_index().tolist()
        )
        monotonic = all(
            per_fold_test_start[i] <= per_fold_test_start[i + 1]
            for i in range(len(per_fold_test_start) - 1)
        )
        results.append(
            SanityCheckResult(
                check_name="prediction_test_windows_monotonic",
                node_name=node_name,
                artifact_type=artifact_type,
                severity="medium",
                passed=monotonic,
                value=0.0 if monotonic else 1.0,
                details=f"starts={per_fold_test_start[:8]}",
            )
        )
    return results


def _feature_duplicate_checks(node_name: str, frame: pd.DataFrame) -> list[SanityCheckResult]:
    results: list[SanityCheckResult] = []
    artifact_type = "FeatureTableArtifact"
    if not {"timestamp", "asset"}.issubset(frame.columns):
        results.append(
            SanityCheckResult(
                check_name="feature_required_columns",
                node_name=node_name,
                artifact_type=artifact_type,
                severity="high",
                passed=False,
                value=1.0,
                details="missing timestamp/asset",
            )
        )
        return results

    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["asset"] = df["asset"].astype(str)
    duplicate_count = int(df.duplicated(subset=["timestamp", "asset"]).sum())
    results.append(
        SanityCheckResult(
            check_name="feature_duplicate_rows",
            node_name=node_name,
            artifact_type=artifact_type,
            severity="high",
            passed=duplicate_count == 0,
            value=float(duplicate_count),
            details="dedupe_key=['timestamp','asset']",
        )
    )
    return results


def _projection_prev_fold_checks(
    *,
    node_name: str,
    projection_frame: pd.DataFrame,
    bars_frame: pd.DataFrame,
    folds_frame: pd.DataFrame,
    diagnostics_frame: pd.DataFrame,
    lag: int,
) -> list[SanityCheckResult]:
    results: list[SanityCheckResult] = []
    artifact_type = "FeatureTableArtifact"
    proj_cols = [c for c in projection_frame.columns if c.startswith("proj_")]
    if not proj_cols:
        results.append(
            SanityCheckResult(
                check_name="projection_has_proj_columns",
                node_name=node_name,
                artifact_type=artifact_type,
                severity="high",
                passed=False,
                value=0.0,
                details="no proj_* columns present",
            )
        )
        return results

    out = projection_frame.reset_index(drop=True).copy()
    bars = bars_frame.reset_index(drop=True).copy()
    folds = folds_frame.copy()
    diagnostics = diagnostics_frame.copy()
    violations = 0
    checked = 0

    for _, fold_row in folds.iterrows():
        fold_id = int(fold_row["outer_fold_id"])
        prev_id = fold_id - lag
        prev_diag = diagnostics[diagnostics["outer_fold_id"] == prev_id]
        ts = int(fold_row["test_start_idx"])
        te = int(fold_row["test_end_idx"])
        if ts >= len(out):
            continue
        te = min(te, len(out) - 1)
        test_slice = out.iloc[ts : te + 1]
        if test_slice.empty:
            continue
        checked += 1
        for col in proj_cols:
            source_col = col.removeprefix("proj_")
            if source_col not in diagnostics.columns:
                continue
            expected = (
                float(pd.to_numeric(prev_diag[source_col], errors="coerce").fillna(0.0).mean())
                if not prev_diag.empty
                else 0.0
            )
            observed = pd.to_numeric(test_slice[col], errors="coerce").fillna(0.0)
            # projection currently writes constant value across fold test interval.
            if not ((observed - expected).abs() <= 1e-9).all():
                violations += 1
                break

    results.append(
        SanityCheckResult(
            check_name="projection_prev_fold_mapping",
            node_name=node_name,
            artifact_type=artifact_type,
            severity="critical",
            passed=violations == 0,
            value=float(violations),
            details=f"checked_folds={checked}, lag={lag}",
        )
    )

    # Additional alignment check against bars cardinality.
    results.append(
        SanityCheckResult(
            check_name="projection_row_count_matches_bars",
            node_name=node_name,
            artifact_type=artifact_type,
            severity="high",
            passed=len(out) == len(bars),
            value=float(abs(len(out) - len(bars))),
            details=f"projection_rows={len(out)} bars_rows={len(bars)}",
        )
    )
    return results


def run_sanity_checks(run_id: str, runs_root: Path | str) -> pd.DataFrame:
    """Generate strict no-leakage/integrity report for a run."""
    inspector = load_run(run_id, runs_root)
    refs = inspector.list_artifacts()
    by_id = {r.artifact_id: r for r in refs}
    rows: list[SanityCheckResult] = []

    # Core checks on prediction and feature artifacts.
    for ref in refs:
        artifact_type = str(ref.artifact_type.value if hasattr(ref.artifact_type, "value") else ref.artifact_type)
        frame = inspector.artifact_store.load(ref.artifact_id)
        if artifact_type == "PredictionArtifact":
            rows.extend(_prediction_integrity_checks(ref.node_name, frame))
        if artifact_type == "FeatureTableArtifact":
            rows.extend(_feature_duplicate_checks(ref.node_name, frame))

    # Projection-specific leakage assertions (prev fold diagnostics -> current fold test).
    for ref in refs:
        artifact_type = str(ref.artifact_type.value if hasattr(ref.artifact_type, "value") else ref.artifact_type)
        if artifact_type != "FeatureTableArtifact":
            continue
        metadata = ref.metadata if isinstance(ref.metadata, dict) else {}
        node_type = str(metadata.get("node_type", ""))
        mode = str(metadata.get("mode", ""))
        if node_type != "ProjectionNode" or mode != "prev_fold_to_test":
            continue

        # Projection upstream should include bars + folds + diagnostics.
        up_refs = [by_id[u] for u in ref.upstream_ids if u in by_id]
        bars_ref = next(
            (
                r
                for r in up_refs
                if str(r.artifact_type.value if hasattr(r.artifact_type, "value") else r.artifact_type)
                == "BarsArtifact"
            ),
            None,
        )
        folds_ref = next(
            (
                r
                for r in up_refs
                if str(r.artifact_type.value if hasattr(r.artifact_type, "value") else r.artifact_type)
                == "FoldPlanArtifact"
            ),
            None,
        )
        diag_refs = [
            r
            for r in up_refs
            if str(r.artifact_type.value if hasattr(r.artifact_type, "value") else r.artifact_type)
            == "DiagnosticArtifact"
        ]
        if bars_ref is None or folds_ref is None or not diag_refs:
            rows.append(
                SanityCheckResult(
                    check_name="projection_required_upstream",
                    node_name=ref.node_name,
                    artifact_type="FeatureTableArtifact",
                    severity="critical",
                    passed=False,
                    value=1.0,
                    details="missing bars/folds/diagnostic upstream refs",
                )
            )
            continue

        projection_frame = inspector.artifact_store.load(ref.artifact_id)
        bars_frame = inspector.artifact_store.load(bars_ref.artifact_id)
        folds_frame = inspector.artifact_store.load(folds_ref.artifact_id)
        diagnostics_frame = pd.concat(
            [inspector.artifact_store.load(d.artifact_id) for d in diag_refs],
            ignore_index=True,
            sort=False,
        )
        lag = int(metadata.get("lag", 1))
        rows.extend(
            _projection_prev_fold_checks(
                node_name=ref.node_name,
                projection_frame=projection_frame,
                bars_frame=bars_frame,
                folds_frame=folds_frame,
                diagnostics_frame=diagnostics_frame,
                lag=lag,
            )
        )

    report = pd.DataFrame([r.to_row() for r in rows])
    if report.empty:
        return pd.DataFrame(
            columns=[
                "check_name",
                "node_name",
                "artifact_type",
                "severity",
                "passed",
                "value",
                "details",
            ]
        )
    return report.sort_values(["passed", "severity", "node_name", "check_name"]).reset_index(drop=True)


def write_sanity_report(
    *,
    run_id: str,
    runs_root: Path | str,
    output_csv: Path | str,
) -> pd.DataFrame:
    report = run_sanity_checks(run_id=run_id, runs_root=runs_root)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    return report


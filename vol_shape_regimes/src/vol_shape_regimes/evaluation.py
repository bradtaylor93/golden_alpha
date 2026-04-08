"""Evaluation, diagnostics, and interpretable summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


def evaluate_state_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None = None,
    top_k: int = 2,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if probs is not None:
        p = np.asarray(probs, dtype=float)
        try:
            metrics["log_loss"] = float(log_loss(y_true, p, labels=np.arange(p.shape[1])))
        except ValueError:
            metrics["log_loss"] = float("nan")
        if top_k > 1 and p.shape[1] >= top_k:
            top = np.argsort(p, axis=1)[:, ::-1][:, :top_k]
            hit = np.any(top == y_true[:, None], axis=1)
            metrics[f"top_{top_k}_accuracy"] = float(np.mean(hit))
    else:
        metrics["log_loss"] = float("nan")

    labels = np.arange(max(np.max(y_true), np.max(y_pred)) + 1)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pr, rc, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics["confusion_matrix"] = cm.tolist()
    metrics["per_state"] = [
        {
            "state": int(s),
            "precision": float(pr[i]),
            "recall": float(rc[i]),
            "f1": float(f1[i]),
            "support": int(supp[i]),
        }
        for i, s in enumerate(labels)
    ]
    return metrics


def dwell_time_by_state(states: np.ndarray) -> pd.DataFrame:
    seq = np.asarray(states, dtype=int)
    if len(seq) == 0:
        return pd.DataFrame(columns=["state", "dwell_len"])
    rows: list[dict[str, int]] = []
    cur = int(seq[0])
    run = 1
    for v in seq[1:]:
        if int(v) == cur:
            run += 1
        else:
            rows.append({"state": cur, "dwell_len": run})
            cur = int(v)
            run = 1
    rows.append({"state": cur, "dwell_len": run})
    return pd.DataFrame(rows)


def transition_out_probabilities(matrix: np.ndarray) -> pd.DataFrame:
    k = matrix.shape[0]
    rows: list[dict[str, float | int]] = []
    for s in range(k):
        row: dict[str, float | int] = {"state": s}
        for j in range(k):
            row[f"p_to_{j}"] = float(matrix[s, j])
        rows.append(row)
    return pd.DataFrame(rows)


def interpret_state_label(
    avg_vol: float,
    avg_future_vol: float,
    avg_slope: float,
    avg_spike_count: float,
    global_vol_quantiles: tuple[float, float],
) -> str:
    low_q, high_q = global_vol_quantiles
    if avg_vol <= low_q:
        level = "low"
    elif avg_vol >= high_q:
        level = "high"
    else:
        level = "medium"
    trend = "rising" if avg_slope > 0 else "falling"
    if abs(avg_slope) < 1e-5:
        trend = "flat"
    texture = "clustered" if avg_spike_count >= 3 else "smooth"
    direction = "escalation" if avg_future_vol > avg_vol else "decay"
    return f"{level} {texture} vol, {trend}, {direction}"


def summarize_clusters(
    feature_frame: pd.DataFrame,
    states: np.ndarray,
    dates: pd.Series,
    vol_series_aligned: pd.Series,
    abs_returns_aligned: pd.Series,
    transition_matrix: np.ndarray,
) -> pd.DataFrame:
    frame = feature_frame.copy()
    frame["state"] = np.asarray(states, dtype=int)
    frame["date"] = pd.to_datetime(dates)
    frame["vol_t"] = pd.to_numeric(vol_series_aligned, errors="coerce").to_numpy()
    frame["future_vol"] = frame["vol_t"].shift(-1)
    frame["future_abs_ret"] = pd.to_numeric(abs_returns_aligned, errors="coerce").shift(-1).to_numpy()

    vol_q = (float(frame["vol_t"].quantile(0.33)), float(frame["vol_t"].quantile(0.67)))
    out_rows: list[dict[str, float | int | str]] = []
    for s, grp in frame.groupby("state", sort=True):
        slope_col = "sum_trend_slope" if "sum_trend_slope" in grp.columns else None
        spike_col = "sum_spike_count" if "sum_spike_count" in grp.columns else None
        avg_slope = float(grp[slope_col].mean()) if slope_col else 0.0
        avg_spike = float(grp[spike_col].mean()) if spike_col else 0.0
        label = interpret_state_label(
            avg_vol=float(grp["vol_t"].mean()),
            avg_future_vol=float(grp["future_vol"].mean()),
            avg_slope=avg_slope,
            avg_spike_count=avg_spike,
            global_vol_quantiles=vol_q,
        )
        row: dict[str, float | int | str] = {
            "state": int(s),
            "size": int(len(grp)),
            "prevalence": float(len(grp) / len(frame)),
            "avg_vol_t": float(grp["vol_t"].mean()),
            "avg_future_vol": float(grp["future_vol"].mean()),
            "avg_future_abs_return": float(grp["future_abs_ret"].mean()),
            "avg_slope": avg_slope,
            "avg_spike_count": avg_spike,
            "interpretation": label,
        }
        for j in range(transition_matrix.shape[1]):
            row[f"p_to_{j}"] = float(transition_matrix[int(s), j])
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values("state").reset_index(drop=True)

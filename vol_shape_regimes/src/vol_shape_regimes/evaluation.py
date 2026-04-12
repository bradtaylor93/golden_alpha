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
    roc_auc_score,
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


def change_event_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_state: np.ndarray,
    n_states: int,
) -> dict[str, Any]:
    """Evaluate predictions only on true change events (S_{t+1} != S_t)."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    cur_arr = np.asarray(current_state, dtype=int)
    if len(y_true_arr) != len(y_pred_arr) or len(y_true_arr) != len(cur_arr):
        raise ValueError("y_true, y_pred, and current_state must have equal length.")
    if len(y_true_arr) == 0:
        empty_cm = np.zeros((n_states, n_states), dtype=int)
        return {
            "same_state_fraction": float("nan"),
            "change_event_fraction": float("nan"),
            "n_change_events": 0,
            "accuracy_on_change_events": float("nan"),
            "confusion_matrix_change_events": empty_cm.tolist(),
        }

    same_mask = y_true_arr == cur_arr
    change_mask = ~same_mask
    n_change = int(np.sum(change_mask))
    cm = np.zeros((n_states, n_states), dtype=int)
    if n_change > 0:
        cm = confusion_matrix(
            y_true_arr[change_mask],
            y_pred_arr[change_mask],
            labels=np.arange(n_states),
        )
        change_acc = float(accuracy_score(y_true_arr[change_mask], y_pred_arr[change_mask]))
    else:
        change_acc = float("nan")

    return {
        "same_state_fraction": float(np.mean(same_mask)),
        "change_event_fraction": float(np.mean(change_mask)),
        "n_change_events": n_change,
        "accuracy_on_change_events": change_acc,
        "confusion_matrix_change_events": cm.tolist(),
    }


def evaluate_binary_change_predictions(
    y_true_change: np.ndarray,
    y_pred_change: np.ndarray,
    prob_change: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate binary regime-change prediction metrics."""
    y_true = np.asarray(y_true_change, dtype=int)
    y_pred = np.asarray(y_pred_change, dtype=int)
    if len(y_true) == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "precision_change": float("nan"),
            "recall_change": float("nan"),
            "roc_auc": float("nan"),
            "support_change": 0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pr, rc, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "precision_change": float(pr[1]),
        "recall_change": float(rc[1]),
        "f1_change": float(f1[1]),
        "support_change": int(supp[1]),
        "confusion_matrix": cm.tolist(),
    }
    if prob_change is not None:
        p = np.asarray(prob_change, dtype=float)
        if len(np.unique(y_true)) >= 2:
            out["roc_auc"] = float(roc_auc_score(y_true, p))
        else:
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
    return out


def evaluate_transition_type_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None = None,
    classes: np.ndarray | list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate transition-type classification on change events."""
    y_true_arr = np.asarray(y_true, dtype=str)
    y_pred_arr = np.asarray(y_pred, dtype=str)
    if len(y_true) == 0:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "classes": [],
            "confusion_matrix": [],
            "support": 0,
            "log_loss": float("nan"),
        }
    labels = (
        [str(v) for v in classes]
        if classes is not None
        else sorted(np.unique(np.concatenate([y_true_arr, y_pred_arr])).tolist())
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
        "classes": labels,
        "confusion_matrix": cm.tolist(),
        "support": int(len(y_true_arr)),
    }
    if probs is not None:
        p = np.asarray(probs, dtype=float)
        try:
            out["log_loss"] = float(log_loss(y_true_arr, p, labels=labels))
        except Exception:
            out["log_loss"] = float("nan")
    else:
        out["log_loss"] = float("nan")
    return out


def interpret_state_label(
    avg_vol: float,
    avg_future_vol: float,
    avg_slope: float,
    avg_spike_count: float,
    global_vol_quantiles: tuple[float, float],
    avg_jaggedness: float = 0.0,
    global_jagged_quantiles: tuple[float, float] = (0.0, 0.0),
) -> str:
    low_q, high_q = global_vol_quantiles
    low_jag, high_jag = global_jagged_quantiles

    if avg_vol <= low_q and avg_jaggedness <= low_jag and abs(avg_slope) <= 1e-4:
        return "low smooth vol"
    if avg_vol >= high_q and (avg_slope < 0 or avg_future_vol < avg_vol):
        return "stressed shock-decay vol"
    if avg_slope > 0 or avg_jaggedness >= low_jag:
        return "rising choppy vol"

    level = "low" if avg_vol <= low_q else ("high" if avg_vol >= high_q else "medium")
    trend = "rising" if avg_slope > 0 else ("falling" if avg_slope < 0 else "flat")
    texture = "choppy" if avg_jaggedness >= high_jag else "smooth"
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

    state_means = frame.groupby("state", observed=True)["vol_t"].mean()
    vol_q = (float(state_means.quantile(0.33)), float(state_means.quantile(0.67)))
    if "sum_jaggedness" in frame.columns and frame["sum_jaggedness"].notna().any():
        state_jag = frame.groupby("state", observed=True)["sum_jaggedness"].mean()
        jag_q = (
            float(state_jag.quantile(0.33)),
            float(state_jag.quantile(0.67)),
        )
    else:
        jag_q = (0.0, 0.0)
    out_rows: list[dict[str, float | int | str]] = []
    for s, grp in frame.groupby("state", sort=True):
        slope_col = "sum_trend_slope" if "sum_trend_slope" in grp.columns else None
        spike_col = "sum_spike_count" if "sum_spike_count" in grp.columns else None
        jag_col = "sum_jaggedness" if "sum_jaggedness" in grp.columns else None
        avg_slope = float(grp[slope_col].mean()) if slope_col else 0.0
        avg_spike = float(grp[spike_col].mean()) if spike_col else 0.0
        avg_jag = float(grp[jag_col].mean()) if jag_col else 0.0
        label = interpret_state_label(
            avg_vol=float(grp["vol_t"].mean()),
            avg_future_vol=float(grp["future_vol"].mean()),
            avg_slope=avg_slope,
            avg_spike_count=avg_spike,
            global_vol_quantiles=vol_q,
            avg_jaggedness=avg_jag,
            global_jagged_quantiles=jag_q,
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
            "avg_jaggedness": avg_jag,
            "interpretation": label,
        }
        for j in range(transition_matrix.shape[1]):
            row[f"p_to_{j}"] = float(transition_matrix[int(s), j])
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values("state").reset_index(drop=True)

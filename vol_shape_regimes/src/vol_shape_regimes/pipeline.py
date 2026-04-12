"""End-to-end experiment pipeline for volatility-shape regime modeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vol_shape_regimes.baselines import (
    empirical_transition_baseline,
    persistence_baseline,
    random_frequency_baseline,
    scalar_volatility_markov_predictions,
    scalar_volatility_states,
)
from vol_shape_regimes.clustering import assign_states, fit_clusterer
from vol_shape_regimes.config import ClusteringConfig, ExperimentConfig, load_experiment_config
from vol_shape_regimes.data import load_price_data
from vol_shape_regimes.embeddings import fit_embedding, transform_embedding
from vol_shape_regimes.evaluation import (
    change_event_diagnostics,
    dwell_time_by_state,
    evaluate_binary_change_predictions,
    evaluate_state_predictions,
    evaluate_transition_type_predictions,
    summarize_clusters,
    transition_out_probabilities,
)
from vol_shape_regimes.features import build_vol_series, build_window_features, compute_returns
from vol_shape_regimes.markov import (
    fit_binary_event_model,
    fit_conditional_transition_model,
    fit_transition_type_model,
    fit_markov_model,
    path_log_likelihood,
    predict_binary_event,
    predict_conditional_next_state,
    predict_transition_type,
    transition_entropy,
    transition_matrix_to_frame,
)
from vol_shape_regimes.plotting import (
    plot_confusion_matrix,
    plot_dwell_histogram,
    plot_embedding_scatter,
    plot_state_timeline,
    plot_transition_heatmap,
)
from vol_shape_regimes.utils import as_jsonable, ensure_dir, log_step, utc_now_str


def _split_train_test(n_rows: int, train_ratio: float) -> tuple[int, np.ndarray]:
    train_n = int(np.floor(n_rows * train_ratio))
    train_n = max(50, min(train_n, n_rows - 2))
    train_mask = np.arange(n_rows) < train_n
    if train_n < 30:
        raise ValueError("Train split is too small for robust fitting.")
    return train_n, train_mask


def _representative_windows(
    emb: pd.DataFrame,
    states: np.ndarray,
    centroids: np.ndarray,
    dates: pd.Series,
    top_n: int = 5,
) -> pd.DataFrame:
    x = emb.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for state in sorted(np.unique(states)):
        idx = np.where(states == state)[0]
        if len(idx) == 0:
            continue
        c = centroids[int(state)]
        d = np.linalg.norm(x[idx] - c[None, :], axis=1)
        best = idx[np.argsort(d)[:top_n]]
        for rank, i in enumerate(best, start=1):
            rows.append(
                {
                    "state": int(state),
                    "rank": int(rank),
                    "date": str(pd.to_datetime(dates.iloc[i]).date()),
                    "distance_to_centroid": float(np.linalg.norm(x[i] - c)),
                }
            )
    return pd.DataFrame(rows)


def _build_transition_frame(
    states: np.ndarray,
    emb: pd.DataFrame,
    feature_frame: pd.DataFrame,
    train_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    emb_cols = list(emb.columns)
    for t in range(1, len(states) - 1):
        row: dict[str, Any] = {
            "t": t,
            "date": pd.to_datetime(feature_frame.iloc[t]["date"]),
            "prev_state": int(states[t - 1]),
            "current_state": int(states[t]),
            "next_state": int(states[t + 1]),
            "ret_t": float(feature_frame.iloc[t]["ret"]),
            "vol_t": float(feature_frame.iloc[t]["vol_t"]),
            "sum_trend_slope": float(feature_frame.iloc[t].get("sum_trend_slope", 0.0)),
        }
        for col in emb_cols:
            row[col] = float(emb.iloc[t][col])
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Transition frame is empty.")
    # t <= train_n-2 keeps next_state fully inside train.
    train_df = frame[frame["t"] <= (train_n - 2)].copy()
    test_df = frame[frame["t"] >= (train_n - 1)].copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _time_in_state(states: np.ndarray) -> np.ndarray:
    seq = np.asarray(states, dtype=int)
    out = np.zeros(len(seq), dtype=int)
    if len(seq) == 0:
        return out
    run = 1
    out[0] = run
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            run += 1
        else:
            run = 1
        out[i] = run
    return out


def _build_instability_frame(
    *,
    states: np.ndarray,
    emb: pd.DataFrame,
    centroids: np.ndarray,
    feature_frame: pd.DataFrame,
    train_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build leakage-safe rows for change-event and hazard modeling."""
    emb_cols = list(emb.columns)
    x = emb.to_numpy(dtype=float)
    cur = np.asarray(states, dtype=int)
    if len(cur) != len(feature_frame):
        raise ValueError("states and feature_frame length mismatch")

    time_in_state = _time_in_state(cur)
    centroid_dist = np.linalg.norm(x - centroids[cur], axis=1)
    emb_velocity = np.zeros(len(cur), dtype=float)
    if len(cur) > 1:
        emb_velocity[1:] = np.linalg.norm(x[1:] - x[:-1], axis=1)

    vol_t = pd.to_numeric(feature_frame["vol_t"], errors="coerce").to_numpy(dtype=float)
    vol_accel = np.zeros(len(vol_t), dtype=float)
    if len(vol_t) > 2:
        vol_accel[2:] = (vol_t[2:] - vol_t[1:-1]) - (vol_t[1:-1] - vol_t[:-2])
    vol_of_vol = pd.Series(vol_t).rolling(10, min_periods=3).std().fillna(0.0).to_numpy(dtype=float)
    ret = pd.to_numeric(feature_frame["ret"], errors="coerce").to_numpy(dtype=float)
    ret_disp = pd.Series(ret).rolling(10, min_periods=3).std().fillna(0.0).to_numpy(dtype=float)
    tail_hit = (np.abs(ret) > pd.Series(np.abs(ret)).rolling(50, min_periods=10).quantile(0.9).fillna(0.0)).astype(float)
    tail_intensity = pd.Series(tail_hit).rolling(10, min_periods=1).mean().fillna(0.0).to_numpy(dtype=float)
    entropy_proxy = pd.Series(np.abs(ret)).rolling(20, min_periods=5).std().fillna(0.0).to_numpy(dtype=float)
    entropy_delta = np.r_[0.0, np.diff(entropy_proxy)]

    rows: list[dict[str, Any]] = []
    for t in range(1, len(cur) - 1):
        nxt = int(cur[t + 1])
        cst = int(cur[t])
        row: dict[str, Any] = {
            "t": int(t),
            "date": pd.to_datetime(feature_frame.iloc[t]["date"]),
            "current_state": cst,
            "prev_state": int(cur[t - 1]),
            "next_state": nxt,
            "is_change": int(nxt != cst),
            "transition_type": f"{cst}->{nxt}",
            "distance_to_centroid": float(centroid_dist[t]),
            "embedding_velocity": float(emb_velocity[t]),
            "vol_t": float(vol_t[t]),
            "vol_accel": float(vol_accel[t]),
            "vol_of_vol": float(vol_of_vol[t]),
            "tail_intensity": float(tail_intensity[t]),
            "entropy_delta": float(entropy_delta[t]),
            "ret_dispersion": float(ret_disp[t]),
            "time_in_state": int(time_in_state[t]),
        }
        for col in emb_cols:
            row[col] = float(emb.iloc[t][col])
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Instability frame is empty.")
    train_df = frame[frame["t"] <= (train_n - 2)].copy().reset_index(drop=True)
    test_df = frame[frame["t"] >= (train_n - 1)].copy().reset_index(drop=True)
    return train_df, test_df


def _conditional_design(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cat_cols = ["current_state", "prev_state"]
    num_cols = [c for c in train_df.columns if c.startswith("emb_")] + ["ret_t", "vol_t", "sum_trend_slope"]

    train_cat = pd.get_dummies(train_df[cat_cols].astype(str), prefix=["cur", "prev"])
    test_cat = pd.get_dummies(test_df[cat_cols].astype(str), prefix=["cur", "prev"])
    train_cat, test_cat = train_cat.align(test_cat, join="outer", axis=1, fill_value=0)

    train_x = pd.concat([train_cat, train_df[num_cols].reset_index(drop=True)], axis=1)
    test_x = pd.concat([test_cat, test_df[num_cols].reset_index(drop=True)], axis=1)
    feature_cols = list(train_x.columns)
    train_use = pd.concat([train_x, train_df[["next_state"]].reset_index(drop=True)], axis=1)
    test_use = pd.concat([test_x, test_df[["next_state"]].reset_index(drop=True)], axis=1)
    return train_use, test_use, feature_cols


def _event_design(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cat_cols = ["current_state", "prev_state"]
    base_num = [
        "distance_to_centroid",
        "embedding_velocity",
        "vol_t",
        "vol_accel",
        "vol_of_vol",
        "tail_intensity",
        "entropy_delta",
        "ret_dispersion",
        "time_in_state",
    ]
    emb_cols = [c for c in train_df.columns if c.startswith("emb_")]
    num_cols = emb_cols + [c for c in base_num if c in train_df.columns]

    train_cat = pd.get_dummies(train_df[cat_cols].astype(str), prefix=["cur", "prev"])
    test_cat = pd.get_dummies(test_df[cat_cols].astype(str), prefix=["cur", "prev"])
    train_cat, test_cat = train_cat.align(test_cat, join="outer", axis=1, fill_value=0)
    train_x = pd.concat([train_cat.reset_index(drop=True), train_df[num_cols].reset_index(drop=True)], axis=1)
    test_x = pd.concat([test_cat.reset_index(drop=True), test_df[num_cols].reset_index(drop=True)], axis=1)
    feature_cols = list(train_x.columns)
    train_use = pd.concat(
        [
            train_x,
            train_df[["is_change", "transition_type", "next_state", "current_state", "time_in_state"]].reset_index(drop=True),
        ],
        axis=1,
    )
    test_use = pd.concat(
        [
            test_x,
            test_df[["is_change", "transition_type", "next_state", "current_state", "time_in_state"]].reset_index(drop=True),
        ],
        axis=1,
    )
    return train_use, test_use, feature_cols


def _hazard_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one-step hazard-style dataset from instability rows."""
    out = frame.copy()
    out["event_next"] = out["is_change"].astype(int)
    # duration at t is known causally and can inform hazard.
    out["duration"] = out["time_in_state"].astype(float)
    return out


def _candidate_transition_diagnostics(
    x_train_emb: pd.DataFrame,
    x_all_emb: pd.DataFrame,
    train_n: int,
    cfg: ExperimentConfig,
    clustering_diag: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate first-order next-state diagnostics across clustering candidates."""
    cols = [
        "method",
        "k",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "temporal_stability",
        "min_occupancy",
        "same_state_fraction",
        "change_event_fraction",
        "n_change_events",
        "markov_order1_accuracy",
        "persistence_accuracy",
        "change_event_accuracy",
    ]
    rows: list[dict[str, Any]] = []
    for _, cand in clustering_diag.iterrows():
        method = str(cand["method"])
        k = int(cand["k"])
        single_cfg = ClusteringConfig(
            methods=(method,),
            k_min=k,
            k_max=k,
            random_state=cfg.clustering.random_state,
            min_state_occupancy=cfg.clustering.min_state_occupancy,
            select_metric=cfg.clustering.select_metric,
        )
        c_model, _, _ = fit_clusterer(x_train_emb, single_cfg)
        states = assign_states(c_model, x_all_emb)
        y_true = states[train_n:]
        current = states[train_n - 1 : -1]
        markov = fit_markov_model(
            states_train=states[:train_n],
            order=1,
            n_states=k,
            smoothing=cfg.transitions.laplace_smoothing,
        )
        probs = np.vstack([markov.predict_proba(int(s)) for s in current])
        preds = np.argmax(probs, axis=1).astype(int)
        eval_main = evaluate_state_predictions(
            y_true=y_true,
            y_pred=preds,
            probs=probs,
            top_k=cfg.evaluation.top_k_accuracy,
        )
        eval_change = change_event_diagnostics(
            y_true=y_true,
            y_pred=preds,
            current_state=current,
            n_states=k,
        )
        persistence_pred = persistence_baseline(current)
        persistence_eval = evaluate_state_predictions(
            y_true=y_true,
            y_pred=persistence_pred,
            probs=None,
            top_k=cfg.evaluation.top_k_accuracy,
        )
        rows.append(
            {
                "method": method,
                "k": k,
                "silhouette": float(cand.get("silhouette", float("nan"))),
                "davies_bouldin": float(cand.get("davies_bouldin", float("nan"))),
                "calinski_harabasz": float(cand.get("calinski_harabasz", float("nan"))),
                "temporal_stability": float(cand.get("temporal_stability", float("nan"))),
                "min_occupancy": float(cand.get("min_occupancy", float("nan"))),
                "same_state_fraction": float(eval_change["same_state_fraction"]),
                "change_event_fraction": float(eval_change["change_event_fraction"]),
                "n_change_events": int(eval_change["n_change_events"]),
                "markov_order1_accuracy": float(eval_main["accuracy"]),
                "persistence_accuracy": float(persistence_eval["accuracy"]),
                "change_event_accuracy": float(eval_change["accuracy_on_change_events"]),
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    return out.sort_values(
        by=["markov_order1_accuracy", "silhouette"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _write_markdown_summary(
    output_dir: Path,
    config: ExperimentConfig,
    metrics: dict[str, Any],
    cluster_summary: pd.DataFrame,
    best_clustering_row: dict[str, Any],
    candidate_transition_diag: pd.DataFrame,
) -> None:
    baseline_tbl = pd.DataFrame(metrics["baseline_comparison"])
    best_row = baseline_tbl.sort_values("accuracy", ascending=False).iloc[0]
    txt = [
        f"# Volatility Shape Regime Report: {config.experiment_name}",
        "",
        f"- generated_at: {utc_now_str()}",
        f"- asset: {config.data.asset}",
        f"- rows: {metrics['data_rows']}",
        "",
        "## Best feature/embedding/clustering combo",
        f"- embedding method: {metrics['embedding']['method']}",
        f"- clustering method: {best_clustering_row.get('method')}",
        f"- K: {best_clustering_row.get('k')}",
        f"- silhouette: {best_clustering_row.get('silhouette')}",
        "",
        "## Predictability summary",
        f"- best model: {best_row['model']}",
        f"- best accuracy: {best_row['accuracy']:.4f}",
        f"- persistence accuracy: {metrics['baselines']['persistence']['accuracy']:.4f}",
        f"- fraction S_t+1 = S_t (test): {metrics['markov']['same_state_fraction']:.4f}",
        f"- accuracy on change events only: {metrics['markov']['change_event']['accuracy_on_change_events']:.4f}",
        "",
        "## Cluster stability / interpretability",
        f"- average dwell length: {metrics['dwell']['overall_mean_dwell']:.2f}",
        f"- transition entropy: {metrics['markov']['transition_entropy']:.4f}",
        "",
        "## Key states",
    ]
    for _, row in cluster_summary.iterrows():
        txt.append(
            (
                f"- state {int(row['state'])}: {row['interpretation']}, "
                f"prevalence={row['prevalence']:.3f}, avg_vol={row['avg_vol_t']:.5f}, "
                f"future_vol={row['avg_future_vol']:.5f}"
            )
        )
    txt.append("")
    txt.append("## Interpretation archetype check")
    labels = [str(v).lower() for v in cluster_summary["interpretation"].tolist()]
    txt.append(f"- low smooth vol found: {any('low smooth vol' in v for v in labels)}")
    txt.append(f"- rising/choppy vol found: {any('rising choppy vol' in v for v in labels)}")
    txt.append(f"- stressed shock-decay vol found: {any('stressed shock-decay vol' in v for v in labels)}")
    txt.append("")
    txt.append("## K=4/K=5 and GMM comparison")
    subset = candidate_transition_diag[
        candidate_transition_diag["k"].isin([4, 5]) | (candidate_transition_diag["method"].str.lower() == "gmm")
    ].copy()
    if subset.empty:
        txt.append("- no matching candidates in this run.")
    else:
        for _, row in subset.sort_values(["markov_order1_accuracy", "silhouette"], ascending=[False, False]).iterrows():
            txt.append(
                "- "
                f"{row['method']} K={int(row['k'])}: "
                f"markov_acc={float(row['markov_order1_accuracy']):.4f}, "
                f"change_acc={float(row['change_event_accuracy']):.4f}, "
                f"same_frac={float(row['same_state_fraction']):.4f}, "
                f"silhouette={float(row['silhouette']):.4f}"
            )
    txt.append("")
    txt.append("## Conclusion")
    if best_row["accuracy"] > metrics["baselines"]["persistence"]["accuracy"]:
        txt.append("Vol-shape states improved next-state prediction versus persistence baseline.")
    else:
        txt.append("Vol-shape states did not beat persistence in this run; transitions remain partly unpredictable.")

    (output_dir / "research_summary.md").write_text("\n".join(txt), encoding="utf-8")


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run full volatility-shape state pipeline and save artifacts."""
    output_dir = ensure_dir(Path(config.output.base_dir) / config.experiment_name)
    plots_dir = ensure_dir(output_dir / "plots")
    log_step(f"loading data ({config.data.source})")
    pdata = load_price_data(config.data)
    price_df = pdata.frame.copy()
    price_df["ret"] = compute_returns(price_df["close"], return_type=config.volatility.return_type)
    price_df["abs_ret"] = price_df["ret"].abs()

    log_step(f"building volatility series ({config.volatility.method})")
    vol_series = build_vol_series(price_df, config.volatility.method, config.volatility)
    log_step("building rolling window features")
    feat = build_window_features(vol_series, config.features, dates=price_df["date"])

    # Join aligned raw columns to feature rows.
    align = price_df[["date", "close", "ret", "abs_ret"]].copy()
    feat = feat.merge(align, on="date", how="left")
    feat = feat.sort_values("date").reset_index(drop=True)
    feat = feat.dropna(subset=["close", "ret", "abs_ret", "vol_t"]).reset_index(drop=True)
    if len(feat) < 150:
        raise ValueError("Feature frame too small after alignment.")

    feature_cols = [c for c in feat.columns if c not in {"date", "close", "ret", "abs_ret"}]
    x_all = feat[feature_cols].copy()
    train_n, train_mask = _split_train_test(len(feat), config.evaluation.train_ratio)

    log_step("fitting embedding")
    emb_model, x_train_emb, emb_diag = fit_embedding(x_all.loc[train_mask], config.embedding)
    x_test_emb = transform_embedding(emb_model, x_all.loc[~train_mask])
    x_all_emb = pd.concat([x_train_emb, x_test_emb], axis=0).sort_index()
    x_all_emb = x_all_emb.reset_index(drop=True)

    log_step("fitting clustering candidates")
    cluster_model, train_states, clustering_diag = fit_clusterer(x_train_emb, config.clustering)
    full_states = assign_states(cluster_model, x_all_emb)
    feat["state"] = full_states

    log_step("fitting first-order markov model")
    markov_1 = fit_markov_model(
        states_train=full_states[:train_n],
        order=1,
        n_states=cluster_model.k,
        smoothing=config.transitions.laplace_smoothing,
    )
    matrix_1 = markov_1.transition_1

    # Next-state evaluation dataset for markov family.
    y_true = full_states[train_n:]
    current_states = full_states[train_n - 1 : -1]
    prev_states = full_states[train_n - 2 : -2] if train_n >= 2 else full_states[train_n - 1 : -1]
    probs_markov = np.vstack([markov_1.predict_proba(int(s)) for s in current_states])
    preds_markov = np.argmax(probs_markov, axis=1).astype(int)

    metrics_markov = evaluate_state_predictions(
        y_true=y_true,
        y_pred=preds_markov,
        probs=probs_markov,
        top_k=config.evaluation.top_k_accuracy,
    )
    markov_change_diag = change_event_diagnostics(
        y_true=y_true,
        y_pred=preds_markov,
        current_state=current_states,
        n_states=cluster_model.k,
    )

    second_order_metrics: dict[str, Any] | None = None
    second_order_change_diag: dict[str, Any] | None = None
    matrix_2_df = pd.DataFrame()
    if config.transitions.fit_second_order:
        log_step("fitting second-order markov model")
        markov_2 = fit_markov_model(
            states_train=full_states[:train_n],
            order=2,
            n_states=cluster_model.k,
            smoothing=config.transitions.laplace_smoothing,
        )
        probs2 = np.vstack(
            [
                markov_2.predict_proba(int(cur), int(prev))
                for prev, cur in zip(prev_states, current_states, strict=True)
            ]
        )
        preds2 = np.argmax(probs2, axis=1).astype(int)
        second_order_metrics = evaluate_state_predictions(
            y_true=y_true,
            y_pred=preds2,
            probs=probs2,
            top_k=config.evaluation.top_k_accuracy,
        )
        second_order_change_diag = change_event_diagnostics(
            y_true=y_true,
            y_pred=preds2,
            current_state=current_states,
            n_states=cluster_model.k,
        )
        flat_rows: list[dict[str, Any]] = []
        assert markov_2.transition_2 is not None
        for s_prev in range(markov_2.n_states):
            for s_cur in range(markov_2.n_states):
                row: dict[str, Any] = {"prev_state": s_prev, "current_state": s_cur}
                for s_next in range(markov_2.n_states):
                    row[f"p_to_{s_next}"] = float(markov_2.transition_2[s_prev, s_cur, s_next])
                flat_rows.append(row)
        matrix_2_df = pd.DataFrame(flat_rows)

    log_step("fitting conditional transition model")
    conditional_metrics: dict[str, Any] | None = None
    conditional_change_diag: dict[str, Any] | None = None
    if config.transitions.fit_conditional_model:
        train_trans, test_trans = _build_transition_frame(full_states, x_all_emb, feat, train_n)
        cond_train, cond_test, cond_cols = _conditional_design(train_trans, test_trans)
        cond_model = fit_conditional_transition_model(
            frame=cond_train,
            feature_cols=cond_cols,
            target_col="next_state",
            n_states=cluster_model.k,
            max_iter=config.transitions.conditional_max_iter,
        )
        cond_probs, cond_preds = predict_conditional_next_state(cond_model, cond_test)
        conditional_metrics = evaluate_state_predictions(
            y_true=cond_test["next_state"].to_numpy(dtype=int),
            y_pred=cond_preds,
            probs=cond_probs,
            top_k=config.evaluation.top_k_accuracy,
        )
        conditional_change_diag = change_event_diagnostics(
            y_true=cond_test["next_state"].to_numpy(dtype=int),
            y_pred=cond_preds,
            current_state=test_trans["current_state"].to_numpy(dtype=int),
            n_states=cluster_model.k,
        )

    log_step("fitting change-event / transition-type / hazard models")
    instability_train, instability_test = _build_instability_frame(
        states=full_states,
        emb=x_all_emb,
        centroids=cluster_model.centroids,
        feature_frame=feat,
        train_n=train_n,
    )
    event_train, event_test, event_cols = _event_design(instability_train, instability_test)

    event_metrics: dict[str, Any] | None = None
    transition_type_metrics: dict[str, Any] | None = None
    hazard_metrics: dict[str, Any] | None = None
    event_model = None
    event_prob_test: np.ndarray | None = None
    event_pred_test: np.ndarray | None = None
    hazard_prob_test: np.ndarray | None = None
    hazard_pred_test: np.ndarray | None = None
    if event_train["is_change"].nunique() >= 2:
        event_model = fit_binary_event_model(
            frame=event_train,
            feature_cols=event_cols,
            target_col="is_change",
            class_weight="balanced",
            max_iter=max(500, config.transitions.conditional_max_iter),
            threshold=0.5,
        )
        event_prob_test, event_pred_test = predict_binary_event(event_model, event_test)
        event_metrics = evaluate_binary_change_predictions(
            y_true_change=event_test["is_change"].to_numpy(dtype=int),
            y_pred_change=event_pred_test,
            prob_change=event_prob_test,
        )

        # Transition type only on predicted change events where y_true class exists in train.
        type_train = event_train[event_train["is_change"] == 1].copy()
        type_test = event_test[event_test["is_change"] == 1].copy()
        if not type_train.empty and not type_test.empty and type_train["transition_type"].nunique() >= 2:
            tt_model = fit_transition_type_model(
                frame=type_train,
                feature_cols=event_cols,
                target_col="transition_type",
                class_weight=None,
                max_iter=max(500, config.transitions.conditional_max_iter),
            )
            tt_probs, tt_pred = predict_transition_type(tt_model, type_test)
            transition_type_metrics = evaluate_transition_type_predictions(
                y_true=type_test["transition_type"].astype(str).to_numpy(),
                y_pred=tt_pred,
                probs=tt_probs,
                classes=tt_model.classes_,
            )

        # Hazard-style model: same binary target with explicit time_in_state channel.
        hazard_model = fit_binary_event_model(
            frame=event_train,
            feature_cols=event_cols,
            target_col="is_change",
            class_weight="balanced",
            max_iter=max(500, config.transitions.conditional_max_iter),
            threshold=0.5,
        )
        hazard_prob, hazard_pred = predict_binary_event(hazard_model, event_test)
        hazard_prob_test = hazard_prob
        hazard_pred_test = hazard_pred
        hazard_metrics = evaluate_binary_change_predictions(
            y_true_change=event_test["is_change"].to_numpy(dtype=int),
            y_pred_change=hazard_pred,
            prob_change=hazard_prob,
        )
    else:
        event_metrics = {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "precision_change": float("nan"),
            "recall_change": float("nan"),
            "f1_change": float("nan"),
            "support_change": 0,
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    log_step("running baselines")
    persistence_preds = persistence_baseline(current_states)
    persistence_metrics = evaluate_state_predictions(
        y_true=y_true,
        y_pred=persistence_preds,
        probs=None,
        top_k=config.evaluation.top_k_accuracy,
    )
    persistence_change_diag = change_event_diagnostics(
        y_true=y_true,
        y_pred=persistence_preds,
        current_state=current_states,
        n_states=cluster_model.k,
    )
    empirical_preds = empirical_transition_baseline(markov_1, current_states)
    empirical_metrics = evaluate_state_predictions(y_true=y_true, y_pred=empirical_preds)
    random_preds = random_frequency_baseline(
        n_states=cluster_model.k,
        n_obs=len(y_true),
        state_freq=markov_1.state_freq if markov_1.state_freq is not None else np.ones(cluster_model.k) / cluster_model.k,
        random_state=config.clustering.random_state,
    )
    random_metrics = evaluate_state_predictions(y_true=y_true, y_pred=random_preds)

    scalar_states = scalar_volatility_states(
        feat["vol_t"],
        train_mask=pd.Series(train_mask),
        n_states=cluster_model.k,
    )
    y_scalar, pred_scalar, scalar_model = scalar_volatility_markov_predictions(
        scalar_states=scalar_states,
        train_end_idx=train_n - 1,
        n_states=cluster_model.k,
        smoothing=config.transitions.laplace_smoothing,
    )
    scalar_metrics = evaluate_state_predictions(y_true=y_scalar, y_pred=pred_scalar)

    candidate_transition_diag = _candidate_transition_diagnostics(
        x_train_emb=x_train_emb,
        x_all_emb=x_all_emb,
        train_n=train_n,
        cfg=config,
        clustering_diag=clustering_diag,
    )

    log_step("building interpretability tables")
    cluster_summary = summarize_clusters(
        feature_frame=feat,
        states=full_states,
        dates=feat["date"],
        vol_series_aligned=feat["vol_t"],
        abs_returns_aligned=feat["abs_ret"],
        transition_matrix=matrix_1,
    )
    representatives = _representative_windows(
        emb=x_all_emb,
        states=full_states,
        centroids=cluster_model.centroids,
        dates=feat["date"],
        top_n=5,
    )
    dwell_df = dwell_time_by_state(full_states)
    dwell_summary = (
        dwell_df.groupby("state", observed=True)["dwell_len"]
        .agg(["mean", "median", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_dwell", "median": "median_dwell", "max": "max_dwell", "count": "n_runs"})
    )
    trans_out = transition_out_probabilities(matrix_1)

    log_step("saving CSV/JSON artifacts")
    cluster_summary.to_csv(output_dir / "cluster_summaries.csv", index=False)
    representatives.to_csv(output_dir / "cluster_representative_windows.csv", index=False)
    transition_matrix_to_frame(matrix_1).to_csv(output_dir / "transition_matrix_order1.csv")
    trans_out.to_csv(output_dir / "transition_out_probabilities.csv", index=False)
    dwell_df.to_csv(output_dir / "dwell_times.csv", index=False)
    dwell_summary.to_csv(output_dir / "dwell_summary.csv", index=False)
    clustering_diag.to_csv(output_dir / "clustering_model_selection.csv", index=False)
    candidate_transition_diag.to_csv(output_dir / "candidate_transition_diagnostics.csv", index=False)
    candidate_transition_diag[
        candidate_transition_diag["k"].isin([4, 5]) | (candidate_transition_diag["method"].str.lower() == "gmm")
    ].to_csv(output_dir / "candidate_k4_k5_gmm_comparison.csv", index=False)
    instability_test.to_csv(output_dir / "instability_features_test.csv", index=False)
    event_test_out = event_test.copy()
    if event_prob_test is not None and event_pred_test is not None:
        event_test_out["pred_change_prob"] = event_prob_test
        event_test_out["pred_change"] = event_pred_test
    event_test_out.to_csv(output_dir / "change_event_predictions.csv", index=False)
    if hazard_prob_test is not None and hazard_pred_test is not None:
        hazard_test_out = event_test.copy()
        hazard_test_out["pred_hazard_prob"] = hazard_prob_test
        hazard_test_out["pred_hazard_change"] = hazard_pred_test
        hazard_test_out.to_csv(output_dir / "hazard_change_predictions.csv", index=False)
    if transition_type_metrics is not None:
        type_rows = event_test[event_test["is_change"] == 1].copy()
        if "transition_type" in type_rows.columns:
            type_rows.to_csv(output_dir / "transition_type_events_test.csv", index=False)
    matrix_2_df.to_csv(output_dir / "transition_matrix_order2.csv", index=False)
    feat[["date", "close", "vol_t", "ret", "abs_ret", "state"]].to_csv(output_dir / "state_timeline.csv", index=False)
    pd.DataFrame(markov_change_diag["confusion_matrix_change_events"]).to_csv(
        output_dir / "confusion_change_events_markov_order1.csv",
        index=False,
    )

    log_step("generating plots")
    plot_embedding_scatter(x_all_emb, full_states, feat["date"], plots_dir / "embedding_states.png")
    plot_transition_heatmap(matrix_1, plots_dir / "transition_heatmap.png")
    plot_state_timeline(feat["date"], feat["close"], full_states, plots_dir / "state_timeline.png")
    plot_dwell_histogram(dwell_df, plots_dir / "dwell_histogram.png")
    plot_confusion_matrix(
        np.asarray(metrics_markov["confusion_matrix"]),
        plots_dir / "confusion_markov_order1.png",
        title="Next-state confusion matrix (all events)",
    )
    plot_confusion_matrix(
        np.asarray(markov_change_diag["confusion_matrix_change_events"]),
        plots_dir / "confusion_markov_order1_change_events.png",
        title="Next-state confusion matrix (change events only)",
    )

    baseline_comparison = [
        {"model": "markov_order1", "accuracy": metrics_markov["accuracy"], "macro_f1": metrics_markov["macro_f1"]},
        {"model": "persistence", "accuracy": persistence_metrics["accuracy"], "macro_f1": persistence_metrics["macro_f1"]},
        {"model": "empirical_transition_argmax", "accuracy": empirical_metrics["accuracy"], "macro_f1": empirical_metrics["macro_f1"]},
        {"model": "scalar_vol_markov", "accuracy": scalar_metrics["accuracy"], "macro_f1": scalar_metrics["macro_f1"]},
        {"model": "random_frequency", "accuracy": random_metrics["accuracy"], "macro_f1": random_metrics["macro_f1"]},
    ]
    if second_order_metrics is not None:
        baseline_comparison.append(
            {"model": "markov_order2", "accuracy": second_order_metrics["accuracy"], "macro_f1": second_order_metrics["macro_f1"]}
        )
    if conditional_metrics is not None:
        baseline_comparison.append(
            {
                "model": "conditional_multinomial_logit",
                "accuracy": conditional_metrics["accuracy"],
                "macro_f1": conditional_metrics["macro_f1"],
            }
        )
    if event_metrics is not None and not np.isnan(float(event_metrics.get("balanced_accuracy", np.nan))):
        baseline_comparison.append(
            {
                "model": "change_event_logit",
                "accuracy": float(event_metrics["balanced_accuracy"]),
                "macro_f1": float(event_metrics.get("f1_change", np.nan)),
            }
        )
    if hazard_metrics is not None and not np.isnan(float(hazard_metrics.get("balanced_accuracy", np.nan))):
        baseline_comparison.append(
            {
                "model": "hazard_change_logit",
                "accuracy": float(hazard_metrics["balanced_accuracy"]),
                "macro_f1": float(hazard_metrics.get("f1_change", np.nan)),
            }
        )

    best_cluster_row = clustering_diag.iloc[0].to_dict() if not clustering_diag.empty else {}
    evaluation = {
        "generated_at": utc_now_str(),
        "experiment_name": config.experiment_name,
        "data_rows": int(len(price_df)),
        "feature_rows": int(len(feat)),
        "train_rows": int(train_n),
        "test_rows": int(len(feat) - train_n),
        "n_states": int(cluster_model.k),
        "best_clustering_candidate": best_cluster_row,
        "embedding": emb_diag,
        "markov": {
            **metrics_markov,
            "same_state_fraction": float(markov_change_diag["same_state_fraction"]),
            "change_event": markov_change_diag,
            "transition_entropy": transition_entropy(matrix_1),
            "path_log_likelihood_test": path_log_likelihood(markov_1, full_states[train_n - 1 :]),
        },
        "markov_order2": (
            {**second_order_metrics, "change_event": second_order_change_diag}
            if second_order_metrics is not None
            else None
        ),
        "conditional_model": (
            {**conditional_metrics, "change_event": conditional_change_diag}
            if conditional_metrics is not None
            else None
        ),
        "change_event_model": event_metrics,
        "transition_type_model": transition_type_metrics,
        "hazard_model": hazard_metrics,
        "baselines": {
            "persistence": {**persistence_metrics, "change_event": persistence_change_diag},
            "empirical_transition": empirical_metrics,
            "scalar_volatility": scalar_metrics,
            "random_frequency": random_metrics,
        },
        "baseline_comparison": baseline_comparison,
        "candidate_transition_diagnostics": candidate_transition_diag.to_dict(orient="records"),
        "dwell": {
            "overall_mean_dwell": float(dwell_df["dwell_len"].mean()),
            "overall_median_dwell": float(dwell_df["dwell_len"].median()),
        },
    }
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(as_jsonable(evaluation), indent=2), encoding="utf-8")

    _write_markdown_summary(
        output_dir=output_dir,
        config=config,
        metrics=evaluation,
        cluster_summary=cluster_summary,
        best_clustering_row=best_cluster_row,
        candidate_transition_diag=candidate_transition_diag,
    )

    log_step(f"done. artifacts saved to: {output_dir}")
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vol-shape regime experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    evaluation = run_experiment(cfg)
    top = sorted(evaluation["baseline_comparison"], key=lambda r: r["accuracy"], reverse=True)[0]
    print(
        json.dumps(
            {
                "experiment_name": cfg.experiment_name,
                "best_model": top["model"],
                "best_accuracy": top["accuracy"],
                "output_dir": str(Path(cfg.output.base_dir) / cfg.experiment_name),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

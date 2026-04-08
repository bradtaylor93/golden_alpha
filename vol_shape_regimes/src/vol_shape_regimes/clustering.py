"""Clustering over embeddings with model selection diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from vol_shape_regimes.config import ClusteringConfig


@dataclass
class ClusterModel:
    method: str
    k: int
    model: Any
    centroids: np.ndarray


def _temporal_stability(labels: np.ndarray) -> float:
    if len(labels) <= 1:
        return 1.0
    switches = np.sum(labels[1:] != labels[:-1])
    return float(1.0 - switches / (len(labels) - 1))


def _min_occupancy(labels: np.ndarray, k: int) -> float:
    counts = np.bincount(labels, minlength=k).astype(float)
    return float(np.min(counts) / max(float(np.sum(counts)), 1.0))


def _fit_candidate(x: np.ndarray, method: str, k: int, random_state: int) -> tuple[Any, np.ndarray]:
    method = method.lower()
    if method == "kmeans":
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(x)
        return model, labels.astype(int)
    if method == "gmm":
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state, reg_covar=1e-6)
        labels = model.fit_predict(x)
        return model, labels.astype(int)
    if method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(x)
        return model, labels.astype(int)
    raise ValueError(f"Unsupported clustering method: {method}")


def _compute_centroids(x: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    c = np.zeros((k, x.shape[1]), dtype=float)
    for state in range(k):
        mask = labels == state
        if not np.any(mask):
            continue
        c[state] = np.mean(x[mask], axis=0)
    return c


def _safe_metric(func, x: np.ndarray, labels: np.ndarray) -> float:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")
    try:
        return float(func(x, labels))
    except Exception:
        return float("nan")


def fit_clusterer(
    x_train_emb: pd.DataFrame,
    cfg: ClusteringConfig,
) -> tuple[ClusterModel, np.ndarray, pd.DataFrame]:
    """Fit candidate clusterers and select best configuration."""
    x = x_train_emb.to_numpy(dtype=float)
    candidates: list[dict[str, float | int | str]] = []
    best_score = -np.inf
    best_model: ClusterModel | None = None
    best_labels: np.ndarray | None = None
    metric_name = cfg.select_metric.lower()

    for method in cfg.methods:
        for k in range(cfg.k_min, cfg.k_max + 1):
            model, labels = _fit_candidate(x, method=method, k=k, random_state=cfg.random_state)
            silhouette = _safe_metric(silhouette_score, x, labels)
            dbi = _safe_metric(davies_bouldin_score, x, labels)
            chs = _safe_metric(calinski_harabasz_score, x, labels)
            occupancy = _min_occupancy(labels, k)
            stability = _temporal_stability(labels)

            row = {
                "method": method,
                "k": k,
                "silhouette": silhouette,
                "davies_bouldin": dbi,
                "calinski_harabasz": chs,
                "temporal_stability": stability,
                "min_occupancy": occupancy,
            }
            candidates.append(row)

            score = silhouette if metric_name == "silhouette" else chs
            if np.isnan(score):
                continue
            if occupancy < cfg.min_state_occupancy:
                score -= 10.0
            if score > best_score:
                best_score = score
                centroids = _compute_centroids(x, labels, k)
                best_model = ClusterModel(method=method, k=k, model=model, centroids=centroids)
                best_labels = labels

    if best_model is None or best_labels is None:
        raise RuntimeError("No valid clustering candidate found.")

    diag = pd.DataFrame(candidates).sort_values(
        by=["silhouette", "calinski_harabasz"],
        ascending=[False, False],
    )
    return best_model, best_labels, diag.reset_index(drop=True)


def assign_states(clusterer: ClusterModel, x_emb: pd.DataFrame) -> np.ndarray:
    """Assign states for embedding rows using fitted clusterer."""
    x = x_emb.to_numpy(dtype=float)
    method = clusterer.method.lower()
    model = clusterer.model
    if method == "kmeans":
        return model.predict(x).astype(int)
    if method == "gmm":
        return model.predict(x).astype(int)
    if method == "agglomerative":
        # Agglomerative has no out-of-sample predict; assign nearest train centroid.
        dists = ((x[:, None, :] - clusterer.centroids[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(dists, axis=1).astype(int)
    raise ValueError(f"Unsupported clustering method: {clusterer.method}")

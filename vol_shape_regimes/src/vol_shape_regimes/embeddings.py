"""Embedding layer for volatility-shape features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from vol_shape_regimes.config import EmbeddingConfig


@dataclass
class EmbeddingModel:
    method: str
    scaler: StandardScaler | None
    pca: PCA | None
    feature_names: list[str]


def _to_matrix(x: pd.DataFrame) -> np.ndarray:
    return x.to_numpy(dtype=float, copy=True)


def fit_embedding(
    x_train: pd.DataFrame,
    cfg: EmbeddingConfig,
) -> tuple[EmbeddingModel, pd.DataFrame, dict[str, Any]]:
    """Fit embedding on train only and return embedded train matrix."""
    feature_names = list(x_train.columns)
    x = _to_matrix(x_train)
    scaler: StandardScaler | None = None
    if cfg.standardize:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    method = cfg.method.lower()
    diagnostics: dict[str, Any] = {"method": method}
    if method == "identity":
        emb = x
        model = EmbeddingModel(method=method, scaler=scaler, pca=None, feature_names=feature_names)
        emb_cols = [f"emb_{i:02d}" for i in range(emb.shape[1])]
        return model, pd.DataFrame(emb, columns=emb_cols, index=x_train.index), diagnostics

    if method != "pca":
        raise ValueError(f"Unsupported embedding method: {cfg.method}")

    pca = PCA(
        n_components=min(cfg.max_components, x.shape[1]),
        svd_solver="full",
        random_state=None,
    )
    pca.fit(x)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    keep = int(np.searchsorted(cumulative, cfg.pca_variance_threshold) + 1)
    keep = max(2, min(keep, pca.n_components_))
    pca_final = PCA(n_components=keep, svd_solver="full", random_state=None)
    emb = pca_final.fit_transform(x)

    emb_cols = [f"emb_{i:02d}" for i in range(emb.shape[1])]
    model = EmbeddingModel(method=method, scaler=scaler, pca=pca_final, feature_names=feature_names)

    diagnostics["explained_variance_ratio"] = pca_final.explained_variance_ratio_.tolist()
    diagnostics["explained_variance_cumulative"] = float(np.sum(pca_final.explained_variance_ratio_))
    diagnostics["n_components"] = int(keep)
    diagnostics["top_loadings"] = _top_loadings(pca_final, feature_names, top_n=8)
    return model, pd.DataFrame(emb, columns=emb_cols, index=x_train.index), diagnostics


def transform_embedding(model: EmbeddingModel, x: pd.DataFrame) -> pd.DataFrame:
    """Apply fitted embedding model to new data."""
    missing = set(model.feature_names) - set(x.columns)
    if missing:
        raise ValueError(f"Input data is missing feature columns required by embedding: {sorted(missing)}")
    mat = x[model.feature_names].to_numpy(dtype=float, copy=True)
    if model.scaler is not None:
        mat = model.scaler.transform(mat)
    if model.pca is not None:
        mat = model.pca.transform(mat)
    emb_cols = [f"emb_{i:02d}" for i in range(mat.shape[1])]
    return pd.DataFrame(mat, columns=emb_cols, index=x.index)


def _top_loadings(pca: PCA, feature_names: list[str], top_n: int = 8) -> dict[str, list[dict[str, float | str]]]:
    loadings: dict[str, list[dict[str, float | str]]] = {}
    components = pca.components_
    for i, comp in enumerate(components):
        idx = np.argsort(np.abs(comp))[::-1][:top_n]
        key = f"pc_{i+1}"
        loadings[key] = [{"feature": feature_names[j], "loading": float(comp[j])} for j in idx]
    return loadings

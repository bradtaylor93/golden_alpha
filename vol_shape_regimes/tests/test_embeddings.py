"""Tests for embedding module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vol_shape_regimes.config import EmbeddingConfig
from vol_shape_regimes.embeddings import fit_embedding, transform_embedding


def test_pca_embedding_shapes() -> None:
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(200, 24)), columns=[f"f{i}" for i in range(24)])
    cfg = EmbeddingConfig(method="pca", pca_variance_threshold=0.9, max_components=8)
    model, x_train_emb, _diag = fit_embedding(x, cfg)
    xt = transform_embedding(model, x)
    assert x_train_emb.shape[0] == x.shape[0]
    assert xt.shape[0] == x.shape[0]
    assert xt.shape[1] <= 8
    assert xt.shape[1] > 0


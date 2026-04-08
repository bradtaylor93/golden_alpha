from __future__ import annotations

import numpy as np
import pandas as pd

from vol_shape_regimes.clustering import assign_states, fit_clusterer
from vol_shape_regimes.config import ClusteringConfig


def test_fit_clusterer_kmeans() -> None:
    x_train = np.vstack(
        [
            np.random.default_rng(0).normal(-1.0, 0.2, size=(50, 4)),
            np.random.default_rng(1).normal(1.0, 0.2, size=(50, 4)),
        ]
    )
    x_test = np.random.default_rng(2).normal(0.0, 1.0, size=(20, 4))
    train_df = pd.DataFrame(x_train, columns=["a", "b", "c", "d"])
    test_df = pd.DataFrame(x_test, columns=["a", "b", "c", "d"])

    cfg = ClusteringConfig(methods=("kmeans",), k_min=2, k_max=3, random_state=42, min_state_occupancy=0.05)
    model, train_states, _diag = fit_clusterer(train_df, cfg)
    test_states = assign_states(model, test_df)
    assert model.k in {2, 3}
    assert len(train_states) == 100
    assert len(test_states) == 20

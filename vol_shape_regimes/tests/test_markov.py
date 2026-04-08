from __future__ import annotations

import numpy as np

from vol_shape_regimes.markov import fit_markov_model


def test_markov_row_probabilities_sum_to_one() -> None:
    states = np.array([0, 0, 1, 1, 1, 2, 2, 1], dtype=int)
    model = fit_markov_model(states, n_states=3, order=1, smoothing=1e-3)
    row_sums = model.transition_1.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


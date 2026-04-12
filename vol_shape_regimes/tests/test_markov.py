from __future__ import annotations

import numpy as np
import pandas as pd

from vol_shape_regimes.markov import (
    fit_binary_event_model,
    fit_markov_model,
    fit_transition_type_model,
    predict_binary_event,
    predict_transition_type,
)


def test_markov_row_probabilities_sum_to_one() -> None:
    states = np.array([0, 0, 1, 1, 1, 2, 2, 1], dtype=int)
    model = fit_markov_model(states, n_states=3, order=1, smoothing=1e-3)
    row_sums = model.transition_1.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_event_and_transition_models_smoke() -> None:
    n = 120
    x = pd.DataFrame(
        {
            "f0": np.linspace(-1, 1, n),
            "f1": np.sin(np.linspace(0, 6, n)),
            "f2": np.random.default_rng(0).normal(size=n),
        }
    )
    y_event = (x["f0"] + 0.25 * x["f1"] > 0).astype(int)
    frame_event = x.copy()
    frame_event["event"] = y_event
    m_event = fit_binary_event_model(frame_event, ["f0", "f1", "f2"], "event")
    p_event, yhat_event = predict_binary_event(m_event, frame_event)
    assert len(p_event) == n
    assert len(yhat_event) == n

    y_type = np.where(x["f0"] > 0.3, "1->2", np.where(x["f0"] < -0.3, "1->0", "1->1"))
    frame_type = x.copy()
    frame_type["tt"] = y_type
    m_type = fit_transition_type_model(frame_type, ["f0", "f1", "f2"], "tt")
    p_type, yhat_type = predict_transition_type(m_type, frame_type)
    assert p_type.shape[0] == n
    assert len(yhat_type) == n


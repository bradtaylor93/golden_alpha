"""Baseline models for next-state prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vol_shape_regimes.markov import MarkovModel, fit_markov_model


def persistence_baseline(current_states: np.ndarray) -> np.ndarray:
    return np.asarray(current_states, dtype=int)


def empirical_transition_baseline(markov_model: MarkovModel, current_states: np.ndarray) -> np.ndarray:
    preds = [markov_model.predict(int(s)) for s in current_states]
    return np.asarray(preds, dtype=int)


def random_frequency_baseline(
    n_states: int,
    n_obs: int,
    state_freq: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    probs = np.asarray(state_freq, dtype=float)
    probs = probs / max(float(np.sum(probs)), 1e-12)
    return rng.choice(np.arange(n_states, dtype=int), size=n_obs, p=probs)


def scalar_volatility_states(
    vol_values: pd.Series,
    train_mask: pd.Series,
    n_states: int,
) -> np.ndarray:
    """Create scalar-volatility states using train quantile bins."""
    vol = pd.to_numeric(vol_values, errors="coerce").ffill().fillna(0.0)
    train_vol = vol.loc[train_mask]
    qs = np.linspace(0.0, 1.0, n_states + 1)
    edges = np.quantile(train_vol, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Avoid duplicate bins by enforcing monotonic growth.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    labels = np.digitize(vol.to_numpy(dtype=float), bins=edges[1:-1], right=False).astype(int)
    return labels


def scalar_volatility_markov_predictions(
    scalar_states: np.ndarray,
    train_end_idx: int,
    n_states: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, MarkovModel]:
    """Predict next state on test from scalar-vol states using first-order Markov."""
    train_states = scalar_states[: train_end_idx + 1]
    model = fit_markov_model(train_states, order=1, n_states=n_states, smoothing=smoothing)
    # Predict y_{t+1} for t in test_range[:-1]
    current = scalar_states[train_end_idx:-1]
    preds = np.asarray([model.predict(int(s)) for s in current], dtype=int)
    y_true = scalar_states[train_end_idx + 1 :]
    return y_true, preds, model

"""Markov-style transition models for volatility states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class MarkovModel:
    order: int
    n_states: int
    smoothing: float
    transition_1: np.ndarray
    transition_2: np.ndarray | None = None
    state_freq: np.ndarray | None = None

    def predict_proba(self, current_state: int, previous_state: int | None = None) -> np.ndarray:
        if self.order == 1 or previous_state is None or self.transition_2 is None:
            return self.transition_1[int(current_state)]
        return self.transition_2[int(previous_state), int(current_state)]

    def predict(self, current_state: int, previous_state: int | None = None) -> int:
        return int(np.argmax(self.predict_proba(current_state, previous_state)))


def fit_markov_model(
    states_train: np.ndarray,
    order: int = 1,
    n_states: int | None = None,
    smoothing: float = 1.0,
) -> MarkovModel:
    states = np.asarray(states_train, dtype=int)
    if len(states) < 3:
        raise ValueError("Need at least 3 state observations to fit transition model.")
    k = int(np.max(states) + 1) if n_states is None else int(n_states)
    if k <= 1:
        raise ValueError("Need at least 2 states.")
    if order not in (1, 2):
        raise ValueError("Only first-order and second-order Markov models are supported.")

    counts_1 = np.full((k, k), float(smoothing), dtype=float)
    for i in range(len(states) - 1):
        counts_1[states[i], states[i + 1]] += 1.0
    trans_1 = counts_1 / counts_1.sum(axis=1, keepdims=True)

    state_freq = np.bincount(states, minlength=k).astype(float)
    state_freq = state_freq / max(float(np.sum(state_freq)), 1.0)

    if order == 1:
        return MarkovModel(order=1, n_states=k, smoothing=smoothing, transition_1=trans_1, state_freq=state_freq)

    counts_2 = np.full((k, k, k), float(smoothing), dtype=float)
    for i in range(1, len(states) - 1):
        s_prev = states[i - 1]
        s_cur = states[i]
        s_next = states[i + 1]
        counts_2[s_prev, s_cur, s_next] += 1.0
    trans_2 = counts_2 / counts_2.sum(axis=2, keepdims=True)
    return MarkovModel(
        order=2,
        n_states=k,
        smoothing=smoothing,
        transition_1=trans_1,
        transition_2=trans_2,
        state_freq=state_freq,
    )


def transition_matrix_to_frame(matrix: np.ndarray, prefix: str = "state") -> pd.DataFrame:
    k = matrix.shape[0]
    cols = [f"{prefix}_{j}" for j in range(k)]
    rows = [f"{prefix}_{i}" for i in range(k)]
    return pd.DataFrame(matrix, columns=cols, index=rows)


@dataclass
class ConditionalTransitionModel:
    model: LogisticRegression
    feature_cols: list[str]
    n_states: int


@dataclass
class BinaryEventModel:
    model: LogisticRegression
    feature_cols: list[str]
    threshold: float


@dataclass
class TransitionTypeModel:
    model: LogisticRegression
    feature_cols: list[str]
    classes_: np.ndarray


def fit_conditional_transition_model(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_states: int,
    max_iter: int = 500,
) -> ConditionalTransitionModel:
    if frame.empty:
        raise ValueError("Cannot fit conditional transition model on empty frame.")
    x = frame[feature_cols].to_numpy(dtype=float)
    y = frame[target_col].to_numpy(dtype=int)
    # scikit-learn >=1.8 removed explicit multi_class arg from constructor.
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter,
        random_state=0,
    )
    clf.fit(x, y)
    return ConditionalTransitionModel(model=clf, feature_cols=feature_cols, n_states=n_states)


def predict_conditional_next_state(
    model: ConditionalTransitionModel,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    x = frame[model.feature_cols].to_numpy(dtype=float)
    probs = model.model.predict_proba(x)
    preds = np.argmax(probs, axis=1).astype(int)
    return probs, preds


def fit_binary_event_model(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    class_weight: str | None = "balanced",
    max_iter: int = 500,
    threshold: float = 0.5,
) -> BinaryEventModel:
    """Fit binary logistic model for event probability prediction."""
    if frame.empty:
        raise ValueError("Cannot fit binary event model on empty frame.")
    x = frame[feature_cols].to_numpy(dtype=float)
    y = frame[target_col].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("Binary event model needs both classes in train set.")
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter,
        random_state=0,
        class_weight=class_weight,
    )
    clf.fit(x, y)
    return BinaryEventModel(model=clf, feature_cols=feature_cols, threshold=float(threshold))


def predict_binary_event(
    model: BinaryEventModel,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probability_of_event, hard_prediction)."""
    x = frame[model.feature_cols].to_numpy(dtype=float)
    prob = model.model.predict_proba(x)[:, 1]
    pred = (prob >= model.threshold).astype(int)
    return prob.astype(float), pred


def fit_transition_type_model(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    class_weight: str | None = None,
    max_iter: int = 500,
) -> TransitionTypeModel:
    """Fit multinomial logistic model for transition-type classification."""
    if frame.empty:
        raise ValueError("Cannot fit transition-type model on empty frame.")
    x = frame[feature_cols].to_numpy(dtype=float)
    y = frame[target_col].astype(str).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Transition-type model needs at least 2 classes in train set.")
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter,
        random_state=0,
        class_weight=class_weight,
    )
    clf.fit(x, y)
    return TransitionTypeModel(model=clf, feature_cols=feature_cols, classes_=clf.classes_)


def predict_transition_type(
    model: TransitionTypeModel,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (class_probs, class_predictions) for transition-type model."""
    x = frame[model.feature_cols].to_numpy(dtype=float)
    probs = model.model.predict_proba(x)
    preds = model.model.predict(x)
    return probs.astype(float), preds.astype(str)


def fit_hazard_change_model(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    class_weight: str | None = "balanced",
    max_iter: int = 500,
    threshold: float = 0.5,
) -> BinaryEventModel:
    """Fit hazard-style binary model for next-step regime break probability."""
    return fit_binary_event_model(
        frame=frame,
        feature_cols=feature_cols,
        target_col=target_col,
        class_weight=class_weight,
        max_iter=max_iter,
        threshold=threshold,
    )


def predict_hazard_change(
    model: BinaryEventModel,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict hazard probability and hard change/no-change labels."""
    return predict_binary_event(model, frame)


def path_log_likelihood(
    model: MarkovModel,
    states: np.ndarray,
) -> float:
    states = np.asarray(states, dtype=int)
    if len(states) < 2:
        return float("nan")
    logp = 0.0
    eps = 1e-12
    if model.order == 1:
        for i in range(len(states) - 1):
            logp += float(np.log(max(model.transition_1[states[i], states[i + 1]], eps)))
    else:
        for i in range(1, len(states) - 1):
            logp += float(np.log(max(model.transition_2[states[i - 1], states[i], states[i + 1]], eps)))
    return logp


def transition_entropy(matrix: np.ndarray) -> float:
    eps = 1e-12
    m = np.clip(matrix, eps, 1.0)
    row_ent = -np.sum(m * np.log(m), axis=1)
    return float(np.mean(row_ent))

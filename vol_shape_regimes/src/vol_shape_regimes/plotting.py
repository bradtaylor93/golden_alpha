"""Plotting utilities for experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_embedding_scatter(
    emb: pd.DataFrame,
    states: np.ndarray,
    dates: pd.Series,
    output_path: Path,
) -> None:
    x = emb.to_numpy(dtype=float)
    if x.shape[1] < 2:
        x = np.c_[x[:, 0], np.zeros(len(x))]
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x[:, 0], x[:, 1], c=states, cmap="tab10", s=10, alpha=0.8)
    plt.colorbar(sc, label="state")
    plt.title("Embedding (first 2 dimensions) colored by state")
    plt.xlabel("emb_00")
    plt.ylabel("emb_01")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    # Time-colored map for temporal drift.
    t = pd.to_datetime(dates).map(pd.Timestamp.toordinal).to_numpy()
    plt.figure(figsize=(8, 6))
    sc2 = plt.scatter(x[:, 0], x[:, 1], c=t, cmap="viridis", s=10, alpha=0.8)
    plt.colorbar(sc2, label="time (ordinal)")
    plt.title("Embedding colored by time")
    plt.xlabel("emb_00")
    plt.ylabel("emb_01")
    plt.tight_layout()
    plt.savefig(output_path.with_name("embedding_time.png"), dpi=150)
    plt.close()


def plot_transition_heatmap(matrix: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="magma", aspect="auto")
    plt.colorbar(label="transition probability")
    plt.title("Transition Matrix Heatmap")
    plt.xlabel("next state")
    plt.ylabel("current state")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_state_timeline(
    dates: pd.Series,
    close: pd.Series,
    states: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(pd.to_datetime(dates), pd.to_numeric(close, errors="coerce"), color="black", linewidth=1.2)
    ax1.set_ylabel("price")
    ax1.set_title("State timeline over price")

    ax2 = ax1.twinx()
    ax2.step(pd.to_datetime(dates), states, where="post", color="tab:orange", alpha=0.7)
    ax2.set_ylabel("state")
    ax2.set_yticks(sorted(np.unique(states)))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_dwell_histogram(dwell_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    for state, grp in dwell_df.groupby("state"):
        plt.hist(grp["dwell_len"], bins=20, alpha=0.5, label=f"state {state}")
    plt.legend()
    plt.xlabel("dwell length")
    plt.ylabel("count")
    plt.title("Dwell time histogram by state")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    *,
    title: str = "Next-state confusion matrix",
) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(label="count")
    plt.title(title)
    plt.xlabel("predicted")
    plt.ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

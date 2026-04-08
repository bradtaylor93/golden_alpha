from __future__ import annotations

from pathlib import Path

from vol_shape_regimes.config import (
    ClusteringConfig,
    DataConfig,
    EmbeddingConfig,
    EvaluationConfig,
    ExperimentConfig,
    FeatureConfig,
    OutputConfig,
    TransitionConfig,
    VolatilityConfig,
)
from vol_shape_regimes.pipeline import run_experiment


def test_pipeline_smoke(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        experiment_name="smoke",
        data=DataConfig(source="synthetic", synthetic_rows=450, random_state=11),
        volatility=VolatilityConfig(method="rolling_std", rolling_window=10, ewma_span=12),
        features=FeatureConfig(
            lookback_window=30,
            modes=("summary", "quantile"),
            quantile_points=10,
            histogram_bins=8,
            temporal_downsample=8,
            temporal_dct_components=4,
            pacf_lags=2,
            spike_quantile=0.9,
        ),
        embedding=EmbeddingConfig(method="pca", standardize=True, pca_variance_threshold=0.9, max_components=8),
        clustering=ClusteringConfig(methods=("kmeans",), k_min=3, k_max=3, min_state_occupancy=0.02),
        transitions=TransitionConfig(markov_order=1, fit_second_order=False, fit_conditional_model=False),
        evaluation=EvaluationConfig(train_ratio=0.7, top_k_accuracy=2),
        output=OutputConfig(base_dir=str(tmp_path / "out")),
    )
    metrics = run_experiment(cfg)
    out = tmp_path / "out" / "smoke"
    assert (out / "cluster_summaries.csv").exists()
    assert (out / "transition_matrix_order1.csv").exists()
    assert (out / "evaluation_metrics.json").exists()
    assert isinstance(metrics, dict)


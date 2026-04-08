"""CLI entrypoint for running the default experiment config."""

from __future__ import annotations

from pathlib import Path

from vol_shape_regimes.config import load_experiment_config
from vol_shape_regimes.pipeline import run_experiment


def main() -> None:
    config_path = Path("/workspace/vol_shape_regimes/configs/experiment_default.yaml")
    cfg = load_experiment_config(config_path)
    metrics = run_experiment(cfg)
    print("experiment_name:", cfg.experiment_name)
    print("output_dir:", f"{cfg.output.base_dir}/{cfg.experiment_name}")
    print("markov_order1_accuracy:", metrics["markov"]["accuracy"])
    print("persistence_accuracy:", metrics["baselines"]["persistence"]["accuracy"])


if __name__ == "__main__":
    main()


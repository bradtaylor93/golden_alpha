"""Test whether multi-asset global training improves SPY performance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.data.catalog import DataCatalog
from trading_research.data.vendors.yahoo import YahooMarketDataVendor
from trading_research.recipes.champion_edge import ChampionEdgeRecipe
from trading_research.workflow.inspectors import load_run
from trading_research.workflow.runner import WorkflowRunner


CANDIDATE_NODES = [
    "ret_persistence_baseline",
    "ret_loess_base",
    "ret_kernel_rich",
    "ret_ridge_stable",
    "champion_meta_return",
]


def _oos_metrics(pred: pd.DataFrame, *, asset: str | None = None) -> dict[str, float]:
    test = pred[pred["split_role"] == "test"].copy()
    if asset is not None:
        test = test[test["asset"] == asset]
    if test.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson_corr": float("nan"), "n_test": 0.0}
    test["target"] = pd.to_numeric(test["target"], errors="coerce")
    test["prediction"] = pd.to_numeric(test["prediction"], errors="coerce")
    test = test.dropna(subset=["target", "prediction"])
    if test.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson_corr": float("nan"), "n_test": 0.0}
    err = test["target"] - test["prediction"]
    return {
        "mae": float(err.abs().mean()),
        "rmse": float((err.pow(2).mean()) ** 0.5),
        "pearson_corr": float(test["target"].corr(test["prediction"], method="pearson")),
        "n_test": float(len(test)),
    }


def _collect_metrics(run_id: str, runs_root: Path, *, asset: str | None = None) -> tuple[pd.DataFrame, str]:
    inspector = load_run(run_id, runs_root)
    rows: list[dict[str, Any]] = []
    for node in CANDIDATE_NODES:
        frame = inspector.load_predictions(node)
        rows.append({"node_name": node, **_oos_metrics(frame, asset=asset)})
    selector = inspector.load_artifact("champion_selector", artifact_type="SelectionArtifact")
    chosen = str(selector["chosen_node_name"].iloc[0]) if not selector.empty else "n/a"
    out = pd.DataFrame(rows).sort_values("rmse", ascending=True).reset_index(drop=True)
    return out, chosen


def _run_recipe(
    *,
    recipe: ChampionEdgeRecipe,
    catalog: DataCatalog,
    runs_root: Path,
    tag: str,
) -> str:
    runner = WorkflowRunner(runs_dir=runs_root, data_catalog=catalog)
    run_id = runner.run(
        recipe.compile(),
        run_config={
            "recipe": "ChampionEdgeRecipe",
            "tag": tag,
            "hypothesis": "global_multi_asset_learning_may_improve_spy",
        },
    )
    return run_id


def main() -> None:
    root = Path("trading_research/examples/_output/13_global_learning_asset_expansion")
    data_dir = root / "data"
    runs_dir = root / "runs"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    catalog = DataCatalog(data_dir)

    vendor = YahooMarketDataVendor()
    single_assets = ["SPY"]
    multi_assets = [
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "XLY",
        "TLT",
        "GLD",
        "HYG",
    ]

    bars_single = vendor.fetch_bars(single_assets, period="2y", interval="1h")
    bars_multi = vendor.fetch_bars(multi_assets, period="2y", interval="1h")
    available_multi_assets = sorted(bars_multi["asset"].astype(str).unique().tolist()) if not bars_multi.empty else []

    catalog.persist_processed_bars(
        "spy_2y_1h_single",
        bars_single,
        metadata={"source": "yahoo", "period": "2y", "interval": "1h", "assets": single_assets},
    )
    catalog.persist_processed_bars(
        "multi_2y_1h_global",
        bars_multi,
        metadata={
            "source": "yahoo",
            "period": "2y",
            "interval": "1h",
            "assets_requested": multi_assets,
            "assets_available": available_multi_assets,
        },
    )

    single_recipe = ChampionEdgeRecipe(
        dataset_name="spy_2y_1h_single",
        universe=tuple(single_assets),
        run_name="champion_single_spy",
    )
    multi_recipe = ChampionEdgeRecipe(
        dataset_name="multi_2y_1h_global",
        universe=tuple(available_multi_assets or multi_assets),
        run_name="champion_multi_global",
    )

    run_single = _run_recipe(
        recipe=single_recipe,
        catalog=catalog,
        runs_root=runs_dir,
        tag="single_spy",
    )
    run_multi = _run_recipe(
        recipe=multi_recipe,
        catalog=catalog,
        runs_root=runs_dir,
        tag="multi_global",
    )

    single_spy, single_selected = _collect_metrics(run_single, runs_dir, asset="SPY")
    multi_spy, multi_selected = _collect_metrics(run_multi, runs_dir, asset="SPY")
    multi_global, _ = _collect_metrics(run_multi, runs_dir, asset=None)

    compare = (
        single_spy[["node_name", "rmse", "mae", "pearson_corr"]]
        .rename(
            columns={
                "rmse": "single_spy_rmse",
                "mae": "single_spy_mae",
                "pearson_corr": "single_spy_corr",
            }
        )
        .merge(
            multi_spy[["node_name", "rmse", "mae", "pearson_corr"]].rename(
                columns={
                    "rmse": "multi_spy_rmse",
                    "mae": "multi_spy_mae",
                    "pearson_corr": "multi_spy_corr",
                }
            ),
            on="node_name",
            how="inner",
        )
    )
    compare["rmse_delta_multi_minus_single"] = compare["multi_spy_rmse"] - compare["single_spy_rmse"]
    compare["mae_delta_multi_minus_single"] = compare["multi_spy_mae"] - compare["single_spy_mae"]
    compare = compare.sort_values("rmse_delta_multi_minus_single", ascending=True).reset_index(drop=True)

    def _safe_lookup(df: pd.DataFrame, node: str, col: str) -> float:
        row = df[df["node_name"] == node]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][col])

    summary = {
        "run_single": run_single,
        "run_multi": run_multi,
        "single_selected_model": single_selected,
        "multi_selected_model": multi_selected,
        "assets_requested_multi": multi_assets,
        "assets_available_multi": available_multi_assets,
        "rows_single_dataset": int(len(bars_single)),
        "rows_multi_dataset": int(len(bars_multi)),
        "spy_meta_rmse_single": _safe_lookup(single_spy, "champion_meta_return", "rmse"),
        "spy_meta_rmse_multi": _safe_lookup(multi_spy, "champion_meta_return", "rmse"),
        "spy_meta_rmse_delta_multi_minus_single": _safe_lookup(multi_spy, "champion_meta_return", "rmse")
        - _safe_lookup(single_spy, "champion_meta_return", "rmse"),
        "spy_best_rmse_single": float(single_spy["rmse"].min()),
        "spy_best_rmse_multi": float(multi_spy["rmse"].min()),
        "spy_best_rmse_delta_multi_minus_single": float(multi_spy["rmse"].min() - single_spy["rmse"].min()),
    }

    single_spy.to_csv(reports_dir / "single_spy_candidates.csv", index=False)
    multi_spy.to_csv(reports_dir / "multi_spy_candidates.csv", index=False)
    multi_global.to_csv(reports_dir / "multi_global_candidates.csv", index=False)
    compare.to_csv(reports_dir / "spy_single_vs_multi_comparison.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("single_run_id:", run_single)
    print("multi_run_id:", run_multi)
    print("reports:", reports_dir.resolve())
    print(
        "spy_meta_rmse_single:",
        summary["spy_meta_rmse_single"],
        "spy_meta_rmse_multi:",
        summary["spy_meta_rmse_multi"],
        "delta:",
        summary["spy_meta_rmse_delta_multi_minus_single"],
    )
    print(
        "spy_best_rmse_single:",
        summary["spy_best_rmse_single"],
        "spy_best_rmse_multi:",
        summary["spy_best_rmse_multi"],
        "delta:",
        summary["spy_best_rmse_delta_multi_minus_single"],
    )


if __name__ == "__main__":
    main()

"""Run regime-aware local experts with learned soft gating."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.analysis.sanity_checks import run_sanity_checks
from trading_research.data.catalog import DataCatalog
from trading_research.data.vendors.yahoo import YahooMarketDataVendor
from trading_research.recipes.regime_aware_moe import RegimeAwareMoERecipe
from trading_research.workflow.inspectors import load_run
from trading_research.workflow.runner import WorkflowRunner


def _test_frame(inspector, node_name: str) -> pd.DataFrame:
    frame = inspector.load_predictions(node_name)
    out = frame[frame["split_role"] == "test"][["timestamp", "asset", "outer_fold_id", "target", "prediction"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["asset"] = out["asset"].astype(str)
    out = out.dropna(subset=["timestamp", "asset"])
    out = out.groupby(["timestamp", "asset", "outer_fold_id"], as_index=False).last()
    return out


def _model_metrics(df: pd.DataFrame, pred_col: str = "prediction") -> dict[str, float]:
    if df.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson_corr": float("nan"), "n_test": 0.0}
    err = pd.to_numeric(df["target"], errors="coerce") - pd.to_numeric(df[pred_col], errors="coerce")
    keep = ~err.isna()
    err = err[keep]
    if err.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson_corr": float("nan"), "n_test": 0.0}
    yt = pd.to_numeric(df.loc[keep, "target"], errors="coerce")
    yp = pd.to_numeric(df.loc[keep, pred_col], errors="coerce")
    return {
        "mae": float(err.abs().mean()),
        "rmse": float((err.pow(2).mean()) ** 0.5),
        "pearson_corr": float(yt.corr(yp, method="pearson")),
        "n_test": float(len(err)),
    }


def _softmax_weights(err_df: pd.DataFrame, temp: float = 0.75) -> pd.DataFrame:
    x = err_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x = -x / max(temp, 1e-9)
    x = x.clip(-50, 50)
    exp_x = np.exp(x)
    denom = exp_x.sum(axis=1)
    denom = denom.replace(0.0, np.nan)
    w = exp_x.div(denom, axis=0).fillna(1.0 / exp_x.shape[1])
    return w


def main() -> None:
    root = Path("trading_research/examples/_output/15_regime_aware_moe_gating")
    data_dir = root / "data"
    runs_dir = root / "runs"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    vendor = YahooMarketDataVendor()
    bars = vendor.fetch_bars(["SPY"], period="2y", interval="1h")
    catalog = DataCatalog(data_dir)
    dataset_name = "spy_2y_1h_moe"
    catalog.persist_processed_bars(
        dataset_name,
        bars,
        metadata={"source": "yahoo", "period": "2y", "interval": "1h", "assets": ["SPY"]},
    )

    recipe = RegimeAwareMoERecipe(dataset_name=dataset_name, universe=("SPY",))
    runner = WorkflowRunner(runs_dir=runs_dir, data_catalog=catalog)
    run_id = runner.run(
        recipe.compile(),
        run_config={
            "recipe": "RegimeAwareMoERecipe",
            "description": "Asset-specific local experts with learned error gating.",
        },
    )
    inspector = load_run(run_id, runs_dir)

    trend = _test_frame(inspector, "expert_trend_ridge").rename(columns={"prediction": "pred_trend"})
    meanrev = _test_frame(inspector, "expert_meanrev_loess").rename(columns={"prediction": "pred_meanrev"})
    nonlinear = _test_frame(inspector, "expert_nonlinear_kernel").rename(columns={"prediction": "pred_nonlinear"})

    gate_trend = _test_frame(inspector, "gate_abs_err_trend").rename(columns={"prediction": "gate_err_trend"})
    gate_meanrev = _test_frame(inspector, "gate_abs_err_mean_reversion").rename(
        columns={"prediction": "gate_err_meanrev"}
    )
    gate_nonlinear = _test_frame(inspector, "gate_abs_err_nonlinear").rename(
        columns={"prediction": "gate_err_nonlinear"}
    )
    vol_context = _test_frame(inspector, "vol_context").rename(columns={"prediction": "pred_vol_context"})

    key_cols = ["timestamp", "asset", "outer_fold_id"]
    merged = trend[key_cols + ["target", "pred_trend"]].merge(meanrev[key_cols + ["pred_meanrev"]], on=key_cols)
    merged = merged.merge(nonlinear[key_cols + ["pred_nonlinear"]], on=key_cols)
    merged = merged.merge(gate_trend[key_cols + ["gate_err_trend"]], on=key_cols, how="left")
    merged = merged.merge(gate_meanrev[key_cols + ["gate_err_meanrev"]], on=key_cols, how="left")
    merged = merged.merge(gate_nonlinear[key_cols + ["gate_err_nonlinear"]], on=key_cols, how="left")
    merged = merged.merge(vol_context[key_cols + ["pred_vol_context"]], on=key_cols, how="left")

    err_cols = ["gate_err_trend", "gate_err_meanrev", "gate_err_nonlinear"]
    weights = _softmax_weights(merged[err_cols], temp=0.75)
    weights.columns = ["w_trend", "w_meanrev", "w_nonlinear"]
    merged = pd.concat([merged.reset_index(drop=True), weights.reset_index(drop=True)], axis=1)

    # Regime-aware tilt: in higher predicted-vol regimes, increase mean-reversion weight.
    vol = pd.to_numeric(merged["pred_vol_context"], errors="coerce").fillna(0.0)
    vol_rank = vol.rank(pct=True)
    boost = (vol_rank > 0.7).astype(float) * 0.20
    merged["w_meanrev"] = merged["w_meanrev"] * (1.0 + boost)
    wsum = merged[["w_trend", "w_meanrev", "w_nonlinear"]].sum(axis=1).replace(0.0, np.nan)
    merged["w_trend"] = merged["w_trend"] / wsum
    merged["w_meanrev"] = merged["w_meanrev"] / wsum
    merged["w_nonlinear"] = merged["w_nonlinear"] / wsum

    merged["prediction"] = (
        merged["w_trend"] * merged["pred_trend"]
        + merged["w_meanrev"] * merged["pred_meanrev"]
        + merged["w_nonlinear"] * merged["pred_nonlinear"]
    )

    rows = []
    for node_name, col in [
        ("expert_trend_ridge", "pred_trend"),
        ("expert_meanrev_loess", "pred_meanrev"),
        ("expert_nonlinear_kernel", "pred_nonlinear"),
        ("moe_learned_gating", "prediction"),
    ]:
        rows.append({"node_name": node_name, **_model_metrics(merged, pred_col=col)})
    perf = pd.DataFrame(rows).sort_values("rmse", ascending=True).reset_index(drop=True)

    vol_bucket = pd.qcut(vol_rank, q=4, labels=["Q1_low_vol", "Q2", "Q3", "Q4_high_vol"])
    weight_regime = (
        merged.assign(vol_bucket=vol_bucket)
        .groupby("vol_bucket", observed=True)[["w_trend", "w_meanrev", "w_nonlinear"]]
        .mean()
        .reset_index()
    )

    sanity = run_sanity_checks(run_id, runs_dir)
    sanity.to_csv(reports_dir / "no_leakage_assertions.csv", index=False)
    merged.to_parquet(reports_dir / "moe_predictions.parquet", index=False)
    perf.to_csv(reports_dir / "candidate_oos_performance.csv", index=False)
    weight_regime.to_csv(reports_dir / "gating_weights_by_regime.csv", index=False)

    summary = {
        "run_id": run_id,
        "best_model": str(perf.iloc[0]["node_name"]) if not perf.empty else "n/a",
        "best_rmse": float(perf.iloc[0]["rmse"]) if not perf.empty else float("nan"),
        "failed_sanity_checks": int((~sanity["passed"]).sum()),
        "critical_sanity_failures": int(((~sanity["passed"]) & (sanity["severity"] == "critical")).sum()),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("run_id:", run_id)
    print("reports:", reports_dir.resolve())
    if not perf.empty:
        print("best by RMSE:", perf.iloc[0]["node_name"], "rmse:", float(perf.iloc[0]["rmse"]))
    print(
        "sanity_failed_checks:",
        summary["failed_sanity_checks"],
        "critical:",
        summary["critical_sanity_failures"],
    )


if __name__ == "__main__":
    main()


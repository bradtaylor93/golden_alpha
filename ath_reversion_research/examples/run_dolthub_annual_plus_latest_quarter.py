"""Attach latest quarterly fundamentals to annual Dolt events.

This keeps annual events as the prediction unit and tests whether the most
recent quarterly/TTM fundamentals available before the annual trade date improve
the annual model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_dolthub_aligned_replication as aligned


OUTPUT_DIR = Path("ath_reversion_research/reports/dolthub_annual_plus_latest_quarter")
ANNUAL_EVENTS = Path("ath_reversion_research/reports/dolthub_aligned_replication/aligned_events.csv")
QUARTERLY_EVENTS = Path("ath_reversion_research/reports/dolthub_quarterly_trend_model/quarterly_trend_events.csv")

ANNUAL_FEATURES = aligned.FINANCIAL_FEATURES + aligned.PRICE_FEATURES
QUARTER_FEATURES = [
    "q_sales_qoq",
    "q_sales_yoy",
    "q_sales_yoy_accel",
    "ttm_sales_yoy",
    "ttm_sales_yoy_accel",
    "q_gross_margin",
    "q_operating_margin",
    "q_net_margin",
    "q_cfo_margin",
    "q_fcf_margin",
    "ttm_gross_margin",
    "ttm_operating_margin",
    "ttm_net_margin",
    "ttm_cfo_margin",
    "ttm_fcf_margin",
    "q_operating_margin_yoy_change",
    "q_fcf_margin_yoy_change",
    "ttm_operating_margin_yoy_change",
    "ttm_fcf_margin_yoy_change",
    "q_cfo_to_net_income",
    "q_fcf_to_net_income",
    "q_accruals_to_assets",
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annual = pd.read_csv(ANNUAL_EVENTS, parse_dates=["availability_date", "trade_date", "fwd_12m_return_end_date"])
    quarterly = pd.read_csv(QUARTERLY_EVENTS, parse_dates=["availability_date"])
    events = _attach_latest_quarter(annual, quarterly)
    events.to_csv(OUTPUT_DIR / "annual_with_latest_quarter_events.csv", index=False)
    predictions = []
    for feature_set, features in {
        "annual_price": ANNUAL_FEATURES,
        "annual_price_latest_quarter": ANNUAL_FEATURES + [f"latest_{f}" for f in QUARTER_FEATURES],
        "latest_quarter_only": [f"latest_{f}" for f in QUARTER_FEATURES] + aligned.PRICE_FEATURES,
    }.items():
        predictions.append(_walk_forward(events, feature_set, features))
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    perf = _performance(pred)
    perf.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    _write_report(perf)
    print(perf.to_string(index=False))
    return 0


def _attach_latest_quarter(annual: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    qcols = ["symbol", "availability_date", *QUARTER_FEATURES]
    q = quarterly[qcols].rename(columns={feature: f"latest_{feature}" for feature in QUARTER_FEATURES})
    pieces = []
    for symbol, group in annual.sort_values(["symbol", "availability_date"]).groupby("symbol"):
        qg = q[q["symbol"] == symbol].drop(columns=["symbol"]).sort_values("availability_date")
        if qg.empty:
            pieces.append(group)
            continue
        merged = pd.merge_asof(group.sort_values("availability_date"), qg, on="availability_date", direction="backward")
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def _walk_forward(events: pd.DataFrame, feature_set: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for bucket in ["all", "large_cap", "mid_cap"]:
        subset = events.copy() if bucket == "all" else events[events["market_cap_bucket"] == bucket].copy()
        subset = subset.dropna(subset=["fwd_12m_return", "fwd_12m_return_end_date", "revenue_growth_yoy"])
        for year in sorted(subset["trade_date"].dt.year.unique()):
            start = pd.Timestamp(f"{int(year)}-01-01")
            test = subset[subset["trade_date"].dt.year == year].copy()
            train = subset[(subset["trade_date"] < start) & (subset["fwd_12m_return_end_date"] < start)].copy()
            if len(train) < 100 or test.empty:
                continue
            test["prediction"] = _fit_predict(train, test, features)
            test["target_return"] = test["fwd_12m_return"]
            test["feature_set"] = feature_set
            test["regression_bucket"] = bucket
            test["test_year"] = int(year)
            rows.append(test)
    return pd.concat(rows, ignore_index=True)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    usable = [f for f in features if f in train and train[f].notna().mean() >= 0.40 and train[f].std(skipna=True) > 0]
    low = train[usable].quantile(0.01)
    high = train[usable].quantile(0.99)
    med = train[usable].median()
    x_train = train[usable].clip(lower=low, upper=high, axis=1).fillna(med)
    x_test = test[usable].clip(lower=low, upper=high, axis=1).fillna(med)
    mean = x_train.mean()
    std = x_train.std().replace(0, 1)
    x = ((x_train - mean) / std).fillna(0).to_numpy(float)
    t = ((x_test - mean) / std).fillna(0).to_numpy(float)
    y = train["fwd_12m_return"].to_numpy(float)
    y_mean = y.mean()
    x_mat = np.c_[np.ones(len(x)), x]
    t_mat = np.c_[np.ones(len(t)), t]
    penalty = np.eye(x_mat.shape[1])
    penalty[0, 0] = 0
    beta = np.linalg.solve(x_mat.T @ x_mat + 10.0 * penalty, x_mat.T @ (y - y_mean))
    beta[0] += y_mean
    return t_mat @ beta


def _performance(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_set, bucket), group in pred.groupby(["feature_set", "regression_bucket"]):
        top = group[group["prediction"] >= group["prediction"].quantile(0.80)]
        bottom = group[group["prediction"] <= group["prediction"].quantile(0.20)]
        baseline = ((group["target_return"] - group["target_return"].mean()) ** 2).sum()
        residual = ((group["target_return"] - group["prediction"]) ** 2).sum()
        rows.append({
            "feature_set": feature_set,
            "regression_bucket": bucket,
            "observations": len(group),
            "test_years": group["test_year"].nunique(),
            "oos_r2": 1 - residual / baseline if baseline > 0 else np.nan,
            "pearson_corr": group["prediction"].corr(group["target_return"]),
            "spearman_corr": group["prediction"].rank().corr(group["target_return"].rank()),
            "top_quintile_mean_return": top["target_return"].mean(),
            "top_quintile_median_return": top["target_return"].median(),
            "top_quintile_hit_rate": (top["target_return"] > 0).mean(),
            "bottom_quintile_mean_return": bottom["target_return"].mean(),
            "top_minus_bottom_mean": top["target_return"].mean() - bottom["target_return"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["regression_bucket", "spearman_corr"], ascending=[True, False])


def _write_report(perf: pd.DataFrame) -> None:
    lines = [
        "# DoltHub Annual + Latest Quarter Model",
        "",
        "Keeps annual events as the prediction unit and attaches latest available quarterly/TTM features before each annual trade date.",
        "",
        "## Performance",
        "",
        _markdown_table(perf),
        "",
        "## Interpretation",
        "",
        "- This tests the requested combination: annual + latest quarter + price.",
        "- If latest-quarter features do not improve the annual model, quarterly trends are not adding incremental ranking power in this setup.",
    ]
    (OUTPUT_DIR / "DOLTHUB_ANNUAL_PLUS_LATEST_QUARTER.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "test_years"}:
            if any(token in col for token in ["r2", "corr", "return", "rate"]):
                display[col] = display[col].map(lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())

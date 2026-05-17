"""Model comparison on exact Yahoo/Dolt matched rows.

This isolates whether Yahoo-vs-Dolt performance differences are caused by data
definition mismatches or by the longer Dolt history.  It uses only rows matched
by symbol and fiscal period end, then compares Yahoo-engineered features and
Dolt-engineered features on the same trade dates and forward returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("ath_reversion_research/reports/yahoo_dolt_matched_model_comparison")
MATCHED = Path("ath_reversion_research/reports/yahoo_dolt_feature_reconciliation/matched_rows.csv")

PRICE_CONTEXT = [
    "trailing_3m_return",
    "trailing_6m_return",
    "trailing_12m_return",
    "relative_6m_vs_spy",
    "realized_vol_3m",
    "drawdown_12m",
    "growth_x_trailing_6m",
    "growth_x_fcf_margin",
]

YAHOO_FEATURES = [
    "revenue_growth_yoy_yahoo",
    "gross_margin_yahoo",
    "operating_margin_yahoo",
    "net_margin_yahoo",
    "ebitda_margin_yahoo",
    "cfo_margin_yahoo",
    "fcf_margin_yahoo",
    "asset_turnover_yahoo",
    *PRICE_CONTEXT,
]
DOLT_FEATURES = [
    "revenue_growth_yoy_dolt",
    "gross_margin_dolt",
    "operating_margin_dolt",
    "net_margin_dolt",
    "ebitda_margin_dolt",
    "cfo_margin_dolt",
    "fcf_margin_dolt",
    "asset_turnover_dolt",
    *PRICE_CONTEXT,
]
YAHOO_FINANCIAL = [f for f in YAHOO_FEATURES if not any(token in f for token in ["trailing", "relative", "realized", "drawdown", "growth_x"])]
DOLT_FINANCIAL = [f for f in DOLT_FEATURES if not any(token in f for token in ["trailing", "relative", "realized", "drawdown", "growth_x"])]
YAHOO_PRICE = PRICE_CONTEXT
DOLT_PRICE = PRICE_CONTEXT


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matched = pd.read_csv(MATCHED, parse_dates=["trade_date_yahoo", "fwd_12m_return_end_date_yahoo"])
    matched = matched.rename(columns={"fwd_12m_return_yahoo": "target_return"})
    matched = matched.dropna(subset=["target_return", "trade_date_yahoo"])
    comparisons = {
        "yahoo_financial_only": YAHOO_FINANCIAL,
        "yahoo_price_only": YAHOO_PRICE,
        "yahoo_financial_plus_price": YAHOO_FEATURES,
        "dolt_financial_only": DOLT_FINANCIAL,
        "dolt_price_only": DOLT_PRICE,
        "dolt_financial_plus_price": DOLT_FEATURES,
    }
    predictions = []
    for name, features in comparisons.items():
        predictions.append(_walk_forward(matched, name, features))
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    perf = _performance(pred)
    perf.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    _write_report(matched, perf)
    print(perf.to_string(index=False))
    return 0


def _walk_forward(frame: pd.DataFrame, model_name: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for bucket in ["all", "large_cap", "mid_cap"]:
        subset = frame.copy() if bucket == "all" else frame[frame["market_cap_bucket_yahoo"] == bucket].copy()
        subset = subset.dropna(subset=["target_return", *[f for f in features if f in subset.columns]], how="all")
        for year in sorted(subset["trade_date_yahoo"].dt.year.unique()):
            start = pd.Timestamp(f"{int(year)}-01-01")
            test = subset[subset["trade_date_yahoo"].dt.year == year].copy()
            end_col = "fwd_12m_return_end_date_yahoo"
            train = subset[(subset["trade_date_yahoo"] < start) & (subset[end_col] < start)].copy()
            if len(train) < 25 or test.empty:
                continue
            test["prediction"] = _fit_predict(train, test, features)
            test["model"] = model_name
            test["bucket"] = bucket
            test["test_year"] = int(year)
            rows.append(test)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    usable = [f for f in features if f in train and train[f].notna().mean() >= 0.50 and train[f].astype(float).std(skipna=True) > 0]
    if not usable:
        usable = [features[0]]
    lo = train[usable].quantile(0.01)
    hi = train[usable].quantile(0.99)
    med = train[usable].median()
    xtr = train[usable].astype(float).clip(lower=lo, upper=hi, axis=1).fillna(med)
    xte = test[usable].astype(float).clip(lower=lo, upper=hi, axis=1).fillna(med)
    mean = xtr.mean()
    std = xtr.std().replace(0, 1)
    x = ((xtr - mean) / std).fillna(0).to_numpy(float)
    t = ((xte - mean) / std).fillna(0).to_numpy(float)
    y = train["target_return"].to_numpy(float)
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
    for (model, bucket), group in pred.groupby(["model", "bucket"]):
        top = group[group["prediction"] >= group["prediction"].quantile(0.80)]
        bottom = group[group["prediction"] <= group["prediction"].quantile(0.20)]
        baseline = ((group["target_return"] - group["target_return"].mean()) ** 2).sum()
        residual = ((group["target_return"] - group["prediction"]) ** 2).sum()
        rows.append({
            "model": model,
            "bucket": bucket,
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
    return pd.DataFrame(rows).sort_values(["bucket", "spearman_corr"], ascending=[True, False])


def _write_report(matched: pd.DataFrame, perf: pd.DataFrame) -> None:
    lines = [
        "# Yahoo vs Dolt Matched Model Comparison",
        "",
        "Runs models on exact matched Yahoo/Dolt rows to isolate data definitions from sample-period effects.",
        "",
        "## Dataset",
        "",
        f"- Matched rows with 12m target: {len(matched)}.",
        f"- Matched symbols: {matched['symbol'].nunique()}.",
        f"- Trade years: {sorted(matched['trade_date_yahoo'].dt.year.unique().tolist())}.",
        "",
        "## Model comparison",
        "",
        _markdown_table(perf),
        "",
        "## Interpretation",
        "",
        "- If Yahoo and Dolt financial+price behave similarly on matched rows, the longer-history difference is time/sample-period driven.",
        "- If Yahoo beats Dolt on matched rows, feature definitions are still materially different.",
        "- If price-only remains competitive on matched rows, price context is a major driver even in the recent Yahoo period.",
    ]
    (OUTPUT_DIR / "YAHOO_DOLT_MATCHED_MODEL_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "test_years"}:
            if any(token in col for token in ["r2", "corr", "return", "rate"]):
                display[col] = display[col].map(lambda v: f"{float(v) * 100:.2f}%" if pd.notna(v) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())

"""DoltHub model improvement and feature subset comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("ath_reversion_research/reports/dolthub_model_improvement")
EVENTS = Path("ath_reversion_research/reports/dolthub_full_fundamental_validation/dolt_full_feature_events.csv")
FEATURE_SETS = {
    "income_only": ["revenue_growth_yoy", "revenue_growth_accel", "gross_margin", "operating_margin", "net_margin", "ebitda_margin"],
    "price_only": ["trailing_3m_return", "trailing_6m_return", "trailing_12m_return", "relative_6m_vs_spy", "realized_vol_3m", "drawdown_12m"],
    "income_price": [
        "revenue_growth_yoy",
        "revenue_growth_accel",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "ebitda_margin",
        "trailing_3m_return",
        "trailing_6m_return",
        "trailing_12m_return",
        "relative_6m_vs_spy",
        "realized_vol_3m",
        "drawdown_12m",
    ],
    "growth_quality_price": [
        "revenue_growth_yoy",
        "revenue_growth_accel",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "fcf_margin",
        "cfo_margin",
        "trailing_6m_return",
        "trailing_12m_return",
        "relative_6m_vs_spy",
        "drawdown_12m",
        "growth_x_trailing_6m",
        "growth_x_fcf_margin",
    ],
    "full": [
        "revenue_growth_yoy",
        "revenue_growth_accel",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "ebitda_margin",
        "cfo_margin",
        "fcf_margin",
        "capex_to_revenue",
        "rd_to_revenue",
        "sga_to_revenue",
        "debt_to_assets",
        "debt_to_equity",
        "cash_to_assets",
        "current_ratio",
        "asset_turnover",
        "trailing_3m_return",
        "trailing_6m_return",
        "trailing_12m_return",
        "relative_6m_vs_spy",
        "realized_vol_3m",
        "drawdown_12m",
        "growth_x_trailing_6m",
        "growth_x_fcf_margin",
    ],
}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENTS, parse_dates=["trade_date", "fwd_3m_return_end_date", "fwd_6m_return_end_date", "fwd_12m_return_end_date"])
    predictions = []
    for name, features in FEATURE_SETS.items():
        predictions.append(_walk_forward(events, name, features))
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    perf = _performance(pred)
    cond = _conditions(pred)
    perf.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    cond.to_csv(OUTPUT_DIR / "condition_summary.csv", index=False)
    _write_report(perf, cond)
    print(perf.to_string(index=False))
    return 0


def _walk_forward(events: pd.DataFrame, feature_set: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for target in ["fwd_3m_return", "fwd_6m_return", "fwd_12m_return"]:
        end_col = f"{target}_end_date"
        subset = events.dropna(subset=[target, end_col, "revenue_growth_yoy"]).copy()
        for year in sorted(subset["trade_date"].dt.year.unique()):
            start = pd.Timestamp(f"{int(year)}-01-01")
            test = subset[subset["trade_date"].dt.year == year].copy()
            train = subset[(subset["trade_date"] < start) & (subset[end_col] < start)].copy()
            if len(train) < 100 or test.empty:
                continue
            test["prediction"] = _fit_predict(train, test, target, features)
            test["target_return"] = test[target]
            test["feature_set"] = feature_set
            test["forward_window"] = target
            test["test_year"] = int(year)
            rows.append(test)
    return pd.concat(rows, ignore_index=True)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str]) -> np.ndarray:
    usable = [f for f in features if f in train and train[f].notna().mean() >= 0.5 and train[f].std(skipna=True) > 0]
    low = train[usable].quantile(0.01)
    high = train[usable].quantile(0.99)
    med = train[usable].median()
    x_train = train[usable].clip(lower=low, upper=high, axis=1).fillna(med)
    x_test = test[usable].clip(lower=low, upper=high, axis=1).fillna(med)
    mean = x_train.mean()
    std = x_train.std().replace(0, 1)
    x = ((x_train - mean) / std).fillna(0).to_numpy(float)
    t = ((x_test - mean) / std).fillna(0).to_numpy(float)
    y = train[target].to_numpy(float)
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
    for (feature_set, target), group in pred.groupby(["feature_set", "forward_window"]):
        baseline = ((group["target_return"] - group["target_return"].mean()) ** 2).sum()
        residual = ((group["target_return"] - group["prediction"]) ** 2).sum()
        top = group[group["prediction"] >= group["prediction"].quantile(0.8)]
        bottom = group[group["prediction"] <= group["prediction"].quantile(0.2)]
        rows.append({
            "feature_set": feature_set,
            "forward_window": target,
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
    return pd.DataFrame(rows).sort_values(["forward_window", "spearman_corr"], ascending=[True, False])


def _conditions(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_set, target), group in pred.groupby(["feature_set", "forward_window"]):
        for condition, mask in {
            "model_top_quintile": group["prediction"] >= group["prediction"].quantile(0.8),
            "sales_growth_30pct": group["revenue_growth_yoy"] >= 0.3,
            "sales_growth_30pct_top_half": (group["revenue_growth_yoy"] >= 0.3) & (group["prediction"] >= group["prediction"].median()),
        }.items():
            selected = group[mask]
            if len(selected) < 10:
                continue
            rows.append({
                "feature_set": feature_set,
                "forward_window": target,
                "condition": condition,
                "observations": len(selected),
                "mean_return": selected["target_return"].mean(),
                "median_return": selected["target_return"].median(),
                "hit_rate": (selected["target_return"] > 0).mean(),
            })
    return pd.DataFrame(rows).sort_values(["forward_window", "mean_return"], ascending=[True, False])


def _write_report(perf: pd.DataFrame, cond: pd.DataFrame) -> None:
    lines = [
        "# DoltHub Model Improvement",
        "",
        "Compares feature subsets on the deeper DoltHub annual sample.",
        "",
        "## Performance",
        "",
        _markdown_table(perf),
        "",
        "## Conditions",
        "",
        _markdown_table(cond),
        "",
        "## Interpretation",
        "",
        "- On the longer Dolt sample, price-only and income+price are very close for 12m ranking.",
        "- Adding balance sheet/cash-flow ratios did not improve the model; the full feature set underperformed income+price.",
        "- The longer-history validation supports a simple 12m annual price/fundamentals signal, but not an increasingly complex feature stack.",
    ]
    (OUTPUT_DIR / "DOLTHUB_MODEL_IMPROVEMENT.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "test_years"}:
            if any(token in col for token in ["r2", "corr", "return", "rate"]):
                display[col] = display[col].map(lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())

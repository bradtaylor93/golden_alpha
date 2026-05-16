"""LightGBM comparison for the S&P annual fundamentals ranker.

The experiment uses the same leak controls as the ridge studies:

- annual statements are available only after the configured lag upstream,
- price features are trailing-only,
- each test year trains only on rows whose forward-return window ended before
  the test year starts.

Hyperparameters are fixed; there is no tuning on test rows.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import run_sp500_annual_improvement as baseline


OUTPUT_DIR = Path("ath_reversion_research/reports/lightgbm_fundamental_ranking")
EVENTS = Path("ath_reversion_research/reports/sp500_annual_improvement/enhanced_events.csv")
FEATURES = baseline.FEATURE_SETS["financial_plus_price"]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENTS, parse_dates=["trade_date", "fwd_3m_return_end_date", "fwd_6m_return_end_date", "fwd_12m_return_end_date"])
    predictions = _walk_forward_lgbm(events)
    predictions.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    performance = _performance(predictions)
    conditions = _conditions(predictions)
    performance.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    conditions.to_csv(OUTPUT_DIR / "condition_summary.csv", index=False)
    _write_report(performance, conditions)
    print(performance.to_string(index=False))
    return 0


def _walk_forward_lgbm(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in baseline.annual.FORWARD_WINDOWS:
        end_col = f"{target}_end_date"
        for bucket in ["all", "large_cap", "mid_cap"]:
            subset = events.copy() if bucket == "all" else events[events["market_cap_bucket"] == bucket].copy()
            subset = subset.dropna(subset=[target, end_col, "revenue_growth_yoy"])
            for year in sorted(subset["trade_date"].dt.year.unique()):
                test_start = pd.Timestamp(f"{int(year)}-01-01")
                test = subset[subset["trade_date"].dt.year == year].copy()
                train = subset[(subset["trade_date"] < test_start) & (subset[end_col] < test_start)].copy()
                if len(train) < 25 or test.empty:
                    continue
                test["prediction"] = _fit_predict_lgbm(train, test, target)
                test["target_return"] = test[target]
                test["model"] = "lightgbm_fixed"
                test["forward_window"] = target
                test["regression_bucket"] = bucket
                test["test_year"] = int(year)
                rows.append(test)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fit_predict_lgbm(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    usable = [
        feature
        for feature in FEATURES
        if feature in train and train[feature].notna().mean() >= 0.50 and train[feature].astype(float).std(skipna=True) > 0
    ]
    train_median = train[usable].median()
    train_low = train[usable].quantile(0.01)
    train_high = train[usable].quantile(0.99)
    x_train = train[usable].astype(float).clip(lower=train_low, upper=train_high, axis=1).fillna(train_median)
    x_test = test[usable].astype(float).clip(lower=train_low, upper=train_high, axis=1).fillna(train_median)
    y_train = train[target].astype(float)
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=40,
        learning_rate=0.05,
        num_leaves=7,
        max_depth=3,
        min_child_samples=3,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=5.00,
        random_state=42,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    return model.predict(x_test)


def _performance(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, bucket), group in predictions.groupby(["forward_window", "regression_bucket"]):
        baseline_sse = ((group["target_return"] - group["target_return"].mean()) ** 2).sum()
        residual_sse = ((group["target_return"] - group["prediction"]) ** 2).sum()
        top = group[group["prediction"] >= group["prediction"].quantile(0.80)]
        bottom = group[group["prediction"] <= group["prediction"].quantile(0.20)]
        rows.append(
            {
                "model": "lightgbm_fixed",
                "forward_window": target,
                "regression_bucket": bucket,
                "observations": int(len(group)),
                "oos_r2": float(1.0 - residual_sse / baseline_sse) if baseline_sse > 0 else np.nan,
                "pearson_corr": float(group["prediction"].corr(group["target_return"])),
                "spearman_corr": float(group["prediction"].rank().corr(group["target_return"].rank())),
                "top_quintile_mean_return": float(top["target_return"].mean()),
                "top_quintile_median_return": float(top["target_return"].median()),
                "top_quintile_hit_rate": float((top["target_return"] > 0).mean()),
                "bottom_quintile_mean_return": float(bottom["target_return"].mean()),
                "top_minus_bottom_mean": float(top["target_return"].mean() - bottom["target_return"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["forward_window", "regression_bucket"])


def _conditions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, bucket), group in predictions.groupby(["forward_window", "regression_bucket"]):
        masks = {
            "model_top_quintile": group["prediction"] >= group["prediction"].quantile(0.80),
            "growth30_positive_margin": (group["revenue_growth_yoy"] >= 0.30) & (group["operating_margin"] > 0),
            "growth30_model_top_half": (group["revenue_growth_yoy"] >= 0.30) & (group["prediction"] >= group["prediction"].median()),
        }
        for condition, mask in masks.items():
            selected = group[mask]
            if len(selected) < 5:
                continue
            rows.append(
                {
                    "forward_window": target,
                    "regression_bucket": bucket,
                    "condition": condition,
                    "observations": int(len(selected)),
                    "mean_return": float(selected["target_return"].mean()),
                    "median_return": float(selected["target_return"].median()),
                    "hit_rate": float((selected["target_return"] > 0).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["forward_window", "regression_bucket", "mean_return"], ascending=[True, True, False])


def _write_report(performance: pd.DataFrame, conditions: pd.DataFrame) -> None:
    ridge = pd.read_csv("ath_reversion_research/reports/sp500_annual_improvement/performance_summary.csv")
    ridge = ridge[(ridge["feature_set"] == "financial_plus_price")]
    lines = [
        "# LightGBM Fundamental Ranking Study",
        "",
        "Fixed-parameter LightGBM compared with the current ridge financial+price model.",
        "",
        "## Leakage controls",
        "",
        "- Uses the same purged annual walk-forward setup as the ridge model.",
        "- No LightGBM hyperparameter tuning was performed on test rows.",
        "- Features are annual financials plus causal trailing price context.",
        "",
        "## LightGBM results",
        "",
        _markdown_table(performance),
        "",
        "## Ridge benchmark rows",
        "",
        _markdown_table(ridge),
        "",
        "## LightGBM condition subsets",
        "",
        _markdown_table(conditions.head(40)),
        "",
        "## Interpretation",
        "",
        "- LightGBM is only useful if it beats the ridge financial+price model on rank correlation and top-quintile portfolio characteristics.",
        "- With this short Yahoo/current-S&P sample, nonlinear models are at high risk of overfitting.",
    ]
    (OUTPUT_DIR / "LIGHTGBM_FUNDAMENTAL_RANKING.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations"}:
            if any(token in col for token in ["r2", "corr", "return", "rate", "mean", "median"]):
                display[col] = display[col].map(lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())

"""Temporal stability diagnostics for DoltHub annual fundamental signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_dolthub_aligned_replication as aligned


OUTPUT_DIR = Path("ath_reversion_research/reports/dolthub_temporal_stability")
EVENTS = Path("ath_reversion_research/reports/dolthub_aligned_replication/aligned_events.csv")
FEATURE_SETS = {
    "price_only": aligned.PRICE_FEATURES,
    "financial_only": aligned.FINANCIAL_FEATURES,
    "financial_plus_price": aligned.FINANCIAL_FEATURES + aligned.PRICE_FEATURES,
}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENTS, parse_dates=["trade_date", "fwd_12m_return_end_date"])
    predictions = []
    coefficients = []
    for training_mode in ["expanding", "rolling_3y", "rolling_5y"]:
        for feature_set, features in FEATURE_SETS.items():
            pred, coef = _walk_forward(events, feature_set, features, training_mode)
            predictions.append(pred)
            coefficients.append(coef)
    pred = pd.concat(predictions, ignore_index=True)
    coef = pd.concat(coefficients, ignore_index=True)
    pred.to_csv(OUTPUT_DIR / "temporal_predictions.csv", index=False)
    coef.to_csv(OUTPUT_DIR / "temporal_coefficients.csv", index=False)
    summary = _summary(pred)
    yearly = _yearly_summary(pred)
    coef_stability = _coefficient_stability(coef)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    yearly.to_csv(OUTPUT_DIR / "yearly_summary.csv", index=False)
    coef_stability.to_csv(OUTPUT_DIR / "coefficient_stability.csv", index=False)
    _write_report(summary, yearly, coef_stability)
    print(summary.to_string(index=False))
    print("\nYearly")
    print(yearly.head(40).to_string(index=False))
    return 0


def _walk_forward(events: pd.DataFrame, feature_set: str, features: list[str], training_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coef_rows = []
    subset = events[(events["market_cap_bucket"] == "large_cap")].dropna(subset=["fwd_12m_return", "fwd_12m_return_end_date", "revenue_growth_yoy"]).copy()
    for year in sorted(subset["trade_date"].dt.year.unique()):
        start = pd.Timestamp(f"{int(year)}-01-01")
        test = subset[subset["trade_date"].dt.year == year].copy()
        train = subset[(subset["trade_date"] < start) & (subset["fwd_12m_return_end_date"] < start)].copy()
        if training_mode == "rolling_3y":
            train = train[train["trade_date"] >= start - pd.DateOffset(years=3)]
        elif training_mode == "rolling_5y":
            train = train[train["trade_date"] >= start - pd.DateOffset(years=5)]
        if len(train) < 100 or test.empty:
            continue
        pred, coefs = _fit_predict(train, test, features)
        test["prediction"] = pred
        test["target_return"] = test["fwd_12m_return"]
        test["feature_set"] = feature_set
        test["training_mode"] = training_mode
        test["test_year"] = int(year)
        rows.append(test)
        for feature, value in coefs.items():
            coef_rows.append({"feature_set": feature_set, "training_mode": training_mode, "test_year": int(year), "feature": feature, "coef": value})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coef_rows)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    usable = [f for f in features if f in train and train[f].notna().mean() >= 0.50 and train[f].std(skipna=True) > 0]
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
    return t_mat @ beta, dict(zip(usable, beta[1:]))


def _performance(group: pd.DataFrame) -> dict[str, float | int]:
    top = group[group["prediction"] >= group["prediction"].quantile(0.80)]
    bottom = group[group["prediction"] <= group["prediction"].quantile(0.20)]
    baseline = ((group["target_return"] - group["target_return"].mean()) ** 2).sum()
    residual = ((group["target_return"] - group["prediction"]) ** 2).sum()
    return {
        "observations": int(len(group)),
        "test_years": int(group["test_year"].nunique()),
        "oos_r2": float(1 - residual / baseline) if baseline > 0 else np.nan,
        "pearson_corr": float(group["prediction"].corr(group["target_return"])),
        "spearman_corr": float(group["prediction"].rank().corr(group["target_return"].rank())),
        "top_quintile_mean_return": float(top["target_return"].mean()),
        "top_quintile_median_return": float(top["target_return"].median()),
        "top_quintile_hit_rate": float((top["target_return"] > 0).mean()),
        "bottom_quintile_mean_return": float(bottom["target_return"].mean()),
        "top_minus_bottom_mean": float(top["target_return"].mean() - bottom["target_return"].mean()),
    }


def _summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in pred.groupby(["training_mode", "feature_set"]):
        training_mode, feature_set = keys
        row = _performance(group)
        row.update({"training_mode": training_mode, "feature_set": feature_set})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["spearman_corr"], ascending=False)


def _yearly_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in pred.groupby(["training_mode", "feature_set", "test_year"]):
        training_mode, feature_set, year = keys
        row = _performance(group)
        row.update({"training_mode": training_mode, "feature_set": feature_set, "test_year": int(year)})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["test_year", "training_mode", "feature_set"])


def _coefficient_stability(coef: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in coef.groupby(["training_mode", "feature_set", "feature"]):
        signs = np.sign(group["coef"])
        nonzero = signs[signs != 0]
        rows.append({
            "training_mode": keys[0],
            "feature_set": keys[1],
            "feature": keys[2],
            "mean_coef": float(group["coef"].mean()),
            "median_coef": float(group["coef"].median()),
            "sign_consistency": float((nonzero == np.sign(group["coef"].mean())).mean()) if len(nonzero) else np.nan,
            "folds": int(group["test_year"].nunique()),
        })
    return pd.DataFrame(rows).sort_values(["feature_set", "training_mode", "sign_consistency"], ascending=[True, True, False])


def _write_report(summary: pd.DataFrame, yearly: pd.DataFrame, coef_stability: pd.DataFrame) -> None:
    lines = [
        "# DoltHub Temporal Stability Diagnostics",
        "",
        "Tests whether older training years hurt fundamentals by comparing expanding, 3-year rolling, and 5-year rolling training windows.",
        "",
        "## Overall performance",
        "",
        _markdown_table(summary),
        "",
        "## Year-by-year performance",
        "",
        _markdown_table(yearly),
        "",
        "## Coefficient stability",
        "",
        _markdown_table(coef_stability.head(80)),
        "",
        "## Interpretation",
        "",
        "- If rolling windows beat expanding windows, the signal is non-stationary and older years may hurt.",
        "- If financial-only coefficients have poor sign consistency, fundamentals are unstable as direct predictors.",
        "- If price-only is stable while financial-plus-price is not, fundamentals are better as filters than direct model inputs.",
    ]
    (OUTPUT_DIR / "DOLTHUB_TEMPORAL_STABILITY.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "test_years", "test_year", "folds"}:
            if any(token in col for token in ["r2", "corr", "return", "rate", "coef", "consistency"]):
                display[col] = display[col].map(lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())

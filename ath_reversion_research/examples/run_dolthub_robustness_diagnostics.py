"""Robustness diagnostics for DoltHub large-cap 12m ranking.

Tests shorter rolling windows, robust feature transforms, and feature drift for
the DoltHub aligned large-cap annual model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_dolthub_aligned_replication as aligned


OUTPUT_DIR = Path("ath_reversion_research/reports/dolthub_robustness_diagnostics")
EVENTS = Path("ath_reversion_research/reports/dolthub_aligned_replication/aligned_events.csv")
FEATURE_SETS = {
    "price_only": aligned.PRICE_FEATURES,
    "financial_only": aligned.FINANCIAL_FEATURES,
    "financial_plus_price": aligned.FINANCIAL_FEATURES + aligned.PRICE_FEATURES,
}
TRAINING_MODES = {
    "expanding": None,
    "rolling_1y": 1,
    "rolling_2y": 2,
    "rolling_3y": 3,
    "rolling_5y": 5,
}
TRANSFORMS = ["winsor_z", "rank", "robust_z"]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENTS, parse_dates=["trade_date", "fwd_12m_return_end_date"])
    events = events[(events["market_cap_bucket"] == "large_cap")].dropna(subset=["fwd_12m_return", "fwd_12m_return_end_date", "revenue_growth_yoy"])
    predictions = []
    coefficients = []
    for feature_set, features in FEATURE_SETS.items():
        for transform in TRANSFORMS:
            for mode, years in TRAINING_MODES.items():
                pred, coef = _walk_forward(events, feature_set, features, transform, mode, years)
                if not pred.empty:
                    predictions.append(pred)
                if not coef.empty:
                    coefficients.append(coef)
    pred = pd.concat(predictions, ignore_index=True)
    coef = pd.concat(coefficients, ignore_index=True)
    summary = _summary(pred)
    yearly = _yearly(pred)
    drift = _feature_drift(events)
    coef_stability = _coef_stability(coef)
    pred.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    yearly.to_csv(OUTPUT_DIR / "yearly_summary.csv", index=False)
    drift.to_csv(OUTPUT_DIR / "feature_drift.csv", index=False)
    coef_stability.to_csv(OUTPUT_DIR / "coefficient_stability.csv", index=False)
    _write_report(summary, yearly, drift, coef_stability)
    print(summary.head(30).to_string(index=False))
    return 0


def _walk_forward(
    events: pd.DataFrame,
    feature_set: str,
    features: list[str],
    transform: str,
    mode: str,
    years: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coef_rows = []
    for year in sorted(events["trade_date"].dt.year.unique()):
        start = pd.Timestamp(f"{int(year)}-01-01")
        test = events[events["trade_date"].dt.year == year].copy()
        train = events[(events["trade_date"] < start) & (events["fwd_12m_return_end_date"] < start)].copy()
        if years is not None:
            train = train[train["trade_date"] >= start - pd.DateOffset(years=years)]
        if len(train) < 75 or test.empty:
            continue
        pred, coefs = _fit_predict(train, test, features, transform)
        test["prediction"] = pred
        test["target_return"] = test["fwd_12m_return"]
        test["feature_set"] = feature_set
        test["transform"] = transform
        test["training_mode"] = mode
        test["test_year"] = int(year)
        rows.append(test)
        for feature, coef in coefs.items():
            coef_rows.append({"feature_set": feature_set, "transform": transform, "training_mode": mode, "test_year": int(year), "feature": feature, "coef": coef})
    if not rows:
        return pd.DataFrame(), pd.DataFrame(coef_rows)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coef_rows)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], transform: str) -> tuple[np.ndarray, dict[str, float]]:
    usable = [f for f in features if f in train and train[f].notna().mean() >= 0.5 and train[f].std(skipna=True) > 0]
    x_train = train[usable].astype(float)
    x_test = test[usable].astype(float)
    if transform == "winsor_z":
        lo = x_train.quantile(0.01)
        hi = x_train.quantile(0.99)
        med = x_train.median()
        x_train = x_train.clip(lower=lo, upper=hi, axis=1).fillna(med)
        x_test = x_test.clip(lower=lo, upper=hi, axis=1).fillna(med)
        center = x_train.mean()
        scale = x_train.std().replace(0, 1)
        x_train = (x_train - center) / scale
        x_test = (x_test - center) / scale
    elif transform == "robust_z":
        lo = x_train.quantile(0.05)
        hi = x_train.quantile(0.95)
        med = x_train.median()
        iqr = (x_train.quantile(0.75) - x_train.quantile(0.25)).replace(0, 1)
        x_train = (x_train.clip(lower=lo, upper=hi, axis=1).fillna(med) - med) / iqr
        x_test = (x_test.clip(lower=lo, upper=hi, axis=1).fillna(med) - med) / iqr
    elif transform == "rank":
        train_ranked = pd.DataFrame(index=x_train.index)
        test_ranked = pd.DataFrame(index=x_test.index)
        for col in usable:
            values = x_train[col].dropna().sort_values().to_numpy()
            train_ranked[col] = x_train[col].rank(pct=True).fillna(0.5)
            test_ranked[col] = x_test[col].apply(lambda value: np.searchsorted(values, value, side="right") / len(values) if pd.notna(value) and len(values) else 0.5)
            test_ranked[col] = test_ranked[col].fillna(0.5)
        x_train = train_ranked
        x_test = test_ranked
    else:
        raise ValueError(transform)
    x = x_train.fillna(0).to_numpy(float)
    t = x_test.fillna(0).to_numpy(float)
    y = train["fwd_12m_return"].to_numpy(float)
    y_mean = y.mean()
    x_mat = np.c_[np.ones(len(x)), x]
    t_mat = np.c_[np.ones(len(t)), t]
    penalty = np.eye(x_mat.shape[1])
    penalty[0, 0] = 0
    beta = np.linalg.solve(x_mat.T @ x_mat + 10.0 * penalty, x_mat.T @ (y - y_mean))
    beta[0] += y_mean
    return t_mat @ beta, dict(zip(usable, beta[1:]))


def _perf(group: pd.DataFrame) -> dict[str, float | int]:
    top = group[group["prediction"] >= group["prediction"].quantile(0.8)]
    bottom = group[group["prediction"] <= group["prediction"].quantile(0.2)]
    baseline = ((group["target_return"] - group["target_return"].mean()) ** 2).sum()
    residual = ((group["target_return"] - group["prediction"]) ** 2).sum()
    return {
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
    }


def _summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in pred.groupby(["feature_set", "transform", "training_mode"]):
        row = _perf(group)
        row.update({"feature_set": keys[0], "transform": keys[1], "training_mode": keys[2]})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["spearman_corr", "top_quintile_mean_return"], ascending=False)


def _yearly(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in pred.groupby(["feature_set", "transform", "training_mode", "test_year"]):
        row = _perf(group)
        row.update({"feature_set": keys[0], "transform": keys[1], "training_mode": keys[2], "test_year": int(keys[3])})
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_drift(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in sorted(set(sum(FEATURE_SETS.values(), []))):
        if feature not in events:
            continue
        by_year = events.groupby(events["trade_date"].dt.year)[feature].agg(["mean", "std", "median"]).reset_index()
        if len(by_year) < 2:
            continue
        rows.append({
            "feature": feature,
            "year_count": len(by_year),
            "mean_min": by_year["mean"].min(),
            "mean_max": by_year["mean"].max(),
            "mean_range": by_year["mean"].max() - by_year["mean"].min(),
            "median_min": by_year["median"].min(),
            "median_max": by_year["median"].max(),
            "std_median": by_year["std"].median(),
        })
    return pd.DataFrame(rows).sort_values("mean_range", ascending=False)


def _coef_stability(coef: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in coef.groupby(["feature_set", "transform", "training_mode", "feature"]):
        signs = np.sign(group["coef"])
        nz = signs[signs != 0]
        rows.append({
            "feature_set": keys[0],
            "transform": keys[1],
            "training_mode": keys[2],
            "feature": keys[3],
            "mean_coef": group["coef"].mean(),
            "median_coef": group["coef"].median(),
            "sign_consistency": (nz == np.sign(group["coef"].mean())).mean() if len(nz) else np.nan,
            "folds": group["test_year"].nunique(),
        })
    return pd.DataFrame(rows).sort_values(["feature_set", "transform", "training_mode", "sign_consistency"], ascending=[True, True, True, False])


def _write_report(summary: pd.DataFrame, yearly: pd.DataFrame, drift: pd.DataFrame, coef: pd.DataFrame) -> None:
    lines = [
        "# DoltHub Robustness Diagnostics",
        "",
        "Tests shorter rolling windows, robust transformations, feature drift, and coefficient stability for the Dolt large-cap 12m model.",
        "",
        "## Overall model variants",
        "",
        _markdown_table(summary),
        "",
        "## Yearly performance",
        "",
        _markdown_table(yearly),
        "",
        "## Feature drift",
        "",
        _markdown_table(drift.head(60)),
        "",
        "## Coefficient stability",
        "",
        _markdown_table(coef.head(100)),
        "",
        "## Interpretation",
        "",
        "- Shorter rolling windows do not fix the fundamentals issue; expanding and 5-year rolling are generally better than 1-3 year windows.",
        "- Rank and robust transformations should only be adopted if they improve OOS rank metrics versus winsorized z-scores.",
        "- Large feature drift and low coefficient sign consistency indicate non-stationarity in direct fundamental predictors.",
    ]
    (OUTPUT_DIR / "DOLTHUB_ROBUSTNESS_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "test_years", "test_year", "folds", "year_count"}:
            if any(token in col for token in ["r2", "corr", "return", "rate", "coef", "consistency", "mean", "median", "std", "range"]):
                display[col] = display[col].map(lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())

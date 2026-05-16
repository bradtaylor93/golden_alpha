"""Reconcile Yahoo and Dolt annual fundamental features.

This compares underlying fundamentals and engineered ratios for rows where both
sources have the same symbol and fiscal period end.  It is intended to explain
why Yahoo and Dolt model results differ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("ath_reversion_research/reports/yahoo_dolt_feature_reconciliation")
YAHOO_EVENTS = Path("ath_reversion_research/reports/sp500_annual_fundamental_regression/financial_ratio_events.csv")
DOLT_EVENTS = Path("ath_reversion_research/reports/dolthub_aligned_replication/aligned_events.csv")

FEATURE_MAP = {
    # raw statement fields
    "revenue": ("total_revenue", "sales"),
    "gross_profit": ("gross_profit", "gross_profit"),
    "operating_income": ("operating_income", "operating_income"),
    "net_income": ("net_income", "net_income"),
    "ebitda": ("ebitda", "ebitda"),
    "operating_cash_flow": ("operating_cash_flow", "net_cash_from_operating_activities"),
    "total_assets": ("total_assets", "total_assets"),
    "cash": ("cash", "cash_and_equivalents"),
    "current_assets": ("current_assets", "total_current_assets"),
    "current_liabilities": ("current_liabilities", "total_current_liabilities"),
    # Dolt only has long_term_debt in aligned file, not full total debt.
    "debt": ("total_debt", "long_term_debt"),
    "equity": ("stockholders_equity", "total_equity"),
    # engineered features
    "revenue_growth_yoy": ("revenue_growth_yoy", "revenue_growth_yoy"),
    "gross_margin": ("gross_margin", "gross_margin"),
    "operating_margin": ("operating_margin", "operating_margin"),
    "net_margin": ("net_margin", "net_margin"),
    "ebitda_margin": ("ebitda_margin", "ebitda_margin"),
    "cfo_margin": ("cfo_margin", "cfo_margin"),
    "fcf_margin": ("fcf_margin", "fcf_margin"),
    "capex_to_revenue": ("capex_to_revenue", "capex_to_revenue"),
    "debt_to_assets": ("debt_to_assets", "debt_to_assets"),
    "debt_to_equity": ("debt_to_equity", "debt_to_equity"),
    "cash_to_assets": ("cash_to_assets", "cash_to_assets"),
    "current_ratio": ("current_ratio", "current_ratio"),
    "asset_turnover": ("asset_turnover", "asset_turnover"),
    # price/return alignment checks
    "trade_close": ("trade_close", "trade_close"),
    "fwd_12m_return": ("fwd_12m_return", "fwd_12m_return"),
}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yahoo = pd.read_csv(YAHOO_EVENTS, parse_dates=["fiscal_period_end", "trade_date"])
    dolt = pd.read_csv(DOLT_EVENTS, parse_dates=["fiscal_period_end", "trade_date"])
    matched = yahoo.merge(
        dolt,
        on=["symbol", "fiscal_period_end"],
        suffixes=("_yahoo", "_dolt"),
        how="inner",
    )
    matched.to_csv(OUTPUT_DIR / "matched_rows.csv", index=False)

    coverage = _coverage_summary(yahoo, dolt, matched)
    comparison = _feature_comparison(matched)
    discrepancies = _largest_discrepancies(matched)
    coverage.to_csv(OUTPUT_DIR / "coverage_summary.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "feature_comparison.csv", index=False)
    discrepancies.to_csv(OUTPUT_DIR / "largest_discrepancies.csv", index=False)
    _write_report(coverage, comparison, discrepancies)
    print("Coverage")
    print(coverage.to_string(index=False))
    print("\nFeature comparison")
    print(comparison.to_string(index=False))
    return 0


def _coverage_summary(yahoo: pd.DataFrame, dolt: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "yahoo_rows", "value": len(yahoo)},
            {"metric": "dolt_rows", "value": len(dolt)},
            {"metric": "matched_rows", "value": len(matched)},
            {"metric": "yahoo_symbols", "value": yahoo["symbol"].nunique()},
            {"metric": "dolt_symbols", "value": dolt["symbol"].nunique()},
            {"metric": "matched_symbols", "value": matched["symbol"].nunique()},
            {"metric": "matched_min_fiscal_year", "value": int(matched["fiscal_period_end"].dt.year.min()) if not matched.empty else np.nan},
            {"metric": "matched_max_fiscal_year", "value": int(matched["fiscal_period_end"].dt.year.max()) if not matched.empty else np.nan},
        ]
    )


def _feature_comparison(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, (y_col, d_col) in FEATURE_MAP.items():
        y_name = f"{y_col}_yahoo" if f"{y_col}_yahoo" in matched.columns else y_col
        d_name = f"{d_col}_dolt" if f"{d_col}_dolt" in matched.columns else d_col
        if y_name not in matched.columns or d_name not in matched.columns:
            rows.append({"feature": label, "status": "missing_column", "yahoo_column": y_name, "dolt_column": d_name})
            continue
        pair = matched[[y_name, d_name]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if pair.empty:
            rows.append({"feature": label, "status": "no_overlap", "yahoo_column": y_name, "dolt_column": d_name})
            continue
        y = pair[y_name]
        d = pair[d_name]
        diff = d - y
        denom = y.abs().replace(0.0, np.nan)
        pct_diff = (diff.abs() / denom).replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "feature": label,
                "status": "ok",
                "observations": int(len(pair)),
                "pearson_corr": float(y.corr(d)),
                "spearman_corr": float(y.rank().corr(d.rank())),
                "median_abs_pct_diff": float(pct_diff.median()),
                "p90_abs_pct_diff": float(pct_diff.quantile(0.90)),
                "median_signed_pct_diff": float((diff / denom).median()),
                "same_sign_rate": float((np.sign(y) == np.sign(d)).mean()),
                "yahoo_median": float(y.median()),
                "dolt_median": float(d.median()),
                "yahoo_column": y_name,
                "dolt_column": d_name,
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "pearson_corr"], ascending=[True, False])


def _largest_discrepancies(matched: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for label, (y_col, d_col) in FEATURE_MAP.items():
        y_name = f"{y_col}_yahoo" if f"{y_col}_yahoo" in matched.columns else y_col
        d_name = f"{d_col}_dolt" if f"{d_col}_dolt" in matched.columns else d_col
        if y_name not in matched.columns or d_name not in matched.columns:
            continue
        frame = matched[["symbol", "fiscal_period_end", y_name, d_name]].copy()
        frame[y_name] = pd.to_numeric(frame[y_name], errors="coerce")
        frame[d_name] = pd.to_numeric(frame[d_name], errors="coerce")
        frame = frame.dropna()
        if frame.empty:
            continue
        frame["feature"] = label
        frame["abs_pct_diff"] = ((frame[d_name] - frame[y_name]).abs() / frame[y_name].abs().replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(subset=["abs_pct_diff"]).sort_values("abs_pct_diff", ascending=False).head(top_n)
        frame = frame.rename(columns={y_name: "yahoo_value", d_name: "dolt_value"})
        rows.append(frame[["feature", "symbol", "fiscal_period_end", "yahoo_value", "dolt_value", "abs_pct_diff"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _write_report(coverage: pd.DataFrame, comparison: pd.DataFrame, discrepancies: pd.DataFrame) -> None:
    high_level = comparison[comparison["status"] == "ok"].copy()
    problem = high_level[(high_level["pearson_corr"].fillna(0) < 0.95) | (high_level["median_abs_pct_diff"].fillna(0) > 0.05)]
    lines = [
        "# Yahoo vs Dolt Feature Reconciliation",
        "",
        "Compares matched rows by `symbol + fiscal_period_end` to determine whether Yahoo and Dolt fundamentals are equivalent.",
        "",
        "## Coverage",
        "",
        _markdown_table(coverage),
        "",
        "## Feature comparison",
        "",
        _markdown_table(comparison),
        "",
        "## Fields requiring attention",
        "",
        _markdown_table(problem),
        "",
        "## Largest discrepancies",
        "",
        _markdown_table(discrepancies.head(80)),
        "",
        "## Interpretation",
        "",
        "- Revenue/sales and many margin features should be highly correlated if definitions match.",
        "- Debt, capex, FCF, and cash-flow features are expected to differ more because Dolt and Yahoo expose different line-item definitions/sign conventions.",
        "- If price/return fields differ, model comparisons are not apples-to-apples even with matched fundamentals.",
        "- Features with low correlation or large median absolute percentage difference should not be mixed without remapping definitions.",
    ]
    (OUTPUT_DIR / "YAHOO_DOLT_FEATURE_RECONCILIATION.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "value"}:
            if any(token in col for token in ["corr", "diff", "rate"]):
                display[col] = display[col].map(lambda v: f"{float(v) * 100:.2f}%" if pd.notna(v) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())

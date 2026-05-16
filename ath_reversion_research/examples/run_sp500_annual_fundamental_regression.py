"""Annual fundamentals regression on a broader current S&P 500 universe.

This increases cross-sectional sample size while keeping the same annual
no-lookahead methodology:

- 90-day post-fiscal-year-end availability lag,
- forward returns from first trading day after availability,
- purged walk-forward training by test year.

The universe is current S&P 500 constituents, so survivorship bias remains.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

import run_financial_ratio_regression as annual


OUTPUT_DIR = Path("ath_reversion_research/reports/sp500_annual_fundamental_regression")
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = _download_sp500_symbols()
    pd.DataFrame({"symbol": symbols}).to_csv(OUTPUT_DIR / "sp500_current_symbols.csv", index=False)

    config = annual.RegressionConfig()
    annual.OUTPUT_DIR = OUTPUT_DIR
    fundamentals, market_caps = annual._download_fundamentals(symbols)
    fundamentals.to_csv(OUTPUT_DIR / "annual_fundamentals_raw.csv", index=False)
    market_caps.to_csv(OUTPUT_DIR / "current_market_caps.csv", index=False)

    features = annual._build_feature_events(fundamentals, market_caps, config)
    prices = annual.download_yahoo_ohlcv(symbols, start=config.start, chunk_size=25)
    events = annual._attach_forward_returns(features, prices, config.reporting_lag_days)
    events.to_csv(OUTPUT_DIR / "financial_ratio_events.csv", index=False)

    predictions = annual._walk_forward_predictions(events, config)
    predictions.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    performance = annual._performance_summary(predictions)
    condition_summary = annual._condition_summary(predictions)
    coefficient_summary = annual._coefficient_summary(predictions)
    performance.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    condition_summary.to_csv(OUTPUT_DIR / "condition_summary.csv", index=False)
    coefficient_summary.to_csv(OUTPUT_DIR / "coefficient_summary.csv", index=False)
    _write_report(symbols, events, predictions, performance, condition_summary, coefficient_summary, config)

    print("S&P 500 annual fundamentals")
    print("symbols", len(symbols), "events", len(events), "predictions", len(predictions))
    print(performance.to_string(index=False))
    return 0


def _download_sp500_symbols() -> list[str]:
    response = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text))[0]
    symbols = table["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper().tolist()
    return sorted(set(symbols))


def _write_report(
    symbols: list[str],
    events: pd.DataFrame,
    predictions: pd.DataFrame,
    performance: pd.DataFrame,
    condition_summary: pd.DataFrame,
    coefficient_summary: pd.DataFrame,
    config: annual.RegressionConfig,
) -> None:
    prior_perf = Path("ath_reversion_research/reports/financial_ratio_regression/performance_summary.csv")
    prior_events = Path("ath_reversion_research/reports/financial_ratio_regression/financial_ratio_events.csv")
    prior_rows = len(pd.read_csv(prior_events)) if prior_events.exists() else 0
    prior_pred_rows = len(pd.read_csv("ath_reversion_research/reports/financial_ratio_regression/walk_forward_predictions.csv")) if Path("ath_reversion_research/reports/financial_ratio_regression/walk_forward_predictions.csv").exists() else 0
    lines = [
        "# S&P 500 Annual Fundamentals Regression",
        "",
        "This reruns the annual financial-ratio model on current S&P 500 constituents to increase cross-sectional sample size.",
        "",
        "## Leakage controls",
        "",
        f"- Annual financials are assumed available only {config.reporting_lag_days} calendar days after fiscal year end.",
        "- Forward returns start from first trading day on or after that availability date.",
        "- Training rows are purged unless their forward-return end date is before the test year starts.",
        "- Current S&P 500 membership is not point-in-time, so survivorship bias remains.",
        "",
        "## Sample size",
        "",
        f"- Current S&P 500 symbols requested: {len(symbols)}.",
        f"- Events: {len(events)} versus {prior_rows} in the prior expanded-stock annual study.",
        f"- Prediction rows: {len(predictions)} versus {prior_pred_rows} prior.",
        f"- Symbols with predictions: {predictions['symbol'].nunique() if not predictions.empty else 0}.",
        "",
        "## Predictive power",
        "",
        annual._markdown_table(performance),
        "",
        "## Higher-value subsets",
        "",
        annual._markdown_table(condition_summary.head(50)),
        "",
        "## Average standardized coefficients",
        "",
        annual._markdown_table(coefficient_summary.head(50)),
        "",
        "## Interpretation",
        "",
        "- Using current S&P 500 constituents increases annual sample size materially, but adds survivorship bias.",
        "- Compare the large-cap rows to the prior annual study; they are the most relevant apples-to-apples results.",
        "- If predictive power improves mainly from broader current membership, the next step should be point-in-time constituents rather than more current-list expansion.",
    ]
    (OUTPUT_DIR / "SP500_ANNUAL_FUNDAMENTAL_REGRESSION.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

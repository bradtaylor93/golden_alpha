"""Revenue growth versus future stock returns event study.

The study is intentionally conservative about leakage. Yahoo Finance annual
income statements expose fiscal period end dates, not exact filing timestamps,
so each annual revenue observation is assumed unavailable until 90 calendar days
after fiscal year end. Forward returns are measured only from the first trading
day on or after that assumed availability date.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from ath_reversion_research import download_yahoo_ohlcv, symbols_for


OUTPUT_DIR = Path("ath_reversion_research/reports/revenue_growth_event_study")
EXPANDED_SYMBOLS = Path("ath_reversion_research/reports/expanded_stock_signals/downloaded_symbols.csv")
REPORTING_LAG_DAYS = 90
FORWARD_WINDOWS = {
    "fwd_3m_return": 63,
    "fwd_6m_return": 126,
    "fwd_12m_return": 252,
}


@dataclass(frozen=True)
class StudyConfig:
    start: str = "2018-01-01"
    reporting_lag_days: int = REPORTING_LAG_DAYS
    min_revenue: float = 100_000_000.0


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = StudyConfig()
    symbols = _load_symbols()
    revenue = _download_annual_revenue(symbols)
    revenue.to_csv(OUTPUT_DIR / "annual_revenue_raw.csv", index=False)

    growth = _build_growth_events(revenue, config)
    prices = download_yahoo_ohlcv(symbols, start=config.start, chunk_size=25)
    events = _attach_forward_returns(growth, prices, config.reporting_lag_days)
    events.to_csv(OUTPUT_DIR / "revenue_growth_events.csv", index=False)

    correlations = _correlations(events)
    segment_summary = _segment_summary(events)
    high_growth = _high_growth_summary(events, threshold=0.30)
    deciles = _decile_summary(events)

    correlations.to_csv(OUTPUT_DIR / "correlations.csv", index=False)
    segment_summary.to_csv(OUTPUT_DIR / "segment_summary.csv", index=False)
    high_growth.to_csv(OUTPUT_DIR / "high_growth_30pct_summary.csv", index=False)
    deciles.to_csv(OUTPUT_DIR / "decile_summary.csv", index=False)
    _write_report(events, correlations, segment_summary, high_growth, deciles, config)

    print("Correlations")
    print(correlations.to_string(index=False))
    print("\n30%+ revenue growth summary")
    print(high_growth.to_string(index=False))
    return 0


def _load_symbols() -> list[str]:
    symbols: list[str] = []
    if EXPANDED_SYMBOLS.exists():
        frame = pd.read_csv(EXPANDED_SYMBOLS)
        if "downloaded_symbols" in frame.columns:
            symbols.extend(frame["downloaded_symbols"].dropna().astype(str).tolist())
    symbols.extend(symbols_for(["broad_stock_sample"]))
    cleaned: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        upper = symbol.upper()
        if upper not in seen:
            cleaned.append(upper)
            seen.add(upper)
    return cleaned


def _download_annual_revenue(symbols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        try:
            statement = yf.Ticker(symbol).income_stmt
        except Exception as exc:  # pragma: no cover - network/vendor defensive
            rows.append({"symbol": symbol, "error": str(exc)})
            continue
        if statement is None or statement.empty or "Total Revenue" not in statement.index:
            rows.append({"symbol": symbol, "error": "missing_total_revenue"})
            continue
        revenue = statement.loc["Total Revenue"].dropna()
        for fiscal_period_end, value in revenue.items():
            rows.append(
                {
                    "symbol": symbol,
                    "fiscal_period_end": pd.Timestamp(fiscal_period_end).normalize(),
                    "total_revenue": float(value),
                    "error": "",
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No revenue data downloaded")
    return frame.sort_values(["symbol", "fiscal_period_end"]).reset_index(drop=True)


def _build_growth_events(revenue: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    clean = revenue[(revenue["error"].fillna("") == "") & revenue["total_revenue"].notna()].copy()
    clean["fiscal_period_end"] = pd.to_datetime(clean["fiscal_period_end"])
    clean = clean.sort_values(["symbol", "fiscal_period_end"])
    clean["prior_revenue"] = clean.groupby("symbol")["total_revenue"].shift(1)
    clean["revenue_growth_yoy"] = clean["total_revenue"] / clean["prior_revenue"] - 1.0
    clean["availability_date"] = clean["fiscal_period_end"] + pd.Timedelta(days=config.reporting_lag_days)
    clean = clean[
        (clean["prior_revenue"] >= config.min_revenue)
        & np.isfinite(clean["revenue_growth_yoy"])
        & (clean["availability_date"] >= pd.Timestamp(config.start))
    ].copy()
    return clean.reset_index(drop=True)


def _attach_forward_returns(events: pd.DataFrame, prices: pd.DataFrame, reporting_lag_days: int) -> pd.DataFrame:
    price_map = {
        symbol: group.sort_values("date").reset_index(drop=True)
        for symbol, group in prices.groupby("symbol")
    }
    rows: list[dict[str, object]] = []
    for event in events.to_dict("records"):
        symbol = str(event["symbol"])
        symbol_prices = price_map.get(symbol)
        if symbol_prices is None or symbol_prices.empty:
            continue
        dates = pd.to_datetime(symbol_prices["date"])
        start_pos = int(dates.searchsorted(pd.Timestamp(event["availability_date"]), side="left"))
        if start_pos >= len(symbol_prices):
            continue
        start_close = float(symbol_prices.loc[start_pos, "close"])
        row = dict(event)
        row["trade_date"] = symbol_prices.loc[start_pos, "date"]
        row["trade_close"] = start_close
        row["reporting_lag_days"] = reporting_lag_days
        for name, bars_ahead in FORWARD_WINDOWS.items():
            end_pos = start_pos + bars_ahead
            if end_pos < len(symbol_prices):
                row[name] = float(symbol_prices.loc[end_pos, "close"] / start_close - 1.0)
                row[f"{name}_end_date"] = symbol_prices.loc[end_pos, "date"]
            else:
                row[name] = np.nan
                row[f"{name}_end_date"] = pd.NaT
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _correlations(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for return_col in FORWARD_WINDOWS:
        pair = events[["revenue_growth_yoy", return_col]].dropna()
        growth_rank = pair["revenue_growth_yoy"].rank()
        return_rank = pair[return_col].rank()
        rows.append(
            {
                "forward_window": return_col,
                "observations": int(len(pair)),
                "pearson_corr": float(pair["revenue_growth_yoy"].corr(pair[return_col], method="pearson")),
                "spearman_corr": float(growth_rank.corr(return_rank, method="pearson")),
            }
        )
    return pd.DataFrame(rows)


def _growth_bucket(growth: float) -> str:
    if growth < 0.0:
        return "<0%"
    if growth < 0.10:
        return "0-10%"
    if growth < 0.20:
        return "10-20%"
    if growth < 0.30:
        return "20-30%"
    if growth < 0.50:
        return "30-50%"
    return "50%+"


def _segment_summary(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["growth_segment"] = frame["revenue_growth_yoy"].map(_growth_bucket)
    order = ["<0%", "0-10%", "10-20%", "20-30%", "30-50%", "50%+"]
    rows = []
    for segment in order:
        group = frame[frame["growth_segment"] == segment]
        if group.empty:
            continue
        row: dict[str, object] = {
            "growth_segment": segment,
            "observations": int(len(group)),
            "avg_revenue_growth": float(group["revenue_growth_yoy"].mean()),
        }
        for col in FORWARD_WINDOWS:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_median"] = float(group[col].median())
            row[f"{col}_hit_rate"] = float((group[col] > 0.0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _high_growth_summary(events: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for label, mask in {
        ">=30% revenue growth": events["revenue_growth_yoy"] >= threshold,
        "<30% revenue growth": events["revenue_growth_yoy"] < threshold,
    }.items():
        group = events[mask]
        row: dict[str, object] = {
            "segment": label,
            "observations": int(len(group)),
            "avg_revenue_growth": float(group["revenue_growth_yoy"].mean()),
            "median_revenue_growth": float(group["revenue_growth_yoy"].median()),
        }
        for col in FORWARD_WINDOWS:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_median"] = float(group[col].median())
            row[f"{col}_hit_rate"] = float((group[col] > 0.0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _decile_summary(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["growth_decile"] = pd.qcut(frame["revenue_growth_yoy"], 10, labels=False, duplicates="drop") + 1
    rows = []
    for decile, group in frame.groupby("growth_decile"):
        row: dict[str, object] = {
            "growth_decile": int(decile),
            "observations": int(len(group)),
            "min_growth": float(group["revenue_growth_yoy"].min()),
            "max_growth": float(group["revenue_growth_yoy"].max()),
            "avg_growth": float(group["revenue_growth_yoy"].mean()),
        }
        for col in FORWARD_WINDOWS:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_median"] = float(group[col].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("growth_decile")


def _write_report(
    events: pd.DataFrame,
    correlations: pd.DataFrame,
    segment_summary: pd.DataFrame,
    high_growth: pd.DataFrame,
    deciles: pd.DataFrame,
    config: StudyConfig,
) -> None:
    lines = [
        "# Revenue Growth Event Study",
        "",
        "Question: does YoY annual revenue growth predict 3m, 6m, or 12m stock returns, and is there an opportunity in 30%+ growers?",
        "",
        "## Leakage controls",
        "",
        f"- Annual revenue is assumed tradable only {config.reporting_lag_days} calendar days after fiscal year end.",
        "- Forward returns are measured from the first trading day on or after that availability date.",
        "- The study does not use the fiscal period-end date as the trade date.",
        "- Exact filing timestamps are not available from Yahoo Finance here; this is a conservative lag approximation, not point-in-time fundamentals.",
        "",
        "## Dataset",
        "",
        f"- Events with at least one forward-return window: {len(events)}.",
        f"- Symbols with usable events: {events['symbol'].nunique() if not events.empty else 0}.",
        f"- Trade-date range: {events['trade_date'].min()} to {events['trade_date'].max()}.",
        "",
        "## Correlation between revenue growth and future returns",
        "",
        _markdown_table(correlations),
        "",
        "## Revenue growth segment performance",
        "",
        _markdown_table(segment_summary),
        "",
        "## 30%+ revenue growth filter",
        "",
        _markdown_table(high_growth),
        "",
        "## Growth deciles",
        "",
        _markdown_table(deciles),
        "",
        "## Interpretation",
        "",
        "- On this Yahoo-based sample, the correlation between annual revenue growth and later returns is weak and not stable enough to trade alone.",
        "- The 30%+ growth filter identifies higher-growth companies, but the return advantage is not uniformly stronger across 3m/6m/12m horizons.",
        "- The opportunity, if any, is likely in combining revenue growth with quality, valuation, margin expansion, or price momentum rather than using revenue growth as a standalone signal.",
        "- Results are limited by annual-statement history depth, current-universe survivorship bias, and approximate reporting availability dates.",
    ]
    (OUTPUT_DIR / "REVENUE_GROWTH_EVENT_STUDY.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if column in {"growth_decile", "observations"}:
            continue
        if any(token in column for token in ["growth", "return", "corr", "rate", "mean", "median", "min_", "max_"]):
            if pd.api.types.is_numeric_dtype(display[column]):
                display[column] = display[column].map(lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())

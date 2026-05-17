"""RSI(14) band analysis from 2013 onward.

This is a diagnostic rather than a portfolio backtest. For every stock/day, it
computes close-to-close RSI(14), then measures the next-close-to-14-trading-day
forward return. The one-day execution lag avoids using the same close that
generated the RSI as the entry price.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_historical_market_cap_valuation as hist


PRICE_FILE = Path("ath_reversion_research/reports/dolthub_full_fundamental_validation/price_subset_yahoo.csv")
OUTPUT_DIR = Path("ath_reversion_research/reports/rsi14_band_analysis")
START_DATE = "2013-01-01"
RSI_WINDOW = 14
FORWARD_DAYS = 14
BANDS = list(range(0, 101, 10))
FINE_BANDS = list(range(0, 101, 5))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observations = _build_observations()
    band_summary = _summarize_bands(observations, BANDS, "10pt")
    fine_summary = _summarize_bands(observations, FINE_BANDS, "5pt")
    yearly_summary = _yearly_best_bands(observations)

    band_summary.to_csv(OUTPUT_DIR / "rsi14_band_summary_10pt.csv", index=False)
    fine_summary.to_csv(OUTPUT_DIR / "rsi14_band_summary_5pt.csv", index=False)
    yearly_summary.to_csv(OUTPUT_DIR / "rsi14_yearly_best_bands.csv", index=False)
    _write_report(band_summary, fine_summary, yearly_summary, observations)
    print(band_summary.to_string(index=False))
    return 0


def _build_observations() -> pd.DataFrame:
    prices = pd.read_csv(PRICE_FILE, usecols=["date", "symbol", "close"], parse_dates=["date"])
    prices = prices[prices["date"] >= "2012-01-01"].dropna(subset=["close"])
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    close = close.ffill(limit=5)

    rsi = _rsi(close, RSI_WINDOW)
    entry = close.shift(-1)
    exit_ = close.shift(-(FORWARD_DAYS + 1))
    fwd_return = exit_ / entry - 1.0

    stacked = pd.DataFrame(
        {
            "rsi14": rsi.stack(),
            "forward_14d_return": fwd_return.stack(),
        }
    ).reset_index()
    stacked.columns = ["date", "symbol", "rsi14", "forward_14d_return"]
    stacked = stacked[stacked["date"] >= START_DATE].replace([np.inf, -np.inf], np.nan)
    return stacked.dropna(subset=["rsi14", "forward_14d_return"])


def _rsi(close: pd.DataFrame, window: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain != 0.0, 0.0)
    return out.clip(0.0, 100.0)


def _summarize_bands(observations: pd.DataFrame, bands: list[int], label: str) -> pd.DataFrame:
    data = observations.copy()
    data["rsi_band"] = pd.cut(data["rsi14"], bins=bands, right=False, include_lowest=True)
    rows = []
    total = len(data)
    for band, group in data.groupby("rsi_band", observed=True):
        ret = group["forward_14d_return"].dropna()
        if ret.empty:
            continue
        rows.append(
            {
                "band_type": label,
                "rsi_band": _format_band(band),
                "observations": len(ret),
                "portion": len(ret) / total,
                "mean_14d_return": ret.mean(),
                "median_14d_return": ret.median(),
                "hit_rate": (ret > 0).mean(),
                "downside_std": ret[ret < 0].std(ddof=1),
                "sortino_14d": _sortino(ret),
                "annualized_like_sortino": _sortino(ret) * np.sqrt(252 / FORWARD_DAYS),
                "p05_return": ret.quantile(0.05),
                "p95_return": ret.quantile(0.95),
            }
        )
    return pd.DataFrame(rows).sort_values("sortino_14d", ascending=False)


def _sortino(returns: pd.Series) -> float:
    downside = returns[returns < 0.0]
    downside_std = downside.std(ddof=1)
    if pd.isna(downside_std) or downside_std <= 0:
        return np.nan
    return float(returns.mean() / downside_std)


def _format_band(interval: pd.Interval) -> str:
    left = int(interval.left)
    right = int(interval.right)
    return f"{left}-{right}"


def _yearly_best_bands(observations: pd.DataFrame) -> pd.DataFrame:
    data = observations.copy()
    data["year"] = data["date"].dt.year
    data["rsi_band"] = pd.cut(data["rsi14"], bins=BANDS, right=False, include_lowest=True)
    rows = []
    for year, year_group in data.groupby("year"):
        summary = _summarize_bands(year_group, BANDS, "10pt")
        if summary.empty:
            continue
        best = summary.iloc[0].to_dict()
        best["year"] = int(year)
        rows.append(best)
    return pd.DataFrame(rows)[
        ["year", "rsi_band", "observations", "portion", "mean_14d_return", "hit_rate", "sortino_14d"]
    ]


def _write_report(
    band_summary: pd.DataFrame,
    fine_summary: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    observations: pd.DataFrame,
) -> None:
    with (OUTPUT_DIR / "RSI14_BAND_ANALYSIS.md").open("w", encoding="utf-8") as f:
        f.write("# RSI(14) band analysis\n\n")
        f.write(
            "For each stock/day from 2013 onward, this computes RSI(14) using data through the close, "
            "then measures next-close to 14-trading-day-forward return. This is a signal diagnostic, "
            "not a turnover/cost-adjusted portfolio backtest.\n\n"
        )
        f.write("## Dataset\n\n")
        f.write(f"- Observations: {len(observations):,}\n")
        f.write(f"- Symbols: {observations['symbol'].nunique():,}\n")
        f.write(f"- Date range: {observations['date'].min().date()} to {observations['date'].max().date()}\n")
        f.write(f"- Forward return: next close to +{FORWARD_DAYS} trading days\n\n")
        f.write("## 10-point RSI bands sorted by Sortino\n\n")
        f.write(hist._markdown_table(band_summary))
        f.write("\n\n## 5-point RSI bands sorted by Sortino\n\n")
        f.write(hist._markdown_table(fine_summary.head(20)))
        f.write("\n\n## Best 10-point band by year\n\n")
        f.write(hist._markdown_table(yearly_summary))
        f.write("\n\n## Interpretation\n\n")
        f.write(_interpretation(band_summary, fine_summary))
        f.write("\n")


def _interpretation(band_summary: pd.DataFrame, fine_summary: pd.DataFrame) -> str:
    best_10 = band_summary.iloc[0]
    best_5 = fine_summary.iloc[0]
    return "\n".join(
        [
            f"- Best 10-point band by 14-day Sortino: RSI {best_10['rsi_band']} "
            f"with Sortino {best_10['sortino_14d']:.3f} and mean 14d return {best_10['mean_14d_return']:.2%}.",
            f"- Best 5-point band by 14-day Sortino: RSI {best_5['rsi_band']} "
            f"with Sortino {best_5['sortino_14d']:.3f} and mean 14d return {best_5['mean_14d_return']:.2%}.",
            "- This table is descriptive and uses overlapping forward windows, so Sortino should be used for ranking bands rather than as a standalone tradable strategy estimate.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

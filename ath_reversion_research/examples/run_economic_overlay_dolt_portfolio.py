"""Economic/sector daily overlays for the Dolt annual prediction portfolio.

The base portfolio is the Dolt-aligned large-cap annual 12m financial+price
top-quintile equal-weight portfolio.  Overlays use only daily market/sector
state available at the close and are applied with one-day-lagged execution.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ath_reversion_research import download_yahoo_ohlcv
from ath_reversion_research.metrics import summarize_returns


OUTPUT_DIR = Path("ath_reversion_research/reports/economic_overlay_dolt_portfolio")
PREDICTIONS = Path("ath_reversion_research/reports/dolthub_aligned_replication/walk_forward_predictions.csv")
PRICE_CACHE = Path("ath_reversion_research/reports/dolthub_full_fundamental_validation/price_subset_yahoo.csv")
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
COST_BPS = 10.0

SECTOR_ETF = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}
MACRO_SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "HYG", "LQD", *sorted(set(SECTOR_ETF.values()))]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["trade_date", "fwd_12m_return_end_date"])
    signals = predictions[
        (predictions["feature_set"] == "financial_plus_price")
        & (predictions["forward_window"] == "fwd_12m_return")
        & (predictions["regression_bucket"] == "large_cap")
    ].copy()
    sectors = _download_sp500_sectors()
    signals = signals.merge(sectors, on="symbol", how="left")
    symbols = sorted(set(signals["symbol"].dropna().astype(str)) | set(MACRO_SYMBOLS))
    prices = _load_prices(symbols)
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    stock_close = close.reindex(columns=sorted(signals["symbol"].dropna().astype(str).unique()))

    base = _base_top_quintile_weights(signals, stock_close.index, stock_close.columns)
    overlays = {
        "base_top20_equal": base,
        "spy_trend_overlay": _apply_spy_overlay(base, close),
        "sector_trend_overlay": _apply_sector_overlay(base, close, sectors),
        "spy_sector_credit_overlay": _apply_credit_overlay(_apply_sector_overlay(_apply_spy_overlay(base, close), close, sectors), close),
    }
    overlays["spy_sector_credit_overlay_vt25"] = _vol_target_weights(stock_close, overlays["spy_sector_credit_overlay"], 0.25)
    overlays["spy_sector_credit_overlay_vt30"] = _vol_target_weights(stock_close, overlays["spy_sector_credit_overlay"], 0.30)

    returns = {}
    turnovers = {}
    for name, weights in overlays.items():
        returns[name], turnovers[name] = _portfolio_returns(stock_close, weights)
    returns_frame = pd.DataFrame(returns)
    turnover_frame = pd.DataFrame(turnovers)
    first_active = min((weights.abs().sum(axis=1) > 0).idxmax() for weights in overlays.values())
    summary = _summary(returns_frame.loc[first_active:], turnover_frame.loc[first_active:])
    yearly = _yearly(returns_frame.loc[first_active:], turnover_frame.loc[first_active:])
    returns_frame.to_csv(OUTPUT_DIR / "daily_returns.csv", index_label="date")
    turnover_frame.to_csv(OUTPUT_DIR / "daily_turnover.csv", index_label="date")
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    yearly.to_csv(OUTPUT_DIR / "yearly_summary.csv", index=False)
    _write_report(summary, yearly, first_active)
    print(summary.to_string(index=False))
    return 0


def _download_sp500_sectors() -> pd.DataFrame:
    response = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text))[0]
    return pd.DataFrame(
        {
            "symbol": table["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper(),
            "gics_sector": table["GICS Sector"].astype(str),
        }
    )


def _load_prices(symbols: list[str]) -> pd.DataFrame:
    if PRICE_CACHE.exists():
        cached = pd.read_csv(PRICE_CACHE, parse_dates=["date"])
        missing = sorted(set(symbols) - set(cached["symbol"].astype(str).unique()))
        if not missing:
            return cached[cached["symbol"].isin(symbols)].copy()
        extra = download_yahoo_ohlcv(missing, start="2012-01-01", chunk_size=25)
        return pd.concat([cached[cached["symbol"].isin(symbols)], extra], ignore_index=True)
    return download_yahoo_ohlcv(symbols, start="2012-01-01", chunk_size=25)


def _base_top_quintile_weights(signals: pd.DataFrame, trading_index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    target = pd.DataFrame(0.0, index=trading_index, columns=columns)
    for trade_date, group in signals.groupby("trade_date"):
        group = group.dropna(subset=["prediction"]).sort_values("prediction", ascending=False)
        top = group.head(max(1, int(np.ceil(len(group) * 0.20))))
        valid = [symbol for symbol in top["symbol"].astype(str) if symbol in target.columns]
        if not valid:
            continue
        weights = pd.Series(1.0 / len(valid), index=valid)
        start = trading_index.searchsorted(pd.Timestamp(trade_date), side="left")
        end = trading_index.searchsorted(pd.Timestamp(group["fwd_12m_return_end_date"].max()), side="right")
        target.loc[trading_index[start:end], valid] = target.loc[trading_index[start:end], valid].add(weights, axis=1)
    return _normalize(target)


def _apply_spy_overlay(weights: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    spy = close["SPY"]
    trend = spy > spy.rolling(200, min_periods=100).mean()
    mom = spy / spy.shift(126) - 1.0
    vol = spy.pct_change().rolling(63, min_periods=30).std() * np.sqrt(252)
    vol_threshold = vol.rolling(252, min_periods=100).quantile(0.70)
    scale = pd.Series(1.0, index=weights.index)
    scale[(trend.reindex(weights.index) == True) & (mom.reindex(weights.index) > 0)] = 1.15
    scale[trend.reindex(weights.index) == False] = 0.60
    scale[vol.reindex(weights.index) > vol_threshold.reindex(weights.index)] *= 0.80
    return weights.mul(scale.shift(1).fillna(0.75), axis=0)


def _apply_sector_overlay(weights: pd.DataFrame, close: pd.DataFrame, sectors: pd.DataFrame) -> pd.DataFrame:
    symbol_to_etf = sectors.set_index("symbol")["gics_sector"].map(SECTOR_ETF).to_dict()
    sector_scales = {}
    for etf in set(SECTOR_ETF.values()):
        if etf not in close:
            continue
        series = close[etf]
        trend = series > series.rolling(200, min_periods=100).mean()
        mom = series / series.shift(126) - 1.0
        scale = pd.Series(0.65, index=close.index)
        scale[(trend == True) & (mom > 0)] = 1.20
        sector_scales[etf] = scale
    adjusted = weights.copy()
    for symbol in adjusted.columns:
        etf = symbol_to_etf.get(symbol)
        if etf in sector_scales:
            adjusted[symbol] = adjusted[symbol] * sector_scales[etf].reindex(adjusted.index).shift(1).fillna(0.75)
    gross = adjusted.abs().sum(axis=1)
    cap = gross.clip(upper=1.20)
    scale = (cap / gross.replace(0, np.nan)).fillna(0.0)
    return adjusted.mul(scale, axis=0)


def _apply_credit_overlay(weights: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    if "HYG" not in close or "LQD" not in close:
        return weights
    hyg = close["HYG"] / close["HYG"].shift(63) - 1.0
    lqd = close["LQD"] / close["LQD"].shift(63) - 1.0
    credit = hyg - lqd
    scale = pd.Series(1.0, index=weights.index)
    scale[credit.reindex(weights.index) < 0] = 0.75
    scale[credit.reindex(weights.index) > credit.rolling(252, min_periods=100).quantile(0.70).reindex(weights.index)] = 1.10
    return weights.mul(scale.shift(1).fillna(1.0), axis=0)


def _vol_target_weights(close: pd.DataFrame, weights: pd.DataFrame, target_vol: float) -> pd.DataFrame:
    returns, _turnover = _portfolio_returns(close, weights)
    realized = returns.rolling(63, min_periods=20).std().shift(1) * np.sqrt(252)
    scale = (target_vol / realized).clip(lower=0.25, upper=1.60).fillna(0.75)
    return weights.mul(scale, axis=0)


def _portfolio_returns(close: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    returns = close.pct_change().reindex_like(weights).fillna(0.0)
    executed = weights.shift(1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    net = (executed * returns).sum(axis=1) - turnover * (COST_BPS / 10_000.0)
    return net, turnover


def _normalize(weights: pd.DataFrame) -> pd.DataFrame:
    gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
    return weights.div(gross, axis=0).fillna(0.0)


def _summary(returns: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in returns:
        row = summarize_returns(returns[name], turnover[name]).as_dict()
        row["portfolio"] = name
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False)


def _yearly(returns: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in returns.groupby(returns.index.year):
        for name in returns:
            row = summarize_returns(group[name], turnover[name].reindex(group.index)).as_dict()
            row["year"] = int(year)
            row["portfolio"] = name
            rows.append(row)
    return pd.DataFrame(rows)


def _write_report(summary: pd.DataFrame, yearly: pd.DataFrame, first_active: pd.Timestamp) -> None:
    lines = [
        "# Economic Overlay on Dolt Annual Portfolio",
        "",
        "Tests daily economic/market overlays on the Dolt-aligned annual top-quintile portfolio.",
        "",
        "## Leakage controls",
        "",
        "- Overlays use SPY, sector ETF, and credit ETF state known at prior close.",
        "- Portfolio returns use one-day-lagged target weights.",
        "- No realized forward returns are used for allocation.",
        "",
        f"Metrics start: {first_active.date()}.",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "",
        "## Yearly",
        "",
        _markdown_table(yearly),
        "",
        "## Interpretation",
        "",
        "- If overlays improve Sharpe/drawdown versus base, market state helps the annual rank portfolio.",
        "- If not, the simple annual rank portfolio remains preferable on the long Dolt sample.",
    ]
    (OUTPUT_DIR / "ECONOMIC_OVERLAY_DOLT_PORTFOLIO.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if col == "sharpe":
            display[col] = display[col].map(lambda v: f"{float(v):.2f}" if pd.notna(v) else "")
        elif pd.api.types.is_numeric_dtype(display[col]) and col not in {"observations", "year"}:
            if any(token in col for token in ["return", "std", "drawdown", "rate", "turnover"]):
                display[col] = display[col].map(lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())

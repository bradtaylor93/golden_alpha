"""Search for unrelated daily/hourly sleeves with smarter position sizing.

This script uses already-downloaded real Yahoo data from the strategy lab.  It
tests market-neutral, defensive, reversal, and intraday variants that should be
less related to the current high-return large-cap momentum portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research.metrics import summarize_returns


INPUT_DIR = Path("ath_reversion_research/reports/strategy_lab_real_data")
PORTFOLIO_RETURNS = Path("ath_reversion_research/reports/deployable_portfolio/portfolio_returns.csv")
OUTPUT_DIR = Path("ath_reversion_research/reports/unrelated_strategy_lab")
COST_BPS = 10.0

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC", "TLT", "GLD"]
LARGE_CAPS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "AVGO",
    "JPM",
    "V",
    "MA",
    "LLY",
    "UNH",
    "XOM",
    "COST",
    "HD",
    "PG",
    "KO",
    "PEP",
    "WMT",
    "CRM",
    "CAT",
    "DE",
    "NFLX",
    "TSLA",
]


@dataclass(frozen=True)
class Candidate:
    name: str
    grain: str
    returns: pd.Series
    turnover: pd.Series
    gross: pd.Series
    active_positions: pd.Series
    annualization: int


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_bars = pd.read_csv(INPUT_DIR / "daily_downloaded_bars.csv", parse_dates=["date"])
    hourly_bars = pd.read_csv(INPUT_DIR / "hourly_downloaded_bars.csv", parse_dates=["date"])
    benchmark = pd.read_csv(PORTFOLIO_RETURNS, parse_dates=["date"]).set_index("date")["max_return_35_target"]

    daily_close = _pivot(daily_bars, "close")
    daily_volume = _pivot(daily_bars, "volume")
    hourly_close = _pivot(hourly_bars, "close")
    hourly_volume = _pivot(hourly_bars, "volume")

    candidates: list[Candidate] = []
    candidates.extend(_daily_candidates(daily_close, daily_volume))
    candidates.extend(_hourly_candidates(hourly_close, hourly_volume))

    summary = _summarize_candidates(candidates, benchmark)
    daily_candidate_returns = pd.DataFrame(
        {candidate.name: _to_daily(candidate.returns, candidate.grain) for candidate in candidates}
    ).sort_index()
    portfolio_tests = _portfolio_tests(benchmark, daily_candidate_returns, summary)

    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    portfolio_tests.to_csv(OUTPUT_DIR / "portfolio_tests.csv", index=False)
    daily_candidate_returns.to_csv(OUTPUT_DIR / "daily_candidate_returns.csv", index_label="date")
    _write_report(summary, portfolio_tests)
    print(summary.sort_values(["sharpe", "correlation_to_max_return"], ascending=[False, True]).head(20).to_string(index=False))
    print("\nPortfolio tests:")
    print(portfolio_tests.sort_values("annual_return", ascending=False).head(12).to_string(index=False))
    return 0


def _daily_candidates(close: pd.DataFrame, volume: pd.DataFrame) -> list[Candidate]:
    large = close.reindex(columns=[symbol for symbol in LARGE_CAPS if symbol in close])
    sectors = close.reindex(columns=[symbol for symbol in SECTOR_ETFS if symbol in close])
    large_volume = volume.reindex(columns=large.columns)

    weights = {
        "daily_asset_abs_mom_126_top1": _asset_absolute_momentum(close, lookback=126, top_n=1),
        "daily_asset_abs_mom_126_top2": _asset_absolute_momentum(close, lookback=126, top_n=2),
        "daily_asset_abs_mom_252_top1": _asset_absolute_momentum(close, lookback=252, top_n=1),
        "daily_asset_abs_mom_252_top2": _asset_absolute_momentum(close, lookback=252, top_n=2),
        "daily_asset_abs_mom_252_top3": _asset_absolute_momentum(close, lookback=252, top_n=3),
        "daily_sector_market_neutral_rs": _market_neutral_momentum(sectors, lookback=63, top_n=3),
        "daily_large_cap_5d_reversal": _short_term_reversal(large, lookback=5, top_n=8, trend_filter=True),
        "daily_large_cap_rsi_reversal": _rsi_reversal(large, window=14, top_n=8),
        "daily_large_cap_lowvol_trend": _low_vol_trend(large, top_n=8),
        "daily_large_cap_volume_thrust": _volume_thrust(large, large_volume, top_n=8),
        "daily_risk_off_tlt_gld": _risk_off_defensive(close),
    }
    return [_evaluate_weights(name, "1d", w, close, 252) for name, w in weights.items()]


def _hourly_candidates(close: pd.DataFrame, volume: pd.DataFrame) -> list[Candidate]:
    large = close.reindex(columns=[symbol for symbol in LARGE_CAPS[:12] if symbol in close])
    sectors = close.reindex(columns=[symbol for symbol in SECTOR_ETFS if symbol in close])
    large_volume = volume.reindex(columns=large.columns)
    sector_volume = volume.reindex(columns=sectors.columns)

    specs: dict[str, pd.DataFrame] = {
        "hourly_large_cap_vwap_fade_1h": _session_vwap_fade(large, large_volume, top_n=5),
        "hourly_sector_vwap_fade_1h": _session_vwap_fade(sectors, sector_volume, top_n=4),
        "hourly_large_cap_rsi_fade_1h": _intraday_rsi_fade(large, top_n=5),
        "hourly_sector_market_neutral_mom_1h": _market_neutral_momentum(sectors, lookback=24, top_n=3),
        "hourly_large_cap_opening_range_1h": _opening_range_breakout(large, top_n=5),
    }
    results = [_evaluate_weights(name, "1h", weights, close, 252 * 6) for name, weights in specs.items()]

    close_2h = _resample_prices(close, "2h")
    volume_2h = _resample_prices(volume, "2h", how="sum")
    large_2h = close_2h.reindex(columns=[symbol for symbol in LARGE_CAPS[:12] if symbol in close_2h])
    large_volume_2h = volume_2h.reindex(columns=large_2h.columns)
    results.extend(
        [
            _evaluate_weights(
                "hourly_large_cap_vwap_fade_2h",
                "2h",
                _session_vwap_fade(large_2h, large_volume_2h, top_n=5),
                close_2h,
                252 * 3,
            ),
            _evaluate_weights(
                "hourly_large_cap_rsi_fade_2h",
                "2h",
                _intraday_rsi_fade(large_2h, top_n=5),
                close_2h,
                252 * 3,
            ),
        ]
    )
    return results


def _evaluate_weights(
    name: str,
    grain: str,
    weights: pd.DataFrame,
    close: pd.DataFrame,
    annualization: int,
) -> Candidate:
    returns = close.pct_change().fillna(0.0)
    weights = weights.reindex_like(returns).fillna(0.0)
    executed = weights.shift(1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    net = (executed * returns).sum(axis=1) - turnover * (COST_BPS / 10_000.0)
    gross = executed.abs().sum(axis=1)
    active = (executed != 0.0).sum(axis=1)
    return Candidate(
        name=name,
        grain=grain,
        returns=net,
        turnover=turnover,
        gross=gross,
        active_positions=active,
        annualization=annualization,
    )


def _market_neutral_momentum(close: pd.DataFrame, lookback: int, top_n: int) -> pd.DataFrame:
    score = close / close.shift(lookback) - 1.0
    vol = close.pct_change().rolling(max(20, lookback), min_periods=max(10, lookback // 2)).std()
    long = _rank_weights(score / vol, top_n=top_n, side=0.5)
    short = _rank_weights(-(score / vol), top_n=top_n, side=-0.5)
    return (long + short).fillna(0.0)


def _asset_absolute_momentum(close: pd.DataFrame, lookback: int, top_n: int) -> pd.DataFrame:
    symbols = [
        symbol
        for symbol in ["SPY", "TLT", "GLD", "XLP", "XLU", "XLV", "XLE", "XLF", "XLK", "XLI", "XLY"]
        if symbol in close
    ]
    asset_close = close.reindex(columns=symbols)
    score = asset_close / asset_close.shift(lookback) - 1.0
    return _vol_scaled_long_weights(score.where(score > 0.0), asset_close, top_n=top_n)


def _short_term_reversal(close: pd.DataFrame, lookback: int, top_n: int, trend_filter: bool) -> pd.DataFrame:
    score = -(close / close.shift(lookback) - 1.0)
    if trend_filter:
        score = score.where(close > close.rolling(200, min_periods=100).mean())
    return _vol_scaled_long_weights(score.where(score > 0.0), close, top_n=top_n)


def _rsi_reversal(close: pd.DataFrame, window: int, top_n: int) -> pd.DataFrame:
    delta = close.diff()
    gains = delta.clip(lower=0.0).rolling(window, min_periods=window).mean()
    losses = (-delta.clip(upper=0.0)).rolling(window, min_periods=window).mean()
    rsi = 100.0 - 100.0 / (1.0 + gains / losses.replace(0.0, np.nan))
    score = (35.0 - rsi).where((rsi < 35.0) & (close > close.rolling(100, min_periods=50).mean()))
    return _vol_scaled_long_weights(score, close, top_n=top_n)


def _low_vol_trend(close: pd.DataFrame, top_n: int) -> pd.DataFrame:
    trend = close / close.shift(126) - 1.0
    vol = close.pct_change().rolling(63, min_periods=30).std()
    score = trend / vol
    score = score.where((trend > 0.0) & (close > close.rolling(200, min_periods=100).mean()))
    return _vol_scaled_long_weights(score, close, top_n=top_n)


def _volume_thrust(close: pd.DataFrame, volume: pd.DataFrame, top_n: int) -> pd.DataFrame:
    ret = close.pct_change(5)
    rel_vol = volume / volume.rolling(60, min_periods=20).mean()
    score = ret * rel_vol
    score = score.where((ret > 0.03) & (rel_vol > 1.5))
    return _vol_scaled_long_weights(score, close, top_n=top_n)


def _risk_off_defensive(close: pd.DataFrame) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    spy = close["SPY"]
    bear = spy < spy.rolling(200, min_periods=100).mean()
    candidates = close.reindex(columns=[symbol for symbol in ["TLT", "GLD", "XLP", "XLU"] if symbol in close])
    score = candidates / candidates.shift(63) - 1.0
    defensive = _vol_scaled_long_weights(score.where(score > 0.0), candidates, top_n=2)
    weights.loc[:, defensive.columns] = defensive.where(bear, 0.0)
    return weights


def _session_vwap_fade(close: pd.DataFrame, volume: pd.DataFrame, top_n: int) -> pd.DataFrame:
    session = pd.Series(close.index.date, index=close.index)
    vwap = (close * volume).groupby(session).cumsum() / volume.groupby(session).cumsum()
    dev = close / vwap - 1.0
    vol = close.pct_change().rolling(40, min_periods=20).std()
    z = dev / vol.replace(0.0, np.nan)
    long = _rank_weights((-z).where(z < -1.25), top_n=top_n, side=0.5)
    short = _rank_weights(z.where(z > 1.25), top_n=top_n, side=-0.5)
    return (long + short).fillna(0.0)


def _intraday_rsi_fade(close: pd.DataFrame, top_n: int) -> pd.DataFrame:
    delta = close.diff()
    gains = delta.clip(lower=0.0).rolling(8, min_periods=8).mean()
    losses = (-delta.clip(upper=0.0)).rolling(8, min_periods=8).mean()
    rsi = 100.0 - 100.0 / (1.0 + gains / losses.replace(0.0, np.nan))
    long = _rank_weights((25.0 - rsi).where(rsi < 25.0), top_n=top_n, side=0.5)
    short = _rank_weights((rsi - 75.0).where(rsi > 75.0), top_n=top_n, side=-0.5)
    return (long + short).fillna(0.0)


def _opening_range_breakout(close: pd.DataFrame, top_n: int) -> pd.DataFrame:
    session = pd.Series(close.index.date, index=close.index)
    first_close = close.groupby(session).transform("first")
    range_proxy = close.groupby(session).cummax() / close.groupby(session).cummin() - 1.0
    score = close / first_close - 1.0
    signal = score.where((score > 0.005) & (range_proxy > 0.008))
    return _vol_scaled_long_weights(signal, close, top_n=top_n)


def _vol_scaled_long_weights(score: pd.DataFrame, close: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected = score.rank(axis=1, ascending=False, method="first") <= top_n
    vol = close.pct_change().rolling(63, min_periods=20).std().replace(0.0, np.nan)
    raw = selected.astype(float) / vol
    raw = raw.where(selected, 0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    return raw.div(gross, axis=0).fillna(0.0)


def _rank_weights(score: pd.DataFrame, top_n: int, side: float) -> pd.DataFrame:
    ranks = score.rank(axis=1, ascending=False, method="first")
    selected = ranks <= top_n
    raw = score.abs().where(selected, 0.0).fillna(0.0)
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    return raw.div(gross, axis=0).fillna(0.0) * side


def _summarize_candidates(candidates: list[Candidate], benchmark: pd.Series) -> pd.DataFrame:
    rows = []
    benchmark = benchmark.sort_index()
    for candidate in candidates:
        summary = summarize_returns(candidate.returns, candidate.turnover, annualization=candidate.annualization).as_dict()
        daily_returns = _to_daily(candidate.returns, candidate.grain)
        aligned = pd.concat([daily_returns, benchmark], axis=1, join="inner").dropna()
        corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) if len(aligned) > 20 else np.nan
        summary.update(
            {
                "strategy": candidate.name,
                "grain": candidate.grain,
                "correlation_to_max_return": corr,
                "avg_gross_exposure": float(candidate.gross.mean()),
                "avg_active_positions": float(candidate.active_positions.mean()),
            }
        )
        rows.append(summary)
    return pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False).reset_index(drop=True)


def _portfolio_tests(
    benchmark: pd.Series,
    candidates: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    eligible = summary[
        (summary["annual_return"] > 0.0)
        & (summary["sharpe"] > 0.0)
        & (summary["correlation_to_max_return"].abs() < 0.70)
    ]["strategy"].head(5).tolist()
    rows = []
    base = benchmark.sort_index().rename("base")
    rows.append(_portfolio_summary_row("base_max_return_35", base))
    for name in eligible:
        addon = candidates[name].reindex(base.index).fillna(0.0)
        for addon_weight in [0.10, 0.20, 0.30]:
            combo = (1.0 - addon_weight) * base + addon_weight * addon
            rows.append(_portfolio_summary_row(f"base_{int((1-addon_weight)*100)}_{name}_{int(addon_weight*100)}", combo))
        for overlay_weight in [0.25, 0.50]:
            overlay = _vol_target(base + overlay_weight * addon, target_vol=0.35)
            rows.append(_portfolio_summary_row(f"base_overlay_{name}_{int(overlay_weight*100)}_retarget35", overlay))
    if eligible:
        equal_addon = candidates[eligible].reindex(base.index).fillna(0.0).mean(axis=1)
        for addon_weight in [0.15, 0.25]:
            combo = (1.0 - addon_weight) * base + addon_weight * equal_addon
            rows.append(_portfolio_summary_row(f"base_plus_unrelated_basket_{int(addon_weight*100)}", combo))
        for overlay_weight in [0.25, 0.50]:
            overlay = _vol_target(base + overlay_weight * equal_addon, target_vol=0.35)
            rows.append(_portfolio_summary_row(f"base_overlay_unrelated_basket_{int(overlay_weight*100)}_retarget35", overlay))
    return pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False).reset_index(drop=True)


def _portfolio_summary_row(name: str, returns: pd.Series) -> dict[str, float | int | str]:
    summary = summarize_returns(returns).as_dict()
    summary["portfolio"] = name
    return summary


def _vol_target(returns: pd.Series, target_vol: float, lookback: int = 63, max_leverage: float = 2.5) -> pd.Series:
    realized = returns.rolling(lookback, min_periods=20).std().shift(1) * np.sqrt(252)
    scale = (target_vol / realized).clip(lower=0.25, upper=max_leverage).fillna(0.75)
    return returns * scale


def _to_daily(returns: pd.Series, grain: str) -> pd.Series:
    returns = returns.sort_index()
    if grain == "1d":
        return returns
    grouped = returns.groupby(pd.Series(returns.index.date, index=returns.index))
    out = grouped.apply(lambda values: float(np.prod(1.0 + values.to_numpy(dtype=float)) - 1.0))
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _pivot(bars: pd.DataFrame, field: str) -> pd.DataFrame:
    return bars.pivot(index="date", columns="symbol", values=field).sort_index()


def _resample_prices(frame: pd.DataFrame, rule: str, how: str = "last") -> pd.DataFrame:
    if how == "sum":
        return frame.resample(rule).sum(min_count=1).dropna(how="all")
    return frame.resample(rule).last().dropna(how="all")


def _write_report(summary: pd.DataFrame, portfolio_tests: pd.DataFrame) -> None:
    top = summary.sort_values(["sharpe", "annual_return"], ascending=False).head(12)
    low_corr = summary.reindex(summary["correlation_to_max_return"].abs().sort_values().index).head(12)
    tests = portfolio_tests.head(12)
    lines = [
        "# Unrelated Strategy Lab Results",
        "",
        "Searched for daily and hourly sleeves that are less related to the current high-return large-cap momentum portfolio.",
        "All candidates use causal one-bar-lag execution, 10 bps turnover cost, and inverse-volatility or signal-strength sizing.",
        "",
        "## Best standalone candidates",
        "",
        _markdown_table(top),
        "",
        "## Lowest-correlation candidates",
        "",
        _markdown_table(low_corr),
        "",
        "## Portfolio combination tests",
        "",
        _markdown_table(tests),
        "",
        "## Interpretation",
        "",
        "- The best addition was asset-class absolute momentum with inverse-volatility sizing. It is not fully unrelated, but it is meaningfully less concentrated than the large-cap stock-only core.",
        "- Overlaying the asset-momentum sleeve and re-targeting volatility improved return and drawdown versus simply replacing part of the core.",
        "- Hourly VWAP/RSI fade variants remained fragile on Yahoo intraday data after costs, even with signal-strength sizing.",
        "- Market-neutral sector momentum and risk-off ETF trend had low correlation but were too weak after costs.",
    ]
    (OUTPUT_DIR / "UNRELATED_STRATEGY_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "strategy",
        "portfolio",
        "grain",
        "annual_return",
        "annual_std",
        "sharpe",
        "max_drawdown",
        "correlation_to_max_return",
        "total_return",
    ]
    display = frame[[column for column in columns if column in frame.columns]].copy()
    for column in ["annual_return", "annual_std", "max_drawdown", "total_return"]:
        if column in display:
            display[column] = display[column].map(lambda value: f"{float(value) * 100:.2f}%")
    for column in ["sharpe", "correlation_to_max_return"]:
        if column in display:
            display[column] = display[column].map(lambda value: f"{float(value):.2f}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())

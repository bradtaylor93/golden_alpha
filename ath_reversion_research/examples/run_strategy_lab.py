"""Explore sector rotation, momentum, breakout, and VWAP variants on real data.

The goal is breadth-first strategy discovery rather than final production
research.  The script deliberately uses simple, causal rules and identical
portfolio accounting so variants can be compared quickly across universes,
regimes, and daily/hourly grains.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from ath_reversion_research import symbols_for
from ath_reversion_research.data import download_yahoo_ohlcv
from ath_reversion_research.regimes import classify_market_regimes


OUTPUT = Path("ath_reversion_research/reports/strategy_lab_real_data")
TRADING_DAYS = 252
COST_BPS = 10.0

SECTOR_ETFS = [
    "SPY",
    "XLK",
    "XLF",
    "XLV",
    "XLY",
    "XLP",
    "XLE",
    "XLI",
    "XLB",
    "XLU",
    "XLRE",
    "XLC",
    "TLT",
    "GLD",
]
DEFENSIVE = ["XLP", "XLU", "XLV", "TLT", "GLD"]
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
LOW_CAPS = list(symbols_for(["lower_market_cap_under_10bn_sample"]))


@dataclass(frozen=True)
class StrategyResult:
    name: str
    universe: str
    grain: str
    daily: pd.DataFrame
    regime: pd.DataFrame
    annualization: int


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    daily_symbols = sorted(set(SECTOR_ETFS + LARGE_CAPS + LOW_CAPS + ["SPY"]))
    daily_bars = download_yahoo_ohlcv(daily_symbols, start="2010-01-01")
    daily_bars.to_csv(OUTPUT / "daily_downloaded_bars.csv", index=False)

    close = _pivot(daily_bars, "close")
    volume = _pivot(daily_bars, "volume")
    daily_results = _run_daily_research(close, volume, daily_bars)

    hourly_symbols = sorted(set(SECTOR_ETFS + LARGE_CAPS[:12] + LOW_CAPS[:12] + ["SPY"]))
    hourly_bars = _download_hourly(hourly_symbols, period="729d")
    hourly_bars.to_csv(OUTPUT / "hourly_downloaded_bars.csv", index=False)
    hourly_results = _run_hourly_research(hourly_bars)

    all_results = daily_results + hourly_results
    summary = pd.concat([_summary_row(result) for result in all_results], ignore_index=True)
    regime = pd.concat([_regime_rows(result) for result in all_results], ignore_index=True)
    daily_returns = pd.concat(
        [result.daily.assign(strategy=result.name, universe=result.universe, grain=result.grain) for result in all_results],
        ignore_index=True,
    )

    summary.sort_values(["sharpe", "annual_return"], ascending=False).to_csv(OUTPUT / "summary.csv", index=False)
    regime.sort_values(["regime", "sharpe"], ascending=[True, False]).to_csv(
        OUTPUT / "regime_summary.csv", index=False
    )
    daily_returns.to_csv(OUTPUT / "strategy_returns.csv", index=False)

    bear = regime[regime["regime"].astype(str).str.startswith("bear")].copy()
    bear.sort_values(["sharpe", "annual_return"], ascending=False).to_csv(OUTPUT / "bear_market_rankings.csv", index=False)
    _write_report(summary, regime, bear, daily_bars, hourly_bars)
    print(summary.sort_values("sharpe", ascending=False).head(15).to_string(index=False))
    return 0


def _run_daily_research(close: pd.DataFrame, volume: pd.DataFrame, bars: pd.DataFrame) -> list[StrategyResult]:
    regimes = classify_market_regimes(bars)
    specs: list[tuple[str, str, pd.DataFrame]] = []

    sector_close = close.reindex(columns=[symbol for symbol in SECTOR_ETFS if symbol in close])
    specs.extend(
        [
            (
                "sector_rotation_top3_63d",
                "sector_etfs",
                _top_momentum_weights(sector_close.drop(columns=["SPY"], errors="ignore"), lookback=63, top_n=3),
            ),
            (
                "sector_rotation_regime_defensive",
                "sector_etfs",
                _sector_regime_rotation(sector_close),
            ),
            (
                "sector_breakout_55d",
                "sector_etfs",
                _breakout_weights(sector_close.drop(columns=["SPY"], errors="ignore"), breakout=55, trend=126, top_n=4),
            ),
            (
                "sector_bear_short_weak_63d",
                "sector_etfs",
                _bear_short_weak(sector_close.drop(columns=["SPY"], errors="ignore"), benchmark=close["SPY"], lookback=63, bottom_n=3),
            ),
        ]
    )

    for label, symbols, top_n in [
        ("large_caps", LARGE_CAPS, 10),
        ("lower_caps", LOW_CAPS, 8),
    ]:
        universe_close = close.reindex(columns=[symbol for symbol in symbols if symbol in close])
        universe_volume = volume.reindex(columns=universe_close.columns)
        specs.extend(
            [
                (
                    f"{label}_relative_strength_126d",
                    label,
                    _top_momentum_weights(universe_close, lookback=126, top_n=top_n, require_above_ma=True),
                ),
                (
                    f"{label}_breakout_55d_trend",
                    label,
                    _breakout_weights(universe_close, breakout=55, trend=126, top_n=top_n),
                ),
                (
                    f"{label}_momentum_12_1",
                    label,
                    _top_momentum_weights(universe_close, lookback=252, skip=21, top_n=top_n, require_above_ma=True),
                ),
                (
                    f"{label}_vwap_reclaim_20d",
                    label,
                    _vwap_reclaim_weights(universe_close, universe_volume, lookback=20, top_n=top_n),
                ),
                (
                    f"{label}_bear_short_weak_63d",
                    label,
                    _bear_short_weak(universe_close, benchmark=close["SPY"], lookback=63, bottom_n=top_n),
                ),
            ]
        )

    return [_evaluate(name, universe, "1d", weights, close, regimes, annualization=252) for name, universe, weights in specs]


def _run_hourly_research(hourly_bars: pd.DataFrame) -> list[StrategyResult]:
    results: list[StrategyResult] = []
    for grain, bars in [
        ("1h", hourly_bars),
        ("2h", _resample_intraday(hourly_bars, "2h")),
        ("4h", _resample_intraday(hourly_bars, "4h")),
    ]:
        close = _pivot(bars, "close")
        volume = _pivot(bars, "volume")
        regimes = _intraday_regimes(close)
        for label, symbols, top_n in [
            ("hourly_sector", SECTOR_ETFS, 3),
            ("hourly_large_caps", LARGE_CAPS[:12], 5),
            ("hourly_lower_caps", LOW_CAPS[:12], 4),
        ]:
            universe_close = close.reindex(columns=[symbol for symbol in symbols if symbol in close])
            universe_volume = volume.reindex(columns=universe_close.columns)
            specs = [
                (
                    f"{label}_momentum_{grain}_24bar",
                    _top_momentum_weights(universe_close, lookback=24, top_n=top_n, require_above_ma=False),
                ),
                (
                    f"{label}_breakout_{grain}_40bar",
                    _breakout_weights(universe_close, breakout=40, trend=80, top_n=top_n),
                ),
                (
                    f"{label}_vwap_reclaim_{grain}_session",
                    _session_vwap_reclaim(universe_close, universe_volume, top_n=top_n),
                ),
                (
                    f"{label}_bear_short_weak_{grain}_24bar",
                    _bear_short_weak(universe_close, benchmark=close["SPY"], lookback=24, bottom_n=top_n),
                ),
            ]
            for name, weights in specs:
                results.append(_evaluate(name, label, grain, weights, close, regimes, annualization=_annualization_for(grain)))
    return results


def _evaluate(
    name: str,
    universe: str,
    grain: str,
    weights: pd.DataFrame,
    close: pd.DataFrame,
    regimes: pd.DataFrame,
    annualization: int,
) -> StrategyResult:
    returns = close.pct_change().fillna(0.0)
    weights = weights.reindex_like(returns).fillna(0.0)
    executed = weights.shift(1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    gross = executed.abs().sum(axis=1)
    cost = turnover * (COST_BPS / 10_000.0)
    net = (executed * returns).sum(axis=1) - cost
    daily = pd.DataFrame(
        {
            "date": returns.index,
            "net_return": net.to_numpy(),
            "turnover": turnover.to_numpy(),
            "gross_exposure": gross.to_numpy(),
            "active_positions": (executed != 0.0).sum(axis=1).to_numpy(),
        }
    )
    regime_daily = daily.merge(regimes, on="date", how="left")
    regime_rows = []
    for regime_name, group in regime_daily.groupby("regime", dropna=False):
        summary = _summarize_returns(
            group.set_index("date")["net_return"],
            group.set_index("date")["turnover"],
            annualization=annualization,
        )
        summary["regime"] = regime_name if pd.notna(regime_name) else "unclassified"
        regime_rows.append(summary)
    regime = pd.DataFrame(regime_rows)
    return StrategyResult(
        name=name,
        universe=universe,
        grain=grain,
        daily=daily,
        regime=regime,
        annualization=annualization,
    )


def _top_momentum_weights(
    close: pd.DataFrame,
    lookback: int,
    top_n: int,
    skip: int = 0,
    require_above_ma: bool = False,
) -> pd.DataFrame:
    score = close.shift(skip) / close.shift(skip + lookback) - 1.0
    if require_above_ma:
        score = score.where(close > close.rolling(200, min_periods=100).mean())
    score = score.where(score > 0.0)
    return _rank_to_weights(score, top_n=top_n, side=1.0)


def _sector_regime_rotation(close: pd.DataFrame) -> pd.DataFrame:
    benchmark = close["SPY"]
    bull = (benchmark > benchmark.rolling(200, min_periods=100).mean()) & (benchmark.pct_change(63) > 0.0)
    cyclical = [symbol for symbol in close.columns if symbol not in {"SPY", "TLT", "GLD"}]
    risk_on = _top_momentum_weights(close[cyclical], lookback=63, top_n=3)
    defensive = _top_momentum_weights(close[[symbol for symbol in DEFENSIVE if symbol in close]], lookback=63, top_n=2)
    return risk_on.where(bull, defensive).fillna(0.0)


def _breakout_weights(close: pd.DataFrame, breakout: int, trend: int, top_n: int) -> pd.DataFrame:
    prior_high = close.shift(1).rolling(breakout, min_periods=max(20, breakout // 2)).max()
    score = close / close.shift(breakout) - 1.0
    signal = score.where((close >= prior_high) & (close > close.rolling(trend, min_periods=max(20, trend // 2)).mean()))
    return _rank_to_weights(signal, top_n=top_n, side=1.0)


def _vwap_reclaim_weights(close: pd.DataFrame, volume: pd.DataFrame, lookback: int, top_n: int) -> pd.DataFrame:
    dollar_volume = close * volume
    vwap = dollar_volume.rolling(lookback, min_periods=max(5, lookback // 2)).sum() / volume.rolling(
        lookback, min_periods=max(5, lookback // 2)
    ).sum()
    reclaim = (close > vwap) & (close.shift(1) <= vwap.shift(1))
    score = (close / vwap - 1.0).where(reclaim & (vwap > vwap.shift(5)))
    return _rank_to_weights(score, top_n=top_n, side=1.0)


def _session_vwap_reclaim(close: pd.DataFrame, volume: pd.DataFrame, top_n: int) -> pd.DataFrame:
    session = pd.Series(close.index.date, index=close.index)
    dollar = close * volume
    vwap = dollar.groupby(session).cumsum() / volume.groupby(session).cumsum()
    reclaim = (close > vwap) & (close.shift(1) <= vwap.shift(1))
    score = (close / vwap - 1.0).where(reclaim)
    return _rank_to_weights(score, top_n=top_n, side=1.0)


def _bear_short_weak(close: pd.DataFrame, benchmark: pd.Series, lookback: int, bottom_n: int) -> pd.DataFrame:
    bear = benchmark < benchmark.rolling(max(100, lookback * 3), min_periods=max(50, lookback)).mean()
    score = close / close.shift(lookback) - 1.0
    weak = score.where(score < 0.0)
    weights = _rank_to_weights(-weak, top_n=bottom_n, side=-1.0)
    return weights.where(bear, 0.0).fillna(0.0)


def _rank_to_weights(score: pd.DataFrame, top_n: int, side: float) -> pd.DataFrame:
    ranks = score.rank(axis=1, ascending=False, method="first")
    selected = ranks <= top_n
    counts = selected.sum(axis=1).replace(0, np.nan)
    weights = selected.div(counts, axis=0).fillna(0.0) * side
    return weights.reindex(columns=score.columns).fillna(0.0)


def _pivot(bars: pd.DataFrame, field: str) -> pd.DataFrame:
    return bars.pivot(index="date", columns="symbol", values=field).sort_index()


def _download_hourly(symbols: list[str], period: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=symbols,
        period=period,
        interval="60m",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        available = set(raw.columns.get_level_values(0))
        for symbol in symbols:
            if symbol not in available:
                continue
            frame = raw[symbol].rename(columns={column: str(column).lower() for column in raw[symbol].columns})
            if {"open", "high", "low", "close", "volume"} <= set(frame.columns):
                out = frame.reset_index().rename(columns={frame.index.name or "Datetime": "date"})
                if "date" not in out:
                    out = out.rename(columns={out.columns[0]: "date"})
                out["symbol"] = symbol
                out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
                frames.append(out[["date", "symbol", "open", "high", "low", "close", "volume"]])
    if not frames:
        raise ValueError("Yahoo Finance returned no hourly bars")
    return pd.concat(frames, ignore_index=True).dropna(subset=["close"]).sort_values(["symbol", "date"])


def _resample_intraday(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    pieces = []
    for symbol, group in bars.groupby("symbol"):
        g = group.set_index("date").sort_index()
        resampled = g.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        resampled["symbol"] = symbol
        pieces.append(resampled.reset_index())
    return pd.concat(pieces, ignore_index=True).sort_values(["symbol", "date"])


def _intraday_regimes(close: pd.DataFrame) -> pd.DataFrame:
    benchmark = close["SPY"]
    trend = benchmark.rolling(200, min_periods=80).mean()
    returns = benchmark.pct_change()
    vol = returns.rolling(80, min_periods=40).std()
    threshold = vol.rolling(400, min_periods=80).quantile(0.7)
    direction = np.where(benchmark >= trend, "bull", "bear")
    vol_label = np.where(vol >= threshold, "high_vol", "normal_vol")
    regime = pd.Series(direction, index=close.index).str.cat(pd.Series(vol_label, index=close.index), sep="_")
    regime = regime.where(trend.notna(), "unclassified")
    return pd.DataFrame({"date": close.index, "regime": regime.to_numpy()})


def _summary_row(result: StrategyResult) -> pd.DataFrame:
    daily = result.daily.set_index("date")
    summary = _summarize_returns(daily["net_return"], daily["turnover"], annualization=result.annualization)
    summary.update(
        {
            "strategy": result.name,
            "universe": result.universe,
            "grain": result.grain,
            "avg_gross_exposure": float(result.daily["gross_exposure"].mean()),
            "avg_active_positions": float(result.daily["active_positions"].mean()),
        }
    )
    return pd.DataFrame([summary])


def _regime_rows(result: StrategyResult) -> pd.DataFrame:
    out = result.regime.copy()
    out["strategy"] = result.name
    out["universe"] = result.universe
    out["grain"] = result.grain
    return out


def _annualization_for(grain: str) -> int:
    if grain == "1h":
        return 252 * 6
    if grain == "2h":
        return 252 * 3
    return 252 * 2


def _summarize_returns(returns: pd.Series, turnover: pd.Series, annualization: int) -> dict[str, float | int]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return {
            "observations": 0,
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_std": np.nan,
            "sharpe": np.nan,
            "average_daily_return": np.nan,
            "average_daily_turnover": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
        }
    equity = (1.0 + clean).cumprod()
    periods = max(len(clean) / annualization, 1.0 / annualization)
    annual_return = float(equity.iloc[-1] ** (1.0 / periods) - 1.0)
    annual_std = float(clean.std(ddof=1) * np.sqrt(annualization)) if len(clean) > 1 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    aligned_turnover = turnover.reindex(clean.index).fillna(0.0)
    return {
        "observations": int(len(clean)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "annual_return": annual_return,
        "annual_std": annual_std,
        "sharpe": float(annual_return / annual_std) if annual_std > 0 else np.nan,
        "average_daily_return": float(clean.mean()),
        "average_daily_turnover": float(aligned_turnover.mean()),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((clean > 0.0).mean()),
    }


def _write_report(summary: pd.DataFrame, regime: pd.DataFrame, bear: pd.DataFrame, daily_bars: pd.DataFrame, hourly_bars: pd.DataFrame) -> None:
    top = summary.sort_values("sharpe", ascending=False).head(12)
    bear_top = bear.sort_values("sharpe", ascending=False).head(12)
    lines = [
        "# Strategy Lab Real-Data Results",
        "",
        "Exploratory tests for sector rotation/relative strength, breakout, momentum, VWAP reclaim, and bear-market short-weak variants.",
        "",
        "## Data",
        "",
        f"- Daily rows: {len(daily_bars):,}; daily symbols: {daily_bars['symbol'].nunique()}.",
        f"- Hourly rows: {len(hourly_bars):,}; hourly symbols: {hourly_bars['symbol'].nunique()}.",
        "- Daily sample starts in 2010. Hourly Yahoo data is limited to the recent ~729 days.",
        f"- Transaction cost assumption: {COST_BPS:.0f} bps on absolute turnover.",
        "",
        "## Top overall variants",
        "",
        _markdown_table(top),
        "",
        "## Best bear-regime variants",
        "",
        _markdown_table(bear_top),
        "",
        "## Interpretation",
        "",
        "- Sector rotation and relative-strength variants are intended to be risk-on engines.",
        "- Bear-market candidates are primarily the explicit short-weak momentum variants and defensive sector rotation.",
        "- Hourly variants are intentionally short-history exploratory signals; treat them as hypothesis generation.",
        "- Full CSV outputs: `summary.csv`, `regime_summary.csv`, `bear_market_rankings.csv`, and `strategy_returns.csv`.",
    ]
    (OUTPUT / "STRATEGY_LAB_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = ["strategy", "universe", "grain", "regime", "annual_return", "annual_std", "sharpe", "max_drawdown"]
    available = [column for column in columns if column in frame.columns]
    display = frame[available].copy()
    for column in ["annual_return", "annual_std", "max_drawdown"]:
        if column in display:
            display[column] = (display[column] * 100).map(lambda value: f"{value:.2f}%")
    if "sharpe" in display:
        display["sharpe"] = display["sharpe"].map(lambda value: f"{value:.2f}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())

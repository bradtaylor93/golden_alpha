"""Expanded stock-universe signal-strength research.

This script tests whether broader stock breadth and more intelligent position
sizing improve the current high-return/ML portfolio.  It deliberately compares
new sleeves against the saved ML meta portfolio instead of only looking at
standalone Sharpe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research import download_yahoo_ohlcv
from ath_reversion_research.metrics import summarize_returns


OUTPUT_DIR = Path("ath_reversion_research/reports/expanded_stock_signals")
CURRENT_BEST = Path("ath_reversion_research/reports/ml_meta_portfolio/returns.csv")
COST_BPS = 10.0

EXPANDED_STOCKS = sorted(
    {
        # Technology / communication services.
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
        "AVGO",
        "ORCL",
        "ADBE",
        "CRM",
        "AMD",
        "INTC",
        "CSCO",
        "NFLX",
        "TSLA",
        "UBER",
        "SHOP",
        "SNOW",
        "DDOG",
        "NET",
        "MDB",
        "CRWD",
        "ZS",
        "PANW",
        "NOW",
        "PLTR",
        "ANET",
        "TXN",
        "QCOM",
        "MU",
        "AMAT",
        "LRCX",
        "KLAC",
        "SNPS",
        "CDNS",
        "ADSK",
        "TEAM",
        "WDAY",
        "INTU",
        "IBM",
        "ACN",
        "FI",
        "APH",
        "NXPI",
        "MRVL",
        # Financials.
        "JPM",
        "BAC",
        "WFC",
        "C",
        "GS",
        "MS",
        "BLK",
        "AXP",
        "V",
        "MA",
        "PYPL",
        "COIN",
        "SCHW",
        "BX",
        "KKR",
        "ICE",
        "CME",
        "CB",
        "PGR",
        "TRV",
        "AON",
        "MMC",
        "USB",
        "PNC",
        # Healthcare.
        "LLY",
        "UNH",
        "JNJ",
        "MRK",
        "ABBV",
        "PFE",
        "TMO",
        "DHR",
        "ISRG",
        "REGN",
        "VRTX",
        "MRNA",
        "DXCM",
        "ABT",
        "BSX",
        "SYK",
        "MDT",
        "GILD",
        "AMGN",
        "BMY",
        "ZTS",
        "HCA",
        "CI",
        "ELV",
        # Consumer / retail.
        "WMT",
        "COST",
        "HD",
        "LOW",
        "PG",
        "KO",
        "PEP",
        "MCD",
        "SBUX",
        "NKE",
        "LULU",
        "TGT",
        "TJX",
        "ROST",
        "CMG",
        "YUM",
        "DPZ",
        "ORLY",
        "AZO",
        "ULTA",
        "EL",
        "CL",
        "KMB",
        "MDLZ",
        # Industrials / energy / materials.
        "CAT",
        "DE",
        "GE",
        "HON",
        "UNP",
        "UPS",
        "RTX",
        "LMT",
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "FCX",
        "NEM",
        "LIN",
        "APD",
        "SHW",
        "ECL",
        "EMR",
        "ETN",
        "PH",
        "ITW",
        "MMM",
        "BA",
        "GD",
        "NOC",
        "WM",
        "URI",
        "CSX",
        "NSC",
        # Real estate / utilities / defensive.
        "NEE",
        "DUK",
        "SO",
        "AEP",
        "EXC",
        "SRE",
        "AMT",
        "PLD",
        "EQIX",
        "O",
        # Liquid mid/smaller growth names.
        "FIVE",
        "WING",
        "CROX",
        "CELH",
        "BROS",
        "BOOT",
        "DUOL",
        "NCNO",
        "S",
        "PATH",
        "GTLB",
        "AI",
        "UPST",
        "RUN",
        "ENPH",
        "SEDG",
        "SHAK",
        "CHWY",
        "BILL",
        "OPEN",
        "BL",
        "ASAN",
    }
)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bars = download_yahoo_ohlcv(EXPANDED_STOCKS, start="2010-01-01", chunk_size=20)
    close = bars.pivot(index="date", columns="symbol", values="close").sort_index()
    close = close.dropna(axis=1, thresh=int(len(close) * 0.45))
    base = pd.read_csv(CURRENT_BEST, parse_dates=["date"]).set_index("date")["ml_meta_scale_growth"].sort_index()

    weights = {
        "expanded_strength_126_top30": _strength_weights(close, lookback=126, top_n=30),
        "expanded_strength_252_top30": _strength_weights(close, lookback=252, top_n=30),
        "expanded_strength_12_1_top30": _strength_weights(close, lookback=252, skip=21, top_n=30),
        "expanded_blended_strength_top35": _blend_strength_weights(close, top_n=35),
        "expanded_concentrated_strength_top15": _strength_weights(close, lookback=126, top_n=15, power=1.5),
        "expanded_breakout_strength_top25": _breakout_strength(close, top_n=25),
        "expanded_market_neutral_strength": _market_neutral_strength(close, lookback=126, top_n=25),
        "expanded_bear_short_weak": _bear_short_weak(close, lookback=63, top_n=25),
        "expanded_short_term_reversal": _short_term_reversal(close, top_n=25),
    }

    returns = {name: _evaluate(close, frame) for name, frame in weights.items()}
    returns_frame = pd.DataFrame(returns).sort_index()
    summary = _summary(returns_frame, base)
    combos = _combo_tests(base, returns_frame, summary)
    combo_returns = _combo_returns(base, returns_frame, summary)
    combo_holdout = _combo_summary({name: series.loc["2022-01-01":] for name, series in combo_returns.items()})
    next_year = _next_year_estimates(combo_returns)

    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    combos.to_csv(OUTPUT_DIR / "portfolio_tests.csv", index=False)
    combo_holdout.to_csv(OUTPUT_DIR / "holdout_summary.csv", index=False)
    next_year.to_csv(OUTPUT_DIR / "next_year_estimate.csv", index=False)
    returns_frame.to_csv(OUTPUT_DIR / "candidate_returns.csv", index_label="date")
    pd.DataFrame({"downloaded_symbols": sorted(close.columns)}).to_csv(OUTPUT_DIR / "downloaded_symbols.csv", index=False)
    _write_report(summary, combos, combo_holdout, next_year, len(EXPANDED_STOCKS), close.shape[1])
    print(summary.sort_values(["annual_return", "sharpe"], ascending=False).to_string(index=False))
    print("\nPortfolio tests:")
    print(combos.sort_values(["annual_return", "sharpe"], ascending=False).head(12).to_string(index=False))
    return 0


def _strength_weights(
    close: pd.DataFrame,
    lookback: int,
    top_n: int,
    skip: int = 0,
    power: float = 1.0,
) -> pd.DataFrame:
    momentum = close.shift(skip) / close.shift(skip + lookback) - 1.0
    vol = close.pct_change().rolling(63, min_periods=30).std().replace(0.0, np.nan)
    score = (momentum / vol).where(momentum > 0.0)
    return _signal_inverse_vol_weights(score, close, top_n=top_n, power=power)


def _blend_strength_weights(close: pd.DataFrame, top_n: int) -> pd.DataFrame:
    ret_63 = close / close.shift(63) - 1.0
    ret_126 = close / close.shift(126) - 1.0
    ret_252 = close / close.shift(252) - 1.0
    vol = close.pct_change().rolling(63, min_periods=30).std().replace(0.0, np.nan)
    score = (0.25 * ret_63 + 0.45 * ret_126 + 0.30 * ret_252) / vol
    return _signal_inverse_vol_weights(score.where(score > 0.0), close, top_n=top_n, power=1.25)


def _breakout_strength(close: pd.DataFrame, top_n: int) -> pd.DataFrame:
    prior_high = close.shift(1).rolling(55, min_periods=30).max()
    ret = close / close.shift(126) - 1.0
    vol = close.pct_change().rolling(63, min_periods=30).std().replace(0.0, np.nan)
    score = (ret / vol).where((close >= prior_high) & (close > close.rolling(200, min_periods=100).mean()))
    return _signal_inverse_vol_weights(score, close, top_n=top_n, power=1.0)


def _market_neutral_strength(close: pd.DataFrame, lookback: int, top_n: int) -> pd.DataFrame:
    ret = close / close.shift(lookback) - 1.0
    vol = close.pct_change().rolling(63, min_periods=30).std().replace(0.0, np.nan)
    score = ret / vol
    long = _signal_inverse_vol_weights(score, close, top_n=top_n, gross=0.5)
    short = -_signal_inverse_vol_weights(-score, close, top_n=top_n, gross=0.5)
    return (long + short).fillna(0.0)


def _bear_short_weak(close: pd.DataFrame, lookback: int, top_n: int) -> pd.DataFrame:
    spy = close["SPY"] if "SPY" in close else close.mean(axis=1)
    bear = spy < spy.rolling(200, min_periods=100).mean()
    ret = close / close.shift(lookback) - 1.0
    vol = close.pct_change().rolling(63, min_periods=30).std().replace(0.0, np.nan)
    short = -_signal_inverse_vol_weights((-ret / vol).where(ret < 0.0), close, top_n=top_n, gross=1.0)
    return short.where(bear, 0.0).fillna(0.0)


def _short_term_reversal(close: pd.DataFrame, top_n: int) -> pd.DataFrame:
    ret_5 = close / close.shift(5) - 1.0
    trend = close > close.rolling(200, min_periods=100).mean()
    score = (-ret_5).where((ret_5 < -0.02) & trend)
    return _signal_inverse_vol_weights(score, close, top_n=top_n, power=1.0)


def _signal_inverse_vol_weights(
    score: pd.DataFrame,
    close: pd.DataFrame,
    top_n: int,
    power: float = 1.0,
    gross: float = 1.0,
) -> pd.DataFrame:
    selected = score.rank(axis=1, ascending=False, method="first") <= top_n
    vol = close.pct_change().rolling(63, min_periods=30).std().replace(0.0, np.nan)
    raw_score = score.clip(lower=0.0).pow(power)
    raw = (raw_score / vol).where(selected, 0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total = raw.abs().sum(axis=1).replace(0.0, np.nan)
    return raw.div(total, axis=0).fillna(0.0) * gross


def _evaluate(close: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    returns = close.pct_change().fillna(0.0)
    weights = weights.reindex_like(returns).fillna(0.0)
    executed = weights.shift(1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    return (executed * returns).sum(axis=1) - turnover * (COST_BPS / 10_000.0)


def _summary(returns: pd.DataFrame, base: pd.Series) -> pd.DataFrame:
    rows = []
    for name, series in returns.items():
        summary = summarize_returns(series).as_dict()
        aligned = pd.concat([series, base], axis=1, join="inner").dropna()
        summary["strategy"] = name
        summary["correlation_to_ml_growth"] = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) if len(aligned) else np.nan
        rows.append(summary)
    return pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False).reset_index(drop=True)


def _combo_tests(base: pd.Series, candidates: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    return _combo_summary(_combo_returns(base, candidates, summary))


def _combo_returns(base: pd.Series, candidates: pd.DataFrame, summary: pd.DataFrame) -> dict[str, pd.Series]:
    combos = {"base_ml_meta_growth": base}
    eligible = summary[(summary["annual_return"] > 0.0) & (summary["sharpe"] > 0.2)]["strategy"].head(6)
    for name in eligible:
        candidate = candidates[name].reindex(base.index).fillna(0.0)
        for weight in [0.10, 0.20, 0.30]:
            combos[f"base_{100-int(weight*100)}_{name}_{int(weight*100)}"] = (1 - weight) * base + weight * candidate
        for overlay in [0.25, 0.50]:
            combos[f"base_overlay_{name}_{int(overlay*100)}_retarget45"] = _vol_target(base + overlay * candidate, 0.45)
    if len(eligible) > 1:
        basket = candidates[list(eligible)].reindex(base.index).fillna(0.0).mean(axis=1)
        combos["base_overlay_expanded_basket_25_retarget45"] = _vol_target(base + 0.25 * basket, 0.45)
    return combos


def _combo_summary(combos: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, returns in combos.items():
        rows.append(_portfolio_row(name, returns))
    return pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False).reset_index(drop=True)


def _vol_target(returns: pd.Series, target_vol: float) -> pd.Series:
    realized = returns.rolling(63, min_periods=20).std().shift(1) * np.sqrt(252)
    scale = (target_vol / realized).clip(lower=0.25, upper=2.5).fillna(0.75)
    return returns * scale


def _portfolio_row(name: str, returns: pd.Series) -> dict[str, float | int | str]:
    summary = summarize_returns(returns).as_dict()
    summary["portfolio"] = name
    return summary


def _next_year_estimates(combos: dict[str, pd.Series], simulations: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    rows = []
    for name, returns in combos.items():
        recent = returns.loc["2018-01-01":].dropna().to_numpy(dtype=float)
        block = 21
        sampled_returns = np.empty(simulations)
        for idx in range(simulations):
            sampled: list[float] = []
            while len(sampled) < 252:
                start = int(rng.integers(0, max(1, len(recent) - block)))
                sampled.extend(recent[start : start + block])
            sampled_returns[idx] = float(np.prod(1.0 + np.asarray(sampled[:252])) - 1.0)
        rows.append(
            {
                "portfolio": name,
                "expected_return": float(np.mean(sampled_returns)),
                "median_return": float(np.median(sampled_returns)),
                "p05_return": float(np.quantile(sampled_returns, 0.05)),
                "p95_return": float(np.quantile(sampled_returns, 0.95)),
                "loss_probability": float((sampled_returns < 0.0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("expected_return", ascending=False).reset_index(drop=True)


def _write_report(
    summary: pd.DataFrame,
    combos: pd.DataFrame,
    holdout: pd.DataFrame,
    next_year: pd.DataFrame,
    intended: int,
    downloaded: int,
) -> None:
    lines = [
        "# Expanded Stock Signal-Strength Results",
        "",
        f"Downloaded {downloaded} usable symbols from an intended expanded universe of {intended}.",
        "Signals use continuous momentum/breakout/reversal strength combined with inverse-volatility sizing.",
        "",
        "## Best standalone expanded-stock candidates",
        "",
        _markdown_table(summary.head(12)),
        "",
        "## Portfolio combination tests versus current ML growth portfolio",
        "",
        _markdown_table(combos.head(12)),
        "",
        "## 2022-2026 validation window for top combinations",
        "",
        _markdown_table(holdout.head(8)),
        "",
        "## Estimated next-year distribution",
        "",
        _markdown_table(next_year.head(8)),
        "",
        "## Interpretation",
        "",
        "- Broader stock breadth and continuous signal-strength sizing improved the return engine when used as an overlay and re-targeted to volatility.",
        "- The improvement is not a clean diversification win: the best expanded-stock strength sleeves remain correlated to the current ML growth engine.",
        "- The 2022-2026 window is a validation window after repeated research iterations, not an untouched final holdout.",
        "- Market-neutral, bear-short, and reversal variants did not beat the return engine after costs.",
        "- The most useful result is better breadth/sizing for the momentum complex; genuinely unrelated variants were too weak after costs.",
    ]
    (OUTPUT_DIR / "EXPANDED_STOCK_SIGNAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "strategy",
        "portfolio",
        "annual_return",
        "annual_std",
        "sharpe",
        "max_drawdown",
        "correlation_to_ml_growth",
        "total_return",
        "expected_return",
        "median_return",
        "p05_return",
        "p95_return",
        "loss_probability",
    ]
    display = frame[[column for column in columns if column in frame.columns]].copy()
    for column in [
        "annual_return",
        "annual_std",
        "max_drawdown",
        "total_return",
        "expected_return",
        "median_return",
        "p05_return",
        "p95_return",
        "loss_probability",
    ]:
        if column in display:
            display[column] = display[column].map(lambda value: f"{float(value) * 100:.2f}%")
    for column in ["sharpe", "correlation_to_ml_growth"]:
        if column in display:
            display[column] = display[column].map(lambda value: f"{float(value):.2f}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())

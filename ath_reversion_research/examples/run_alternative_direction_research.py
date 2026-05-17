"""Alternative strategy directions beyond equity momentum.

This script searches for sleeves that are structurally different from the
current equity-strength stack: cross-asset ETF trend/reversal, crisis hedges,
sector mean reversion, and volatility ETF regime strategies.  Candidate
selection is performed on pre-2022 data only; 2022 onward is validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ath_reversion_research import download_yahoo_ohlcv
from ath_reversion_research.metrics import summarize_returns


OUTPUT_DIR = Path("ath_reversion_research/reports/alternative_direction")
VALIDATED_RETURNS = Path("ath_reversion_research/reports/validated_improvement/validation_metrics.csv")
SHARPE_RETURNS = Path("ath_reversion_research/reports/sharpe_improvement/returns.csv")
COST_BPS = 10.0
TRAIN_END = "2021-12-31"
VALIDATION_START = "2022-01-01"

ETF_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "GLD",
    "SLV",
    "DBC",
    "USO",
    "UNG",
    "UUP",
    "FXE",
    "FXY",
    "VNQ",
    "XLP",
    "XLU",
    "XLV",
    "XLE",
    "XLF",
    "XLK",
    "XLI",
    "XLY",
    "XLB",
    "XLRE",
    "XLC",
    "SH",
    "PSQ",
    "SDS",
    "VIXY",
    "SVXY",
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bars = download_yahoo_ohlcv(ETF_UNIVERSE, start="2010-01-01", chunk_size=12)
    close = bars.pivot(index="date", columns="symbol", values="close").sort_index()
    close = close.dropna(axis=1, thresh=int(len(close) * 0.35))

    candidates = _candidate_returns(close)
    summary = _summary(candidates)
    selected = _select_pre2022(summary)
    validation = _validation(candidates, selected)
    combos = _combo_tests(candidates, selected)

    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "pre2022_selected.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    combos.to_csv(OUTPUT_DIR / "combo_tests.csv", index=False)
    pd.DataFrame(candidates).to_csv(OUTPUT_DIR / "candidate_returns.csv", index_label="date")
    pd.DataFrame({"downloaded_symbols": sorted(close.columns)}).to_csv(OUTPUT_DIR / "downloaded_symbols.csv", index=False)
    _write_report(summary, selected, validation, combos, len(ETF_UNIVERSE), close.shape[1])
    print(summary.sort_values(["train_objective"], ascending=False).head(15).to_string(index=False))
    print("\nValidation:")
    print(validation.to_string(index=False))
    return 0


def _candidate_returns(close: pd.DataFrame) -> dict[str, pd.Series]:
    weights = {
        "xasset_abs_mom_252_top3": _asset_abs_momentum(close, lookback=252, top_n=3),
        "xasset_abs_mom_126_top2": _asset_abs_momentum(close, lookback=126, top_n=2),
        "xasset_carry_proxy_defensive": _defensive_carry_proxy(close),
        "sector_mean_reversion_pairs": _sector_mean_reversion(close),
        "sector_market_neutral_trend": _sector_market_neutral_trend(close),
        "risk_off_inverse_equity": _risk_off_inverse_equity(close),
        "vol_regime_svxy_vixy": _vol_regime(close),
        "etf_short_term_reversal": _etf_short_reversal(close),
        "credit_stress_rotation": _credit_stress_rotation(close),
    }
    return {name: _evaluate(close, frame) for name, frame in weights.items()}


def _asset_abs_momentum(close: pd.DataFrame, lookback: int, top_n: int) -> pd.DataFrame:
    symbols = [s for s in ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "UUP", "VNQ"] if s in close]
    score = close[symbols] / close[symbols].shift(lookback) - 1.0
    return _inverse_vol_weights(score.where(score > 0.0), close[symbols], top_n=top_n)


def _defensive_carry_proxy(close: pd.DataFrame) -> pd.DataFrame:
    # In risk-off regimes, rotate among bonds, gold, dollar, and defensive sectors.
    spy = close["SPY"]
    risk_off = spy < spy.rolling(200, min_periods=100).mean()
    symbols = [s for s in ["TLT", "IEF", "GLD", "UUP", "XLP", "XLU", "XLV"] if s in close]
    score = close[symbols] / close[symbols].shift(126) - 1.0
    weights = _inverse_vol_weights(score.where(score > 0.0), close[symbols], top_n=2)
    return _embed(weights.where(risk_off.shift(1), 0.0), close.columns)


def _sector_mean_reversion(close: pd.DataFrame) -> pd.DataFrame:
    sectors = [s for s in ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"] if s in close]
    rel = close[sectors].div(close[sectors].mean(axis=1), axis=0)
    z = (rel - rel.rolling(63, min_periods=30).mean()) / rel.rolling(63, min_periods=30).std()
    long = _rank_weights((-z).where(z < -1.0), top_n=3, gross=0.5)
    short = -_rank_weights(z.where(z > 1.0), top_n=3, gross=0.5)
    return _embed((long + short).fillna(0.0), close.columns)


def _sector_market_neutral_trend(close: pd.DataFrame) -> pd.DataFrame:
    sectors = [s for s in ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"] if s in close]
    score = close[sectors] / close[sectors].shift(126) - 1.0
    long = _rank_weights(score, top_n=3, gross=0.5)
    short = -_rank_weights(-score, top_n=3, gross=0.5)
    return _embed((long + short).fillna(0.0), close.columns)


def _risk_off_inverse_equity(close: pd.DataFrame) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    spy = close["SPY"]
    risk_off = (spy < spy.rolling(200, min_periods=100).mean()) & (spy.pct_change(63) < 0.0)
    for symbol, weight in [("SH", 0.55), ("PSQ", 0.30), ("TLT", 0.15)]:
        if symbol in weights:
            weights.loc[risk_off.shift(1).fillna(False), symbol] = weight
    return weights


def _vol_regime(close: pd.DataFrame) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if "SVXY" not in close or "VIXY" not in close:
        return weights
    spy = close["SPY"]
    trend = spy > spy.rolling(200, min_periods=100).mean()
    realized = spy.pct_change().rolling(21, min_periods=10).std()
    calm = realized < realized.rolling(252, min_periods=100).quantile(0.55)
    stress = realized > realized.rolling(252, min_periods=100).quantile(0.85)
    weights.loc[(trend & calm).shift(1).fillna(False), "SVXY"] = 1.0
    weights.loc[(~trend & stress).shift(1).fillna(False), "VIXY"] = 0.50
    return weights


def _etf_short_reversal(close: pd.DataFrame) -> pd.DataFrame:
    symbols = [s for s in ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "XLF", "XLK", "XLE"] if s in close]
    ret = close[symbols] / close[symbols].shift(5) - 1.0
    trend = close[symbols] > close[symbols].rolling(200, min_periods=100).mean()
    score = (-ret).where((ret < -0.02) & trend)
    return _embed(_inverse_vol_weights(score, close[symbols], top_n=3), close.columns)


def _credit_stress_rotation(close: pd.DataFrame) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if "HYG" not in close or "LQD" not in close:
        return weights
    credit = close["HYG"] / close["LQD"]
    stress = credit < credit.rolling(126, min_periods=60).mean()
    if "TLT" in weights:
        weights.loc[stress.shift(1).fillna(False), "TLT"] = 0.65
    if "GLD" in weights:
        weights.loc[stress.shift(1).fillna(False), "GLD"] = 0.35
    if "HYG" in weights:
        weights.loc[(~stress).shift(1).fillna(False), "HYG"] = 1.0
    return weights


def _inverse_vol_weights(score: pd.DataFrame, close: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected = score.rank(axis=1, ascending=False, method="first") <= top_n
    vol = close.pct_change().rolling(63, min_periods=30).std().replace(0.0, np.nan)
    raw = score.clip(lower=0.0).where(selected, 0.0) / vol
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return raw.div(raw.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)


def _rank_weights(score: pd.DataFrame, top_n: int, gross: float) -> pd.DataFrame:
    selected = score.rank(axis=1, ascending=False, method="first") <= top_n
    raw = score.abs().where(selected, 0.0).fillna(0.0)
    return raw.div(raw.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0) * gross


def _embed(weights: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=weights.index, columns=columns)
    out.loc[:, weights.columns] = weights
    return out


def _evaluate(close: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    returns = close.pct_change().fillna(0.0)
    executed = weights.reindex_like(returns).fillna(0.0).shift(1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    return (executed * returns).sum(axis=1) - turnover * (COST_BPS / 10_000.0)


def _summary(candidates: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, returns in candidates.items():
        train = summarize_returns(returns.loc[:TRAIN_END]).as_dict()
        validation = summarize_returns(returns.loc[VALIDATION_START:]).as_dict()
        full = summarize_returns(returns).as_dict()
        rows.append(
            {
                "strategy": name,
                "train_annual_return": train["annual_return"],
                "train_sharpe": train["sharpe"],
                "train_max_drawdown": train["max_drawdown"],
                "validation_annual_return": validation["annual_return"],
                "validation_sharpe": validation["sharpe"],
                "validation_max_drawdown": validation["max_drawdown"],
                "full_annual_return": full["annual_return"],
                "full_sharpe": full["sharpe"],
                "full_max_drawdown": full["max_drawdown"],
                "train_objective": train["sharpe"] + 0.2 * train["annual_return"] + train["max_drawdown"],
            }
        )
    return pd.DataFrame(rows).sort_values("train_objective", ascending=False).reset_index(drop=True)


def _select_pre2022(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.head(3).copy()


def _validation(candidates: dict[str, pd.Series], selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in selected["strategy"]:
        returns = candidates[name]
        validation = summarize_returns(returns.loc[VALIDATION_START:]).as_dict()
        row = {"strategy": name}
        row.update({f"validation_{key}": value for key, value in validation.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _combo_tests(candidates: dict[str, pd.Series], selected: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv("ath_reversion_research/reports/sharpe_improvement/returns.csv", parse_dates=["date"]).set_index("date")[
        "asset_overlay_regime_target35"
    ]
    rows = []
    for name in selected["strategy"]:
        candidate = candidates[name].reindex(base.index).fillna(0.0)
        for weight in [0.10, 0.20, 0.30]:
            combo = (1.0 - weight) * base + weight * candidate
            full = summarize_returns(combo).as_dict()
            validation = summarize_returns(combo.loc[VALIDATION_START:]).as_dict()
            rows.append(
                {
                    "combo": f"asset_overlay_regime_{int((1-weight)*100)}_{name}_{int(weight*100)}",
                    "full_annual_return": full["annual_return"],
                    "full_sharpe": full["sharpe"],
                    "full_max_drawdown": full["max_drawdown"],
                    "validation_annual_return": validation["annual_return"],
                    "validation_sharpe": validation["sharpe"],
                    "validation_max_drawdown": validation["max_drawdown"],
                }
            )
    return pd.DataFrame(rows).sort_values(["validation_sharpe", "validation_annual_return"], ascending=False).reset_index(drop=True)


def _write_report(summary: pd.DataFrame, selected: pd.DataFrame, validation: pd.DataFrame, combos: pd.DataFrame, intended: int, downloaded: int) -> None:
    lines = [
        "# Alternative Direction Results",
        "",
        f"Downloaded {downloaded} usable ETFs from an intended alternative universe of {intended}.",
        "Strategy families include cross-asset trend, defensive credit/stress rotation, sector mean reversion, inverse equity, and volatility ETF regimes.",
        "Candidates and combinations are selected using pre-2022 data only; 2022-2026 is validation.",
        "",
        "## Top pre-2022 alternative candidates",
        "",
        _markdown_table(summary.head(10)),
        "",
        "## 2022-2026 validation for selected candidates",
        "",
        _markdown_table(validation),
        "",
        "## Combination tests with current return/Sharpe candidate",
        "",
        _markdown_table(combos.head(10)),
        "",
        "## Conclusion",
        "",
        "- This different direction did not produce a massive validated improvement.",
        "- Cross-asset and defensive sleeves can reduce equity specificity, but validation Sharpe was not better than the current fixed candidates.",
        "- The useful takeaway is to keep asset-class absolute momentum as a modest overlay, not to replace the current return engine.",
    ]
    (OUTPUT_DIR / "ALTERNATIVE_DIRECTION_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if any(token in column for token in ["return", "drawdown"]):
            display[column] = display[column].map(lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
        elif "sharpe" in column or "objective" in column:
            display[column] = display[column].map(lambda value: f"{float(value):.2f}" if pd.notna(value) else "")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())

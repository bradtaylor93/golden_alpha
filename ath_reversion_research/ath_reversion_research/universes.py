"""Universe definitions for ATH dip and exponential reversion research.

The symbols are intentionally broad and explicit: liquid ETFs, sector samples,
large-cap operating companies, and lower market-cap candidates.  Market-cap
membership changes through time, so production use should replace the static
lists with point-in-time constituents and market caps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Universe:
    """A named collection of symbols with a research purpose."""

    name: str
    symbols: tuple[str, ...]
    description: str


HIGH_MARKET_CAP_UNIVERSE = Universe(
    name="high_market_cap_cross_asset",
    symbols=(
        # Broad and factor ETFs.
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "EFA",
        "EEM",
        "TLT",
        "IEF",
        "GLD",
        "SLV",
        "USO",
        "VNQ",
        # Sector ETFs.
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
        # Real-name mega and large caps across sectors.
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
        "BRK-B",
        "LLY",
        "JPM",
        "V",
        "UNH",
        "XOM",
        "COST",
        "HD",
        "PG",
        "MA",
        "JNJ",
        "AVGO",
        "MRK",
        "ABBV",
        "PEP",
        "KO",
        "WMT",
        "BAC",
        "CRM",
    ),
    description=(
        "Liquid ETF, sector ETF, and high market-cap single-name universe for "
        "capacity-aware tests."
    ),
)


LOWER_MARKET_CAP_UNIVERSE = Universe(
    name="lower_market_cap_under_10bn_sample",
    symbols=(
        "FIVE",
        "WING",
        "CROX",
        "CELH",
        "LTH",
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
        "ZI",
        "RDFN",
        "OPEN",
        "BL",
        "ASAN",
    ),
    description=(
        "Static lower market-cap sample. Validate current and historical market "
        "caps before using results for investment conclusions."
    ),
)


DEFAULT_UNIVERSES = {
    HIGH_MARKET_CAP_UNIVERSE.name: HIGH_MARKET_CAP_UNIVERSE,
    LOWER_MARKET_CAP_UNIVERSE.name: LOWER_MARKET_CAP_UNIVERSE,
}


def symbols_for(names: list[str] | None = None) -> list[str]:
    """Return de-duplicated symbols for selected universe names."""

    selected = names or list(DEFAULT_UNIVERSES)
    out: list[str] = []
    seen: set[str] = set()
    for name in selected:
        universe = DEFAULT_UNIVERSES[name]
        for symbol in universe.symbols:
            if symbol not in seen:
                out.append(symbol)
                seen.add(symbol)
    return out

"""Yahoo Finance market data vendor integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class YahooVendorConfig:
    auto_adjust: bool = True
    prepost: bool = False
    threads: bool = True


class YahooMarketDataVendor:
    """Fetch OHLCV bars from Yahoo Finance and normalize schema."""

    def __init__(self, config: YahooVendorConfig | None = None) -> None:
        self.config = config or YahooVendorConfig()

    def fetch_bars(
        self,
        assets: list[str],
        *,
        interval: str = "1h",
        period: str = "2y",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch bars for assets and return canonical schema.

        Notes:
        - Yahoo intraday retention is limited; for `1h`, `2y` is best-effort.
        - If `start/end` are provided they are used instead of `period`.
        """
        if not assets:
            return pd.DataFrame(columns=["timestamp", "asset", "open", "high", "low", "close", "volume"])

        rows: list[pd.DataFrame] = []
        for asset in assets:
            ticker = yf.Ticker(asset)
            hist = ticker.history(
                period=period if start is None and end is None else None,
                interval=interval,
                start=start,
                end=end,
                auto_adjust=self.config.auto_adjust,
                prepost=self.config.prepost,
            )
            if hist is None or hist.empty:
                continue
            frame = hist.reset_index().rename(
                columns={
                    "Datetime": "timestamp",
                    "Date": "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame["asset"] = asset
            keep = ["timestamp", "asset", "open", "high", "low", "close", "volume"]
            frame = frame[keep].copy()
            frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0).astype(int)
            rows.append(frame)

        if not rows:
            return pd.DataFrame(columns=["timestamp", "asset", "open", "high", "low", "close", "volume"])
        out = pd.concat(rows, ignore_index=True, sort=False)
        return out.sort_values(["asset", "timestamp"]).reset_index(drop=True)

    @staticmethod
    def default_two_year_hourly_window() -> tuple[str, str]:
        """Return a practical two-year window for explicit start/end usage."""
        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(days=730)
        return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


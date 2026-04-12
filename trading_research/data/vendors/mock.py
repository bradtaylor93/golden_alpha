"""Mock vendor for deterministic synthetic bar retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MockVendorConfig:
    seed: int = 7
    start: str = "2020-01-01"
    periods: int = 400
    freq: str = "D"


class MockMarketDataVendor:
    """Simple synthetic OHLCV generator for examples and tests."""

    def __init__(self, config: MockVendorConfig | None = None) -> None:
        self.config = config or MockVendorConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def fetch_bars(self, assets: list[str]) -> pd.DataFrame:
        index = pd.date_range(
            self.config.start, periods=self.config.periods, freq=self.config.freq, tz="UTC"
        )
        rows: list[dict[str, object]] = []
        for asset in assets:
            drift = self._rng.normal(loc=0.0003, scale=0.0002, size=len(index))
            noise = self._rng.normal(loc=0.0, scale=0.01, size=len(index))
            rets = drift + noise
            close = 100.0 * np.cumprod(1.0 + rets)
            open_px = np.roll(close, 1)
            open_px[0] = close[0]
            high = np.maximum(open_px, close) * (1.0 + np.abs(self._rng.normal(0, 0.003, len(index))))
            low = np.minimum(open_px, close) * (1.0 - np.abs(self._rng.normal(0, 0.003, len(index))))
            vol = self._rng.integers(1_000, 100_000, size=len(index))
            for ts, o, h, l, c, v in zip(index, open_px, high, low, close, vol):
                rows.append(
                    {
                        "timestamp": ts,
                        "asset": asset,
                        "open": float(o),
                        "high": float(h),
                        "low": float(l),
                        "close": float(c),
                        "volume": int(v),
                    }
                )
        return pd.DataFrame(rows)

    def fetch_and_persist(
        self,
        *,
        catalog: "DataCatalog",
        dataset_name: str,
        assets: list[str],
        metadata: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        """Fetch bars and persist into the provided catalog."""
        from trading_research.data.catalog import DataCatalog

        if not isinstance(catalog, DataCatalog):
            raise TypeError("catalog must be a DataCatalog instance")
        bars = self.fetch_bars(assets)
        catalog.persist_processed_bars(
            dataset_name,
            bars,
            metadata=metadata or {"source": "mock", "assets": assets},
        )
        return bars

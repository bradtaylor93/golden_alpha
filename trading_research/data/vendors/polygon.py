"""Polygon.io market data vendor integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests


@dataclass(frozen=True)
class PolygonVendorConfig:
    api_key: str
    base_url: str = "https://api.polygon.io"
    adjusted: bool = True
    sort: str = "asc"
    limit: int = 50000
    timeout_seconds: int = 30


class PolygonMarketDataVendor:
    """Pulls OHLCV bars from Polygon aggregates endpoint."""

    def __init__(self, config: PolygonVendorConfig) -> None:
        self.config = config
        self._session = requests.Session()

    def build_request(
        self,
        asset: str,
        *,
        multiplier: int,
        timespan: str,
        start: str,
        end: str,
    ) -> tuple[str, dict[str, Any]]:
        """Public request builder for debugging/tests."""
        return self._build_request(
            asset=asset,
            multiplier=multiplier,
            timespan=timespan,
            start=start,
            end=end,
        )

    def _build_request(
        self,
        asset: str,
        *,
        multiplier: int,
        timespan: str,
        start: str,
        end: str,
    ) -> tuple[str, dict[str, str | int]]:
        endpoint = (
            f"{self.config.base_url}/v2/aggs/ticker/{asset}/range/"
            f"{multiplier}/{timespan}/{start}/{end}"
        )
        params: dict[str, str | int] = {
            "adjusted": str(self.config.adjusted).lower(),
            "sort": self.config.sort,
            "limit": self.config.limit,
            "apiKey": self.config.api_key,
        }
        return endpoint, params

    def _request(
        self,
        asset: str,
        *,
        multiplier: int,
        timespan: str,
        start: str,
        end: str,
    ) -> dict[str, Any]:
        endpoint, params = self._build_request(
            asset,
            multiplier=multiplier,
            timespan=timespan,
            start=start,
            end=end,
        )
        resp = self._session.get(endpoint, params=params, timeout=self.config.timeout_seconds)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") not in {"OK", "DELAYED"}:
            raise RuntimeError(f"Polygon returned unexpected status for {asset}: {payload}")
        return payload

    def fetch_bars(
        self,
        assets: list[str],
        *,
        start: str,
        end: str | None = None,
        multiplier: int = 1,
        timespan: str = "day",
    ) -> pd.DataFrame:
        """Fetch bars for assets and return canonical schema."""
        end_date = end or datetime.now(UTC).strftime("%Y-%m-%d")
        rows: list[dict[str, Any]] = []
        for asset in assets:
            payload = self._request(
                asset=asset,
                multiplier=multiplier,
                timespan=timespan,
                start=start,
                end=end_date,
            )
            for bar in payload.get("results", []):
                ts = pd.to_datetime(int(bar["t"]), unit="ms", utc=True)
                rows.append(
                    {
                        "timestamp": ts,
                        "asset": asset,
                        "open": float(bar.get("o", 0.0)),
                        "high": float(bar.get("h", 0.0)),
                        "low": float(bar.get("l", 0.0)),
                        "close": float(bar.get("c", 0.0)),
                        "volume": int(bar.get("v", 0)),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["timestamp", "asset", "open", "high", "low", "close", "volume"])
        out = pd.DataFrame(rows).sort_values(["asset", "timestamp"]).reset_index(drop=True)
        return out

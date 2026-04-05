"""Example 8: ingest bars from Polygon into the local catalog.

Usage modes:
- With `POLYGON_API_KEY` set: fetch and persist data.
- Without key: print a request preview for quick integration checks.
"""

from __future__ import annotations

import os
from pathlib import Path

from trading_research.data.catalog import DataCatalog
from trading_research.data.vendors import PolygonMarketDataVendor, PolygonVendorConfig


def main() -> None:
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        preview_vendor = PolygonMarketDataVendor(PolygonVendorConfig(api_key="demo"))
        endpoint, params = preview_vendor.build_request(
            "AAPL",
            multiplier=1,
            timespan="day",
            start="2022-01-01",
            end="2022-12-31",
        )
        print("POLYGON_API_KEY not set; request preview mode")
        print("endpoint:", endpoint)
        print("params:", params)
        return

    root = Path("trading_research/examples/_output/08_polygon_ingest")
    catalog = DataCatalog(root / "data")
    vendor = PolygonMarketDataVendor(PolygonVendorConfig(api_key=api_key))

    bars = vendor.fetch_bars(
        assets=["AAPL", "MSFT"],
        start="2022-01-01",
        end="2022-12-31",
        timespan="day",
    )
    catalog.persist_processed_bars(
        "polygon_us_large_cap",
        bars,
        metadata={"source": "polygon", "assets": ["AAPL", "MSFT"], "timespan": "day"},
    )
    print("saved rows:", len(bars))


if __name__ == "__main__":
    main()

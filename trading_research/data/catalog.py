"""Local data catalog for raw and processed datasets."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd

from trading_research.data.schemas import normalize_bars_schema
from trading_research.utils.hashing import stable_hash
from trading_research.utils.io import read_dataframe, write_dataframe
from trading_research.utils.paths import ensure_dir


class DataCatalog:
    """Catalog to persist and load normalized market data artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.raw_dir = ensure_dir(self.root / "raw")
        self.processed_dir = ensure_dir(self.root / "processed")
        self.metadata_path = self.root / "catalog_metadata.parquet"
        if not self.metadata_path.exists():
            write_dataframe(
                self.metadata_path,
                pd.DataFrame(
                    columns=[
                        "dataset_name",
                        "storage_tier",
                        "version_hash",
                        "path",
                        "rows",
                        "metadata",
                    ]
                ),
            )

    def _register(
        self,
        dataset_name: str,
        storage_tier: str,
        path: Path,
        df: pd.DataFrame,
        metadata: dict[str, Any] | None,
    ) -> None:
        meta_df = read_dataframe(self.metadata_path)
        if "metadata" in meta_df.columns:
            meta_df["metadata"] = meta_df["metadata"].map(
                lambda v: v if isinstance(v, str) else json.dumps(v if v is not None else {}, sort_keys=True)
            )
        version_hash = stable_hash(
            {
                "dataset_name": dataset_name,
                "storage_tier": storage_tier,
                "rows": len(df),
                "metadata": metadata or {},
            }
        )
        row = pd.DataFrame(
            [
                {
                    "dataset_name": dataset_name,
                    "storage_tier": storage_tier,
                    "version_hash": version_hash,
                    "path": str(path),
                    "rows": len(df),
                    "metadata": json.dumps(metadata or {}, sort_keys=True),
                }
            ]
        )
        row["metadata"] = row["metadata"].astype(str)
        write_dataframe(self.metadata_path, pd.concat([meta_df, row], ignore_index=True))

    def persist_raw_bars(
        self,
        dataset_name: str,
        bars: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        path = self.raw_dir / f"{dataset_name}.parquet"
        write_dataframe(path, bars)
        self._register(dataset_name, "raw", path, bars, metadata)
        return path

    def persist_processed_bars(
        self,
        dataset_name: str,
        bars: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        normalized = normalize_bars_schema(bars)
        path = self.processed_dir / f"{dataset_name}.parquet"
        write_dataframe(path, normalized)
        self._register(dataset_name, "processed", path, normalized, metadata)
        return path

    def persist_bars(
        self,
        dataset_name: str,
        bars: pd.DataFrame,
        *,
        raw: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Compatibility helper: persist bars to raw/processed tier."""
        if raw:
            return self.persist_raw_bars(dataset_name, bars, metadata=metadata)
        return self.persist_processed_bars(dataset_name, bars, metadata=metadata)

    def load_bars(
        self,
        dataset_name: str,
        raw: bool = False,
        storage_tier: str | None = None,
    ) -> pd.DataFrame:
        """Load bars from processed/raw storage.

        `storage_tier` is accepted for compatibility with older call sites.
        """
        if storage_tier is not None:
            raw = storage_tier == "raw"
        path = (self.raw_dir if raw else self.processed_dir) / f"{dataset_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        return read_dataframe(path)

    def export_csv(self, dataset_name: str, raw: bool = False) -> Path:
        source = (self.raw_dir if raw else self.processed_dir) / f"{dataset_name}.parquet"
        if not source.exists():
            raise FileNotFoundError(source)
        out = source.with_suffix(".csv")
        read_dataframe(source).to_csv(out, index=False)
        return out

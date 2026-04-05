"""Feature family abstraction and registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from trading_research.utils.hashing import stable_hash
from trading_research.utils.io import read_table, write_table


class FeatureFamily(Protocol):
    """Interface every feature family must expose."""

    name: str
    version: str

    def build(self, bars: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
        """Build a feature table with ts/asset keys and feature columns."""


@dataclass(frozen=True)
class FeatureSpec:
    family_name: str
    params: dict[str, object]


class FeatureRegistry:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self._families: dict[str, FeatureFamily] = {}
        self._cache_dir = cache_dir
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def register(self, family: FeatureFamily) -> None:
        self._families[family.name] = family

    def get(self, name: str) -> FeatureFamily:
        if name not in self._families:
            raise KeyError(f"Unknown feature family: {name}")
        return self._families[name]

    def _cache_path(self, family: FeatureFamily, params: dict[str, object]) -> Path | None:
        if self._cache_dir is None:
            return None
        key = stable_hash({"family": family.name, "version": family.version, "params": params})
        return self._cache_dir / f"{family.name}_{key}.parquet"

    def build(self, bars: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
        family = self.get(spec.family_name)
        cache_path = self._cache_path(family, spec.params)
        if cache_path is not None and cache_path.exists():
            return read_table(cache_path)
        features = family.build(bars, spec.params)
        if cache_path is not None:
            write_table(features, cache_path)
        return features


def default_feature_registry(cache_dir: Path | None = None) -> FeatureRegistry:
    from trading_research.features.baseline import BaselineFeatureFamily
    from trading_research.features.cross_asset import CrossAssetFeatureFamily
    from trading_research.features.field_adapter import FieldAdapterFeatureFamily
    from trading_research.features.regime import RegimeFeatureFamily

    reg = FeatureRegistry(cache_dir=cache_dir)
    reg.register(BaselineFeatureFamily())
    reg.register(FieldAdapterFeatureFamily())
    reg.register(CrossAssetFeatureFamily())
    reg.register(RegimeFeatureFamily())
    return reg

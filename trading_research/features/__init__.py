"""Feature system exports."""

from trading_research.features.baseline import BaselineFeatureFamily
from trading_research.features.cross_asset import CrossAssetFeatureFamily
from trading_research.features.field_adapter import FieldAdapterFeatureFamily
from trading_research.features.regime import RegimeFeatureFamily
from trading_research.features.research_pack import ResearchFeaturePackFamily
from trading_research.features.registry import (
    FeatureFamily,
    FeatureRegistry,
    FeatureSpec,
    default_feature_registry,
)

__all__ = [
    "FeatureFamily",
    "FeatureRegistry",
    "FeatureSpec",
    "default_feature_registry",
    "BaselineFeatureFamily",
    "FieldAdapterFeatureFamily",
    "CrossAssetFeatureFamily",
    "RegimeFeatureFamily",
    "ResearchFeaturePackFamily",
]

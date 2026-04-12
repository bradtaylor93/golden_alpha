"""Recipe layer public exports."""

from trading_research.recipes.base import Recipe
from trading_research.recipes.champion_edge import ChampionEdgeRecipe
from trading_research.recipes.comparison import FeatureComparisonRecipe
from trading_research.recipes.diagnostics_meta import DiagnosticAwareMetaRecipe
from trading_research.recipes.horizon_error_meta import HorizonErrorMetaRecipe
from trading_research.recipes.nested_tuning import NestedTuningRecipe
from trading_research.recipes.regime_aware_moe import RegimeAwareMoERecipe
from trading_research.recipes.selector import SelectorRecipe
from trading_research.recipes.single_model import SingleModelRecipe
from trading_research.recipes.stacking import StackedModelRecipe

__all__ = [
    "Recipe",
    "ChampionEdgeRecipe",
    "RegimeAwareMoERecipe",
    "SingleModelRecipe",
    "FeatureComparisonRecipe",
    "NestedTuningRecipe",
    "StackedModelRecipe",
    "DiagnosticAwareMetaRecipe",
    "SelectorRecipe",
    "HorizonErrorMetaRecipe",
]

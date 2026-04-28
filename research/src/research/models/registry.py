"""Model zoo registry. Every entry produces:
  * an unfitted estimator (constructor function)
  * an Optuna search-space helper

The training driver picks one or more candidates by name and runs
each through the same outer CPCV + inner Optuna loop, then ranks
them by OOS log loss. The winner becomes the deployed champion.

Adding a new candidate: define `<name>_classifier` and
`<name>_search_space` in a sibling module, then append a `ModelSpec`
here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from research.models.lgbm import lgbm_classifier, lgbm_search_space
from research.models.xgb import xgb_classifier, xgb_search_space
from research.models.catboost_model import (
    catboost_classifier,
    catboost_search_space,
)
from research.models.sklearn_zoo import (
    extratrees_classifier,
    extratrees_search_space,
    histgb_classifier,
    histgb_search_space,
    logreg_classifier,
    logreg_search_space,
    mlp_classifier,
    mlp_search_space,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    """Stable identifier — appears in `model_id` (e.g. "side_lgbm_…").
       Lowercase, no spaces; safe for filenames + DuckDB IDs."""
    constructor: Callable[..., Any]
    """`(params: dict | None) -> sklearn-compatible estimator`."""
    search_space: Callable[[Any], dict[str, Any]]
    """`(optuna.Trial) -> dict` of suggested params."""
    description: str
    """Human-readable label for the dashboard."""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "lgbm": ModelSpec(
        name="lgbm",
        constructor=lgbm_classifier,
        search_space=lgbm_search_space,
        description="LightGBM (gradient-boosted trees)",
    ),
    "xgb": ModelSpec(
        name="xgb",
        constructor=xgb_classifier,
        search_space=xgb_search_space,
        description="XGBoost (gradient-boosted trees, hist)",
    ),
    "catboost": ModelSpec(
        name="catboost",
        constructor=catboost_classifier,
        search_space=catboost_search_space,
        description="CatBoost (gradient-boosted, ordered)",
    ),
    "logreg": ModelSpec(
        name="logreg",
        constructor=logreg_classifier,
        search_space=logreg_search_space,
        description="Logistic regression (L2)",
    ),
    "extratrees": ModelSpec(
        name="extratrees",
        constructor=extratrees_classifier,
        search_space=extratrees_search_space,
        description="ExtraTrees (random tree ensemble)",
    ),
    "mlp": ModelSpec(
        name="mlp",
        constructor=mlp_classifier,
        search_space=mlp_search_space,
        description="Multi-layer perceptron (sklearn MLP)",
    ),
    "histgb": ModelSpec(
        name="histgb",
        constructor=histgb_classifier,
        search_space=histgb_search_space,
        description="Histogram gradient boosting (sklearn)",
    ),
}

# Default zoo — everything that's safe to run unattended on the
# 1000-bar / 24-feature window. Order doesn't matter; the orchestrator
# evaluates them in parallel and picks the OOS winner.
DEFAULT_CANDIDATES: list[str] = [
    "lgbm",
    "xgb",
    "catboost",
    "logreg",
    "extratrees",
    "mlp",
    "histgb",
]


def get_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]


def parse_candidates(spec: str | None) -> list[str]:
    """Parse a CSV string ("lgbm,xgb,logreg") or `None` (default zoo)
    into a list of candidate names. Validates each."""
    if not spec:
        return list(DEFAULT_CANDIDATES)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    for n in names:
        if n not in MODEL_REGISTRY:
            raise ValueError(
                f"unknown model {n!r}; available: {sorted(MODEL_REGISTRY)}"
            )
    return names

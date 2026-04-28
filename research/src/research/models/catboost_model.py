"""CatBoost classifier factory + Optuna search space.

Distinct from `lgbm.py` mostly in that CatBoost handles its own
calibration internally (`auto_class_weights="Balanced"`) and is
gentler on overfitting at small N — the small-data sweet spot for
this strategy. Same shape as the other zoo specs so the registry can
swap interchangeably.
"""
from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier


def catboost_classifier(params: dict[str, Any] | None = None) -> CatBoostClassifier:
    base: dict[str, Any] = {
        "loss_function": "Logloss",
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "rsm": 0.85,  # CatBoost's "feature_fraction" equivalent
        "subsample": 0.85,
        "bootstrap_type": "Bernoulli",
        "random_seed": 0,
        "thread_count": 1,
        "verbose": 0,
        "allow_writing_files": False,
    }
    if params:
        base.update(params)
    return CatBoostClassifier(**base)


def catboost_search_space(trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "rsm": trial.suggest_float("rsm", 0.6, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }

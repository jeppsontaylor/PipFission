"""XGBoost classifier factory + Optuna search space.

Mirrors `lgbm.py` so the zoo registry can swap them interchangeably.
Sized for the 1000-bar / 24-feature window — conservative depths,
`tree_method="hist"` for the speed bump and predictable behaviour
under small data, and a single thread (the orchestrator runs many
trials in parallel; we don't want each trial gobbling cores).
"""
from __future__ import annotations

from typing import Any

import xgboost as xgb


def xgb_classifier(params: dict[str, Any] | None = None) -> xgb.XGBClassifier:
    base: dict[str, Any] = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "tree_method": "hist",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "n_jobs": 1,
        "verbosity": 0,
    }
    if params:
        base.update(params)
    return xgb.XGBClassifier(**base)


def xgb_search_space(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
    }

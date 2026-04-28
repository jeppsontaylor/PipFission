"""LightGBM classifier factory + Optuna search-space helpers."""
from __future__ import annotations

from typing import Any

import lightgbm as lgb


def lgbm_classifier(params: dict[str, Any] | None = None) -> lgb.LGBMClassifier:
    """Build a `LGBMClassifier` with safe defaults for small-data
    binary tabular classification (1k bars, 24 features). Sized
    conservatively to avoid overfitting in inner CV."""
    base: dict[str, Any] = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "verbose": -1,
        "n_jobs": 1,
    }
    if params:
        base.update(params)
    return lgb.LGBMClassifier(**base)


def lgbm_search_space(trial) -> dict[str, Any]:
    """Optuna `Trial` -> parameter dict. Conservative ranges for the
    1000-bar window to prevent overfitting."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 60),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-6, 1.0, log=True),
    }

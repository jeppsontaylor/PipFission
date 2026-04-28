"""Lightweight non-tree models from scikit-learn — useful as a
diversity check against the gradient-boosted majority. Logistic
regression also serves as a strong baseline that often wins on
small, well-engineered feature sets.

Each `<name>_classifier(params)` returns an unfitted estimator;
each `<name>_search_space(trial)` defines the Optuna search space
matching the registry shape.
"""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def logreg_classifier(params: dict[str, Any] | None = None) -> LogisticRegression:
    """L2-regularised logistic regression. Best partner with the
    StandardScaler in the preprocessing pipeline — without scaling,
    the optimiser struggles on heterogeneous feature scales.

    Notes on sklearn 1.8 compatibility: `penalty` and `n_jobs` were
    deprecated; defaults give L2 / lbfgs single-thread behaviour
    without the warning.
    """
    base: dict[str, Any] = {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 500,
        "random_state": 0,
    }
    if params:
        base.update(params)
    return LogisticRegression(**base)


def logreg_search_space(trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 200, 1000),
    }


def extratrees_classifier(params: dict[str, Any] | None = None) -> ExtraTreesClassifier:
    """Randomised tree ensemble. Faster than RandomForest, often
    matches it on small/medium feature sets, and gives a non-boosted
    perspective for the zoo."""
    base: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": 1,
        "random_state": 0,
    }
    if params:
        base.update(params)
    return ExtraTreesClassifier(**base)


def extratrees_search_space(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 4, 14),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.8]),
    }


def mlp_classifier(params: dict[str, Any] | None = None) -> MLPClassifier:
    """Multi-layer perceptron. Benefits heavily from the StandardScaler
    in the preprocessor — without scaled inputs, sgd/adam diverges or
    plateaus on heterogeneous feature scales. Single hidden layer by
    default; the search space lets Optuna pick wider/deeper nets.
    """
    base: dict[str, Any] = {
        "hidden_layer_sizes": (64,),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "max_iter": 300,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 15,
        "random_state": 0,
    }
    if params:
        base.update(params)
    return MLPClassifier(**base)


def mlp_search_space(trial) -> dict[str, Any]:
    # Hidden architecture as a categorical so Optuna explores discrete
    # topologies rather than continuous neuron counts (which would force
    # tiny gradients or trivial nets).
    arch = trial.suggest_categorical(
        "hidden_layer_sizes",
        ["32", "64", "128", "32_32", "64_32", "64_64"],
    )
    layers = tuple(int(t) for t in arch.split("_"))
    return {
        "hidden_layer_sizes": layers,
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float(
            "learning_rate_init", 1e-4, 1e-2, log=True
        ),
        "max_iter": trial.suggest_int("max_iter", 200, 600),
    }


def histgb_classifier(params: dict[str, Any] | None = None) -> HistGradientBoostingClassifier:
    """Histogram-based gradient boosting. Native sklearn implementation,
    fast on small/medium tabular data, handles NaN gracefully, and
    gives the zoo a third tree-boosted family alongside LightGBM and
    XGBoost — the three rarely agree on the same trial winner.
    """
    base: dict[str, Any] = {
        "loss": "log_loss",
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_depth": None,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "random_state": 0,
    }
    if params:
        base.update(params)
    return HistGradientBoostingClassifier(**base)


def histgb_search_space(trial) -> dict[str, Any]:
    return {
        "max_iter": trial.suggest_int("max_iter", 50, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 1.0, log=True),
    }

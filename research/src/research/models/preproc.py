"""Feature preprocessing pipeline.

Default chain:
  1. `VarianceThreshold` — drop near-constant features that the upstream
     feature engine sometimes emits during quiet markets (an EWMA σ
     stuck at ~0, etc.). Constant columns kill logistic regression's
     conditioning and contribute nothing to tree splits anyway.
  2. `StandardScaler` — zero-mean, unit-variance. Tree models are
     scale-invariant so it's a no-op for them; logreg + MLP rely on
     it heavily.
  3. `SelectKBest(mutual_info_classif)` — optional feature reduction.
     The `k` argument is `"all"` by default (no reduction); the
     training driver sweeps it as part of its Optuna search space so
     each model family can prefer a different effective feature count.

ONNX export consideration: skl2onnx supports VarianceThreshold and
StandardScaler natively. SelectKBest also has a converter (`Imputer`-
style ArrayFeatureExtractor under the hood). Wrapping the model in a
Pipeline produces a single ONNX graph that the live `inference` crate
can serve without a separate normalisation step. Keep new transformers
ONNX-friendly.
"""
from __future__ import annotations

from typing import Any, Union

from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# `feature_select_k` accepts either an integer (top-K features) or the
# sentinel string `"all"` (no selection — pass through). Centralised
# here so the training driver and tests share the same alphabet.
SelectK = Union[int, str]


def _make_select_kbest(k: SelectK) -> SelectKBest:
    """Build a SelectKBest with mutual_info_classif. Mutual info is the
    natural choice for binary tabular classification — it captures
    nonlinear dependence (relevant for tree-friendly features like RSI
    bands) where ANOVA F-test only sees linear separability.

    Pass `k="all"` to make this a no-op pass-through.
    """
    return SelectKBest(score_func=mutual_info_classif, k=k)


def make_preprocessor(
    variance_threshold: float = 1e-8,
    feature_select_k: SelectK = "all",
) -> Pipeline:
    """A pre-classifier preprocessing pipeline.

    `variance_threshold` defaults to 1e-8 — drops only features that
    are *numerically* constant on the training window. Higher values
    aggressively prune low-variance signals (e.g. RSI stuck near 50).

    `feature_select_k` defaults to `"all"` (no selection). The
    training driver overrides this via Optuna; values like 12 / 16 /
    20 trade away features that mutual-info-classif scores poorly. The
    selection is fit on the **training fold only** inside CalibratedCV
    so there's no leakage.

    When `feature_select_k == "all"`, the SelectKBest step is omitted
    entirely. skl2onnx's converter for SelectKBest accumulates float32
    rounding error on each indexed read even when k="all" is a no-op,
    which broke logreg's ONNX roundtrip (max error 0.42 on real bar
    feature distributions). Omitting the no-op step avoids the bug
    AND keeps the ONNX graph smaller.
    """
    steps: list[tuple[str, Any]] = [
        ("variance", VarianceThreshold(threshold=variance_threshold)),
        ("scaler", StandardScaler()),
    ]
    if feature_select_k != "all":
        steps.append(("select", _make_select_kbest(feature_select_k)))
    return Pipeline(steps=steps)


def wrap_with_preprocessor(
    estimator: Any,
    *,
    variance_threshold: float = 1e-8,
    feature_select_k: SelectK = "all",
) -> Pipeline:
    """Glue a pre-classifier preprocessor in front of `estimator` so
    the resulting Pipeline has a single `fit(X, y)` / `predict_proba(X)`
    surface. Used by the zoo to give every candidate the same
    preprocessing baseline.

    Important: pass an *unwrapped* estimator (e.g. an LGBMClassifier),
    not a CalibratedClassifierCV. The training driver may apply a
    calibrator inside the resulting Pipeline (so the final shape is
    `Pipeline([variance, scaler, CalibratedClassifierCV(estimator)])`),
    which exports cleanly to ONNX. Wrapping the calibrator *around*
    the Pipeline produces unreliable ONNX graphs — see Phase A3
    diagnosis.
    """
    pre = make_preprocessor(
        variance_threshold=variance_threshold,
        feature_select_k=feature_select_k,
    )
    return Pipeline(steps=[*pre.steps, ("estimator", estimator)])


# Discrete `k` values the training driver suggests via Optuna. `"all"`
# is the no-op baseline; the tighter integers force the optimiser to
# discover whether feature reduction helps.
SELECT_K_CHOICES: list[SelectK] = ["all", 12, 16, 20]

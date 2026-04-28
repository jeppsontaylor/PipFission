"""Unit tests for the model zoo registry + preprocessor + each spec."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from research.models import (
    DEFAULT_CANDIDATES,
    MODEL_REGISTRY,
    get_spec,
    make_preprocessor,
    parse_candidates,
    wrap_with_preprocessor,
)


def _trial_stub() -> SimpleNamespace:
    """Cheap Optuna `Trial` stand-in: each `suggest_*` returns the
    midpoint of the requested range. Lets us exercise the search-space
    callables without spinning up Optuna."""

    def suggest_int(_name, lo, hi):
        return (lo + hi) // 2

    def suggest_float(_name, lo, hi, **_kwargs):
        return (lo + hi) / 2

    def suggest_categorical(_name, choices):
        return choices[0]

    return SimpleNamespace(
        suggest_int=suggest_int,
        suggest_float=suggest_float,
        suggest_categorical=suggest_categorical,
    )


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_each_model_constructs_with_defaults(name: str):
    spec = get_spec(name)
    clf = spec.constructor()
    # Smoke fit + predict_proba on a tiny synthetic problem — proves
    # the estimator wires up cleanly under the preprocessor.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 8))
    y = (X[:, 0] + 0.3 * rng.normal(size=100) > 0).astype(int)
    pipe = wrap_with_preprocessor(clf)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_search_space_returns_param_dict(name: str):
    spec = get_spec(name)
    params = spec.search_space(_trial_stub())
    assert isinstance(params, dict)
    assert len(params) > 0
    # The constructor must accept the search-space output.
    clf = spec.constructor(params)
    assert clf is not None


def test_default_candidates_includes_diverse_families():
    # Make sure the default zoo isn't accidentally narrowed back to lgbm.
    assert len(DEFAULT_CANDIDATES) >= 4
    # Tree-based and linear should both be present.
    assert "lgbm" in DEFAULT_CANDIDATES
    assert "logreg" in DEFAULT_CANDIDATES


def test_parse_candidates_default():
    assert parse_candidates(None) == list(DEFAULT_CANDIDATES)
    assert parse_candidates("") == list(DEFAULT_CANDIDATES)


def test_parse_candidates_csv_with_whitespace():
    assert parse_candidates(" lgbm , logreg, xgb ") == ["lgbm", "logreg", "xgb"]


def test_parse_candidates_rejects_unknown():
    with pytest.raises(ValueError, match="unknown model"):
        parse_candidates("lgbm,this_does_not_exist")


def test_get_spec_rejects_unknown():
    with pytest.raises(ValueError, match="unknown model"):
        get_spec("totally_not_real")


def test_preprocessor_drops_constant_features():
    X = np.array(
        [[1.0, 5.0, 0.5], [2.0, 5.0, 0.7], [3.0, 5.0, 0.6], [4.0, 5.0, 0.4]]
    )
    y = np.array([0, 1, 0, 1])
    pre = make_preprocessor()
    Xt = pre.fit_transform(X, y)
    # Column 1 is constant → should be dropped by VarianceThreshold.
    # SelectKBest defaults to k="all" so it doesn't drop anything else.
    assert Xt.shape == (4, 2)


def test_preprocessor_centers_and_scales():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=10.0, scale=3.0, size=(200, 4))
    y = (X[:, 0] > 10.0).astype(int)
    pre = make_preprocessor()
    Xt = pre.fit_transform(X, y)
    # Each surviving column should be ~zero-mean / unit-std.
    np.testing.assert_allclose(Xt.mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(Xt.std(axis=0), 1.0, atol=0.05)


def test_preprocessor_select_k_int():
    """SelectKBest with int k drops down to that many columns."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 8))
    y = (X[:, 0] + 0.3 * X[:, 1] + 0.5 * rng.normal(size=200) > 0).astype(int)
    pre = make_preprocessor(feature_select_k=3)
    Xt = pre.fit_transform(X, y)
    assert Xt.shape == (200, 3)


def test_preprocessor_select_k_all_passthrough():
    """k='all' is the no-op default — no features dropped beyond
    VarianceThreshold's near-constant filter."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 5))
    y = (X[:, 0] > 0).astype(int)
    pre = make_preprocessor(feature_select_k="all")
    Xt = pre.fit_transform(X, y)
    assert Xt.shape == (50, 5)


def test_select_k_choices_alphabet_is_sane():
    """The training driver pulls k options from this list — make sure
    the alphabet stays sensible (always includes 'all', no duplicates,
    integer values are positive)."""
    from research.models.preproc import SELECT_K_CHOICES

    assert "all" in SELECT_K_CHOICES
    assert len(SELECT_K_CHOICES) == len(set(map(str, SELECT_K_CHOICES)))
    for k in SELECT_K_CHOICES:
        if isinstance(k, int):
            assert k > 0

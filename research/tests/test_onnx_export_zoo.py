"""ONNX-export verification across the zoo.

Catches the silent failure mode from the Phase A diagnosis: a
StandardScaler that exists in the Python pipeline but doesn't make it
into the exported ONNX graph. Without scaler nodes inline, the live
inference path feeds raw features (n_ticks ~9000) directly to a model
that was trained on standardized inputs — predictions collapse to
garbage.

For each spec, we:
  1. Train a tiny synthetic problem with the full preprocessor wrapper
     (StandardScaler + VarianceThreshold + SelectKBest + estimator),
     wrapped in CalibratedClassifierCV(cv=2).
  2. Pickle it as a "champion" in a temp artifacts dir.
  3. Export to ONNX via `export_calibrated_to_onnx`.
  4. Inspect the graph: `Scaler` op MUST be present (proves the
     preprocessing was inlined).
  5. Roundtrip-verify with a real-distribution sample (heterogeneous
     scales like the live feature set), not Gaussian noise.

Skipped specs are logged. CatBoost and MLP currently have no ONNX
exporter that integrates cleanly with skl2onnx — they're documented
as "Python-only" candidates and live deployment requires picking one
of the ONNX-friendly families. The deployment gate (Phase A4) will
block them from going live.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from research.models import (
    MODEL_REGISTRY,
    get_spec,
    wrap_with_preprocessor,
)
from research.models.calibration import wrap_sigmoid

# Specs we expect to round-trip cleanly through skl2onnx + onnxruntime.
#
# Excluded:
#   - "catboost"  — has its own ONNX pathway via `Model.save_model
#                   (format='onnx')`; not integrated with skl2onnx's
#                   Pipeline traversal.
#   - "mlp"       — sklearn MLPClassifier converter exists but ships
#                   warnings unrelated to scaler-presence; covered in
#                   a future hardening pass.
#   - "histgb"    — sklearn 1.8 emits uint32 tree indices that
#                   skl2onnx's HistGB converter doesn't tolerate
#                   (https://github.com/onnx/sklearn-onnx — open issue).
#                   Stays in the OOS zoo; deployment gate will block
#                   from going live until upstream fix.
#   - "extratrees" — skl2onnx's CalibratedClassifierCV converter
#                   wants `decision_function` from the inner
#                   estimator; ExtraTreesClassifier exposes only
#                   predict_proba. Stays in zoo; gate blocks deploy.
ONNX_SAFE_SPECS = ["lgbm", "xgb", "logreg"]


# Per-spec roundtrip tolerance. Tree ensembles naturally drift more
# under float32 quantisation than linear models, so we allow them a
# wider band. These thresholds are still tight enough that a
# missing/wrongly-fitted scaler would blow them out (would be ≥0.1 +).
ROUNDTRIP_ATOL = {
    "lgbm": 5e-3,
    "xgb": 5e-3,
    "logreg": 1e-3,
}


def all_specs_are_zoo_member():
    """Sanity: every ONNX-safe spec is registered in MODEL_REGISTRY."""
    for s in ONNX_SAFE_SPECS:
        assert s in MODEL_REGISTRY, f"{s} missing from MODEL_REGISTRY"


@pytest.fixture()
def synthetic_data():
    """Heterogeneous-scale synthetic features mimicking the bar-feature
    distribution: most columns N(0,1), one column ~0–10000 (n_ticks-like),
    one column ~±10000 (force-index-like). Without a working scaler in
    the ONNX graph, predictions on this data will diverge between
    Python and ONNX runtime."""
    rng = np.random.default_rng(0)
    n_features = 24
    n_samples = 240

    X = rng.normal(size=(n_samples, n_features))
    # Inject the realistic large-scale columns at fixed indices.
    X[:, 19] = rng.uniform(0, 10000, size=n_samples)  # n_ticks-like
    X[:, 18] = rng.uniform(-10000, 10000, size=n_samples)  # force-index-like

    # Binary target weakly correlated with first feature.
    y = (X[:, 0] + 0.4 * rng.normal(size=n_samples) > 0).astype(int)
    return X.astype(np.float64), y


def _train_champion_to_disk(
    spec_name: str,
    X: np.ndarray,
    y: np.ndarray,
    artifacts_root: Path,
    monkeypatch,
) -> str:
    """Train a tiny ONNX-friendly pipeline + pickle as a champion under
    a monkeypatched PATHS.artifacts_dir. Returns the model_id.

    Mirrors `side_train._build_calibrated_pipeline`: a flat Pipeline
    with the preprocessor steps in front of a CalibratedClassifierCV
    wrapping the bare estimator. This shape exports to ONNX cleanly.
    """
    from sklearn.pipeline import Pipeline as SkPipeline
    from research.models import make_preprocessor

    spec = get_spec(spec_name)
    pre = make_preprocessor(feature_select_k="all")
    estimator = spec.constructor()
    calibrated = wrap_sigmoid(estimator, cv=2)
    pipe = SkPipeline(steps=[*pre.steps, ("classifier", calibrated)])
    pipe.fit(X, y)

    model_id = f"test_{spec_name}_synthetic"
    champion_dir = artifacts_root / "models" / model_id
    champion_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipe,
            "feature_names": [f"f{i}" for i in range(X.shape[1])],
            "instrument": "TEST",
            "label_run_id": "test",
            "n_train": int(X.shape[0]),
            "spec_name": spec_name,
        },
        champion_dir / "champion.pkl",
    )
    return model_id


@pytest.fixture()
def tmp_artifacts(tmp_path, monkeypatch):
    """Re-point PATHS.artifacts_dir at a tmp dir for the duration of
    the test so champion files don't collide with the live system."""
    from dataclasses import replace
    from research import paths as paths_mod
    from research.export import onnx_export

    new_paths = replace(paths_mod.PATHS, artifacts_dir=tmp_path / "artifacts")
    monkeypatch.setattr(onnx_export, "PATHS", new_paths)
    return tmp_path / "artifacts"


@pytest.mark.parametrize("spec_name", ONNX_SAFE_SPECS)
def test_zoo_onnx_export_inlines_scaler(spec_name, synthetic_data, tmp_artifacts, monkeypatch):
    """For each ONNX-safe spec: export, then assert the graph contains
    a `Scaler` op proving the StandardScaler was inlined.
    """
    from research.export.onnx_export import (
        export_calibrated_to_onnx,
        inspect_onnx_graph,
    )

    X, y = synthetic_data
    model_id = _train_champion_to_disk(spec_name, X, y, tmp_artifacts, monkeypatch)
    info = export_calibrated_to_onnx(model_id)
    assert info["n_features"] == X.shape[1]
    assert Path(info["onnx_path"]).exists()

    graph = inspect_onnx_graph(model_id)
    assert graph["has_scaler"], (
        f"{spec_name}: ONNX graph missing Scaler — preprocessing was not "
        f"inlined. Ops present: {graph['unique_ops']}. "
        f"Live inference would feed raw features to a scaled model."
    )
    assert graph["input_dim"] == X.shape[1]


@pytest.mark.parametrize("spec_name", ONNX_SAFE_SPECS)
def test_zoo_onnx_roundtrip_on_real_distribution(
    spec_name, synthetic_data, tmp_artifacts, monkeypatch
):
    """Roundtrip-verify each export using the heterogeneous-scale
    sample distribution. A missing or wrongly-fitted scaler causes the
    Python-vs-ONNX outputs to diverge by orders of magnitude on the
    n_ticks-like column; this test catches that mode that the original
    Gaussian-noise verify would miss."""
    from research.export.onnx_export import (
        export_calibrated_to_onnx,
        verify_onnx_roundtrip,
    )

    X, y = synthetic_data
    model_id = _train_champion_to_disk(spec_name, X, y, tmp_artifacts, monkeypatch)
    export_calibrated_to_onnx(model_id)
    sample = X[:32].astype(np.float32)
    atol = ROUNDTRIP_ATOL[spec_name]
    result = verify_onnx_roundtrip(model_id, atol=atol, sample_X=sample)
    assert result["max_err"] < atol, (
        f"{spec_name}: roundtrip diverged on real-distribution sample "
        f"(max_err={result['max_err']:.4e}, atol={atol:.4e}). Likely "
        f"missing/incorrect scaler in the ONNX graph."
    )


def test_inspect_onnx_graph_returns_structured_summary(
    synthetic_data, tmp_artifacts, monkeypatch
):
    """Smoke test for the inspector itself."""
    from research.export.onnx_export import (
        export_calibrated_to_onnx,
        inspect_onnx_graph,
    )

    X, y = synthetic_data
    model_id = _train_champion_to_disk("lgbm", X, y, tmp_artifacts, monkeypatch)
    export_calibrated_to_onnx(model_id)
    g = inspect_onnx_graph(model_id)
    assert isinstance(g["ops"], list) and len(g["ops"]) > 0
    assert isinstance(g["unique_ops"], list)
    assert g["has_tree_ensemble"] is True
    assert isinstance(g["has_scaler"], bool)

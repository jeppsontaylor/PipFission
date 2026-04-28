"""Convert the calibrated LightGBM champion to ONNX so the Rust live
engine can load it via the `ort` crate.

Pipeline:
  1. Load `<artifacts>/models/<model_id>/champion.pkl` (joblib).
  2. Wrap the inner LightGBM booster in onnxmltools' converter.
  3. Apply Platt-sigmoid post-processing as an ONNX subgraph (or, for
     v1, persist the calibrator's slope/intercept in the manifest and
     let the Rust inference crate apply them at runtime — that path is
     more robust to converter version drift).
  4. Round-trip verify: Python `predict_proba` ≈ ONNX runtime to 1e-5.

Implementation note: skl2onnx + onnxmltools both have well-documented
LightGBM converters but the API drifts across versions. We use the
ONNXMLTools entry-point and pass a `FloatTensorType` input with
explicit shape `(None, n_features)`. If the converter fails to find
the LGBM converter automatically, we register it manually.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnx
import onnxruntime as ort

from research.paths import PATHS


def _register_lightgbm_converter() -> None:
    """Idempotent: register LightGBM with onnxmltools' shape calculator
    + converter so skl2onnx can dispatch to it. The user-facing entry
    point on newer versions is `update_registered_converter`."""
    try:
        from lightgbm import LGBMClassifier
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
            convert_lightgbm,
        )
        from skl2onnx import update_registered_converter
        from skl2onnx.common.shape_calculator import (
            calculate_linear_classifier_output_shapes,
        )

        update_registered_converter(
            LGBMClassifier,
            "LightGbmLGBMClassifier",
            calculate_linear_classifier_output_shapes,
            convert_lightgbm,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
    except Exception:  # noqa: BLE001
        # If registration fails, fall through and let convert_sklearn raise
        # a clear error at conversion time.
        pass


def _register_xgboost_converter() -> None:
    """Idempotent: register XGBoost's sklearn-API classifier with
    onnxmltools' converter so skl2onnx can traverse Pipelines that
    contain `XGBClassifier`. Same pattern as LightGBM registration —
    skl2onnx does not ship XGBoost support out of the box because
    XGBoost is a separate vendor.
    """
    try:
        from xgboost import XGBClassifier
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
            convert_xgboost,
        )
        from skl2onnx import update_registered_converter
        from skl2onnx.common.shape_calculator import (
            calculate_linear_classifier_output_shapes,
        )

        update_registered_converter(
            XGBClassifier,
            "XGBoostXGBClassifier",
            calculate_linear_classifier_output_shapes,
            convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
    except Exception:  # noqa: BLE001
        pass


def _register_external_converters() -> None:
    """Register every non-sklearn estimator we ship in the zoo. Called
    once at the top of `export_calibrated_to_onnx` so the converter
    table is populated before skl2onnx walks the Pipeline graph."""
    _register_lightgbm_converter()
    _register_xgboost_converter()


def export_calibrated_to_onnx(model_id: str) -> dict[str, Any]:
    """Convert the champion CalibratedClassifierCV (LGBM/XGB/sklearn
    inside) to ONNX. Returns a manifest dict with absolute paths and
    verification info.
    """
    _register_external_converters()
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    champion_dir = PATHS.artifacts_dir / "models" / model_id
    pkl = champion_dir / "champion.pkl"
    if not pkl.exists():
        raise RuntimeError(f"champion missing at {pkl}")
    blob = joblib.load(pkl)
    model = blob["model"]
    feat_names = blob["feature_names"]
    n_features = len(feat_names)

    initial_type = [("input", FloatTensorType([None, n_features]))]
    # Pin both opsets so the file is loadable by `onnxruntime` and `ort`
    # (Rust). ai.onnx.ml v3 is the highest version supported by the
    # version of skl2onnx pinned in requirements.txt; default target opset
    # for ai.onnx is set to a conservative 17 for broad compatibility.
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset={"": 17, "ai.onnx.ml": 3},
        options={id(model): {"zipmap": False}},
    )
    out = champion_dir / "champion.onnx"
    onnx.save_model(onnx_model, str(out))
    return {
        "model_id": model_id,
        "onnx_path": str(out),
        "n_features": n_features,
        "feature_names": feat_names,
    }


def verify_onnx_roundtrip(
    model_id: str,
    n_samples: int = 32,
    atol: float = 1e-4,
    sample_X: np.ndarray | None = None,
) -> dict[str, float]:
    """Score Python and ONNX on the same input, assert outputs match.

    `sample_X` lets the caller feed **real bar features** rather than
    Gaussian-shaped noise. This is critical when the pipeline includes
    a StandardScaler — Gaussian-distributed test inputs minimize the
    contribution of any missing scaler step in the ONNX graph, so a
    silently-broken export passes. With real-distribution samples
    (e.g. raw n_ticks ~9000), a missing scaler on either side produces
    catastrophic divergence the test catches.

    When `sample_X` is None, falls back to the original Gaussian input
    for backward-compatibility (used by the basic smoke test).
    """
    champion_dir = PATHS.artifacts_dir / "models" / model_id
    pkl = champion_dir / "champion.pkl"
    onnx_path = champion_dir / "champion.onnx"
    blob = joblib.load(pkl)
    model = blob["model"]
    n_features = len(blob["feature_names"])
    if sample_X is None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    else:
        if sample_X.shape[1] != n_features:
            raise RuntimeError(
                f"sample_X has {sample_X.shape[1]} cols, expected {n_features}"
            )
        X = sample_X.astype(np.float32)
        n_samples = X.shape[0]

    py_probs = model.predict_proba(X)[:, 1]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = {sess.get_inputs()[0].name: X}
    outputs = sess.run(None, inputs)
    # The skl2onnx output for binary classification is typically a list:
    # outputs[0] = labels (N,), outputs[1] = probabilities (N, 2).
    onnx_probs = None
    for o in outputs:
        arr = np.asarray(o)
        if arr.ndim == 2 and arr.shape == (n_samples, 2):
            onnx_probs = arr[:, 1]
            break
    if onnx_probs is None:
        raise RuntimeError(
            f"ONNX output shape unexpected: {[np.asarray(o).shape for o in outputs]}"
        )
    max_err = float(np.max(np.abs(py_probs - onnx_probs)))
    if max_err > atol:
        raise RuntimeError(f"ONNX roundtrip max error {max_err:.6f} > atol {atol}")
    return {"max_err": max_err, "n_samples": float(n_samples)}


def inspect_onnx_graph(model_id: str) -> dict[str, Any]:
    """Return a structural summary of the exported ONNX graph. Used by
    tests + the deployment gate to confirm preprocessing steps were
    actually inlined: a Pipeline's `StandardScaler` must produce a
    `Scaler` op, and `VarianceThreshold` shows up as a feature-extractor
    or constant-fold pattern. If a scaler-using model exports without
    `Scaler`, training-vs-inference scales diverge silently and the
    live model returns garbage.
    """
    onnx_path = PATHS.artifacts_dir / "models" / model_id / "champion.onnx"
    m = onnx.load(str(onnx_path))
    op_types = [n.op_type for n in m.graph.node]
    return {
        "n_nodes": len(op_types),
        "ops": op_types,
        "unique_ops": sorted(set(op_types)),
        "has_scaler": any(t == "Scaler" for t in op_types),
        "has_tree_ensemble": any("TreeEnsemble" in t for t in op_types),
        "input_dim": (
            m.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
            if m.graph.input
            else None
        ),
    }

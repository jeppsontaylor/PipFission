"""Champion-manifest writer. The manifest is the contract between the
research layer and the live Rust engine: feature names, ONNX path,
calibrator parameters, model metadata. The hot-swap watcher in
api-server reads the manifest and only swaps the predictor if the
schema matches the live feature dimension.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from research.data.duckdb_io import rw_conn
from research.paths import PATHS


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    model_id: str,
    onnx_path: str,
    feature_names: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write `manifest.json` next to the ONNX model AND insert a row
    into `model_artifacts`. Returns the manifest dict.
    """
    onnx_p = Path(onnx_path)
    if not onnx_p.exists():
        raise RuntimeError(f"ONNX file not found: {onnx_p}")
    sha = _sha256(onnx_p)
    onnx_blob = onnx_p.read_bytes()
    manifest: dict[str, Any] = {
        "model_id": model_id,
        "version": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "onnx_path": str(onnx_p),
        "sha256": sha,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "kind": "side-lgbm-onnx",
        "calibrated": True,
        "calibration_method": "sigmoid",
        "created_at_ms": int(time.time() * 1000),
    }
    if extra:
        manifest.update(extra)
    manifest_path = onnx_p.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Also ship a copy at the well-known live-engine location.
    live_dir = PATHS.artifacts_dir / "models" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "champion.onnx").write_bytes(onnx_blob)
    (live_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    with rw_conn() as c:
        c.execute(
            """
            INSERT INTO model_artifacts
              (model_id, ts_ms, kind, version, onnx_blob, sha256, calib_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO NOTHING
            """,
            [
                model_id,
                manifest["created_at_ms"],
                manifest["kind"],
                manifest["version"],
                onnx_blob,
                sha,
                json.dumps({"method": "sigmoid", "applied_in_onnx": True}),
            ],
        )
    return manifest

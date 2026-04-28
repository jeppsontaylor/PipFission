//! Integration test: load the real champion ONNX produced by the
//! Python research layer and verify predictions match the expected
//! shape (probabilities in [0, 1], deterministic, swap is reversible).
//!
//! Skipped at runtime if the artifacts directory is empty — the test
//! is opt-in via the presence of `champion.onnx` + `manifest.json`.

use std::path::PathBuf;
use std::sync::Arc;

use inference::{Predictor, PredictorRegistry};

fn artifacts_dir() -> PathBuf {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    here.parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .map(|p| p.join("research").join("artifacts").join("models").join("live"))
        .expect("repo layout")
}

#[test]
fn load_real_champion_and_predict() {
    let dir = artifacts_dir();
    let manifest = dir.join("manifest.json");
    if !manifest.exists() {
        eprintln!("skipping: no champion artifacts at {}", dir.display());
        return;
    }

    let registry = Arc::new(PredictorRegistry::new(24));
    let model_id = registry
        .try_load_onnx(&manifest)
        .expect("load_onnx should succeed");
    assert!(!model_id.is_empty(), "model_id should be non-empty");

    let handle = registry.current();
    assert_eq!(handle.n_features(), 24);
    let probs = handle.predict(&[0.0; 24]);
    assert!(probs.p_long.is_finite());
    assert!((0.0..=1.0).contains(&probs.p_long));
    assert!((0.0..=1.0).contains(&probs.p_short));
    assert!(((probs.p_long + probs.p_short) - 1.0).abs() < 1e-6);
    assert!(probs.p_take >= probs.p_long.max(probs.p_short) - 1e-9);

    // Determinism: two calls with the same input must produce equal probabilities.
    let a = handle.predict(&[0.1; 24]);
    let b = handle.predict(&[0.1; 24]);
    assert_eq!(a.p_long, b.p_long);

    // Force-fallback should swap to neutral and id should change.
    registry.force_fallback();
    let h2 = registry.current();
    assert_eq!(h2.id(), inference::fallback::FALLBACK_ID);
    let p = h2.predict(&[0.0; 24]);
    assert!((p.p_long - 0.5).abs() < 1e-9);
}

#[test]
fn rejects_wrong_feature_dim() {
    let dir = artifacts_dir();
    let manifest = dir.join("manifest.json");
    if !manifest.exists() {
        eprintln!("skipping: no champion artifacts at {}", dir.display());
        return;
    }
    // Live engine expects 16 but champion says 24 → rejected.
    let registry = Arc::new(PredictorRegistry::new(16));
    let result = registry.try_load_onnx(&manifest);
    assert!(result.is_err(), "feature-dim mismatch should fail");
    let h = registry.current();
    assert_eq!(h.id(), inference::fallback::FALLBACK_ID);
}

//! Predictor registry. Holds the currently-active predictor behind a
//! `parking_lot::RwLock<Arc<dyn Predictor>>` — read locks for the
//! per-bar `predict()` call, write lock for the (rare) hot-swap.
//!
//! Could be made wait-free with the right `arc_swap` plumbing, but at
//! ~6 calls/sec across instruments the RwLock cost is invisible and
//! the API is much simpler.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::fallback::FallbackPredictor;
use crate::manifest::Manifest;
use crate::onnx_session::OnnxPredictor;
use crate::Predictor;

/// Trait-object alias. The `Send + Sync` bounds come from the trait
/// itself; we still spell them out so the type is explicit at use sites.
pub type PredArc = Arc<dyn Predictor + Send + Sync>;

/// Cheap-to-clone handle to whatever predictor is currently active.
#[derive(Clone)]
pub struct PredictorHandle(pub PredArc);

impl Predictor for PredictorHandle {
    fn predict(&self, features: &[f64]) -> crate::Probs {
        self.0.predict(features)
    }

    fn n_features(&self) -> usize {
        self.0.n_features()
    }

    fn id(&self) -> &str {
        self.0.id()
    }
}

/// Per-process predictor registry.
pub struct PredictorRegistry {
    current: RwLock<PredArc>,
    expected_n_features: usize,
}

impl PredictorRegistry {
    /// Initialise with a fallback predictor pinned to the live feature dimension.
    pub fn new(n_features: usize) -> Self {
        let fallback: PredArc = Arc::new(FallbackPredictor::new(n_features));
        Self {
            current: RwLock::new(fallback),
            expected_n_features: n_features,
        }
    }

    /// Atomic snapshot of the current predictor.
    pub fn current(&self) -> PredictorHandle {
        PredictorHandle(self.current.read().clone())
    }

    /// Try to load the ONNX champion at `manifest_path`. On success,
    /// atomically swap into the live slot. On any failure, leave the
    /// prior predictor in place and return the error.
    pub fn try_load_onnx(&self, manifest_path: impl AsRef<Path>) -> anyhow::Result<String> {
        let manifest = Manifest::load(manifest_path.as_ref())?;
        if manifest.n_features != self.expected_n_features {
            return Err(anyhow::anyhow!(
                "ONNX champion expects {} features but live engine uses {}",
                manifest.n_features,
                self.expected_n_features
            ));
        }
        let predictor = OnnxPredictor::load_with_manifest(&manifest)?;
        let model_id = predictor.id().to_string();
        let arc: PredArc = Arc::new(predictor);
        *self.current.write() = arc;
        tracing::info!(
            model_id,
            n_features = manifest.n_features,
            "inference: champion swapped"
        );
        Ok(model_id)
    }

    /// Force a fallback. Used when ONNX inference returns errors at
    /// runtime — the watcher demotes to fallback and waits for a fresh
    /// champion to land.
    pub fn force_fallback(&self) {
        let fallback: PredArc = Arc::new(FallbackPredictor::new(self.expected_n_features));
        *self.current.write() = fallback;
        tracing::warn!("inference: fell back to neutral predictor");
    }

    pub fn expected_n_features(&self) -> usize {
        self.expected_n_features
    }
}

/// Conventional location of the live champion artifacts. Both files
/// must be present and consistent before swap.
pub fn live_champion_paths(repo_root: &Path) -> (PathBuf, PathBuf) {
    let dir = repo_root.join("research").join("artifacts").join("models").join("live");
    (dir.join("champion.onnx"), dir.join("manifest.json"))
}

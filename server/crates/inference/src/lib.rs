//! Live model inference. Loads the champion ONNX produced by the
//! Python research layer, runs predictions per bar, and atomically
//! hot-swaps when a new champion appears at the well-known path.
//! Falls back to `strategy::OnlineLogReg` if anything goes wrong so
//! the live engine never goes silent.
//!
//! ## Wire contract
//! The Python research layer publishes:
//!   - `<repo>/research/artifacts/models/live/champion.onnx`
//!   - `<repo>/research/artifacts/models/live/manifest.json`
//!
//! The manifest carries `model_id`, `feature_names`, `n_features`,
//! `kind`, and `sha256`. We only swap the live predictor when:
//!   1. The ONNX file's sha256 matches the manifest.
//!   2. `n_features` == the live engine's feature dimension.
//!   3. A 1-row warm-up prediction succeeds (sanity check).
//!
//! Otherwise we keep the prior predictor and emit a warning.

#![deny(unsafe_code)]

pub mod calibrator;
pub mod fallback;
pub mod hot_swap;
pub mod manifest;
pub mod onnx_session;
pub mod registry;

pub use calibrator::Calibrator;
pub use fallback::FallbackPredictor;
pub use hot_swap::{spawn_hot_swap_watcher, HotSwapEvent};
pub use manifest::Manifest;
pub use onnx_session::OnnxPredictor;
pub use registry::{PredictorHandle, PredictorRegistry};

use serde::{Deserialize, Serialize};

/// Per-bar probabilities consumed by the trader. Aligned with the
/// `trader::Probs` shape so the inference output drops directly in.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Probs {
    pub p_long: f64,
    pub p_short: f64,
    pub p_take: f64,
    pub calibrated: f64,
}

/// Trait every predictor implements. Cheap to call per bar.
pub trait Predictor: Send + Sync {
    /// Score a single feature vector. The slice MUST have length equal
    /// to whatever the predictor was loaded for; a length mismatch is
    /// the predictor's contract violation.
    fn predict(&self, features: &[f64]) -> Probs;

    /// How many features this predictor expects. Used by the registry
    /// to gate hot-swaps: we won't swap to a predictor whose feature
    /// dim differs from the live engine's.
    fn n_features(&self) -> usize;

    /// Diagnostic identifier (e.g. ONNX `model_id` or "fallback-logreg").
    fn id(&self) -> &str;
}

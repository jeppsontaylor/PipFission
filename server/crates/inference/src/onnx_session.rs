//! ONNX-backed predictor over `ort::Session`.
//!
//! Loads a champion ONNX written by `research/export/onnx_export.py`
//! (skl2onnx + onnxmltools, target opset 17, ai.onnx.ml v3). The output
//! signature is two tensors: a label tensor (i64, shape `[N]`) and a
//! probability tensor (f32, shape `[N, 2]`). We only consume the
//! probability tensor.
//!
//! Per-bar inference runs at batch size 1. The session is built with
//! `optimization_level::Level3` so the runtime constant-folds the tree
//! ensemble + sigmoid into something fast.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;

use crate::calibrator::Calibrator;
use crate::manifest::Manifest;
use crate::{Predictor, Probs};

/// ONNX-backed predictor. Holds an `ort::Session` (Send-safe under a
/// mutex; the underlying ORT runtime serialises calls anyway) plus the
/// manifest's metadata.
pub struct OnnxPredictor {
    session: Mutex<Session>,
    n_features: usize,
    model_id: String,
    calibrator: Calibrator,
    /// Cached input tensor name (skl2onnx default = "input").
    input_name: String,
    /// Cached probability output name. skl2onnx emits two outputs;
    /// we sniff the right one at load time.
    proba_name: String,
}

impl OnnxPredictor {
    /// Load a manifest + ONNX file. Verifies sha256 matches the
    /// manifest's recorded hash. Returns an error if anything is off
    /// — caller stays on the fallback predictor in that case.
    pub fn load_with_manifest(manifest: &Manifest) -> Result<Self> {
        let onnx_path = Path::new(&manifest.onnx_path);
        if !onnx_path.exists() {
            return Err(anyhow!(
                "manifest references missing ONNX at {}",
                onnx_path.display()
            ));
        }
        let observed = crate::manifest::sha256_file(onnx_path)?;
        if observed != manifest.sha256 {
            return Err(anyhow!(
                "ONNX hash mismatch — manifest says {} but file is {}",
                &manifest.sha256[..12],
                &observed[..12]
            ));
        }
        Self::load_from_path(onnx_path, manifest.n_features, manifest.model_id.clone())
    }

    /// Lower-level constructor; exposed for tests where you want to
    /// load an ONNX file without round-tripping a manifest.
    pub fn load_from_path(
        onnx_path: &Path,
        n_features: usize,
        model_id: String,
    ) -> Result<Self> {
        let session = Session::builder()
            .with_context(|| "ort::Session::builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .with_context(|| "set optimization level")?
            .commit_from_file(onnx_path)
            .with_context(|| format!("load ONNX from {}", onnx_path.display()))?;

        let input_name = session
            .inputs
            .first()
            .ok_or_else(|| anyhow!("ONNX has no inputs"))?
            .name
            .clone();
        // skl2onnx emits two outputs for binary classification: "label"
        // (i64) and "probabilities" (f32, shape [N, 2]). Output names
        // can vary; we pick the f32 2-D one.
        let proba_name = session
            .outputs
            .iter()
            .find(|o| {
                let dtype_str = format!("{:?}", o.output_type);
                dtype_str.contains("Float32") || dtype_str.contains("FLOAT")
            })
            .ok_or_else(|| {
                anyhow!(
                    "no float output found in ONNX session: outputs = {:?}",
                    session.outputs.iter().map(|o| &o.name).collect::<Vec<_>>()
                )
            })?
            .name
            .clone();

        let predictor = Self {
            session: Mutex::new(session),
            n_features,
            model_id,
            calibrator: Calibrator::default(),
            input_name,
            proba_name,
        };
        // 1-row warm-up so the first live call doesn't pay JIT cost.
        let warm = vec![0.0_f64; n_features];
        let _ = predictor.predict(&warm);
        Ok(predictor)
    }

    fn predict_p_long(&self, features: &[f64]) -> Result<f64> {
        if features.len() != self.n_features {
            return Err(anyhow!(
                "feature vector length {} doesn't match expected {}",
                features.len(),
                self.n_features
            ));
        }
        let row: Vec<f32> = features.iter().map(|&v| v as f32).collect();
        let arr = Array2::from_shape_vec((1, self.n_features), row)?;
        let input_value = Value::from_array(arr).map_err(|e| anyhow!("from_array: {e}"))?;
        let mut session = self.session.lock().map_err(|_| anyhow!("session mutex poisoned"))?;
        let outputs = session
            .run(ort::inputs![self.input_name.clone() => input_value])
            .map_err(|e| anyhow!("ort run: {e}"))?;
        let proba = outputs
            .get(self.proba_name.as_str())
            .ok_or_else(|| anyhow!("output {} missing", self.proba_name))?;
        let view = proba
            .try_extract_array::<f32>()
            .map_err(|e| anyhow!("extract f32 array: {e}"))?;
        // Shape is [1, 2]; class 1 == positive.
        let view2 = view
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| anyhow!("output not 2-D: {e}"))?;
        if view2.shape() != [1usize, 2usize] {
            return Err(anyhow!(
                "unexpected output shape {:?}; want [1, 2]",
                view2.shape()
            ));
        }
        let p_pos = view2[[0, 1]] as f64;
        Ok(p_pos.clamp(0.0, 1.0))
    }
}

impl Predictor for OnnxPredictor {
    fn predict(&self, features: &[f64]) -> Probs {
        match self.predict_p_long(features) {
            Ok(p_long) => {
                let p_long = self.calibrator.apply(p_long);
                let p_short = (1.0 - p_long).clamp(0.0, 1.0);
                let p_take = p_long.max(p_short);
                Probs {
                    p_long,
                    p_short,
                    p_take,
                    calibrated: p_take,
                }
            }
            Err(e) => {
                // Stay deterministic on errors: emit a neutral 0.5
                // probability so the trader thresholds will skip the
                // bar. The registry will downgrade to fallback on the
                // next prediction.
                tracing::warn!(
                    error = %e,
                    model_id = %self.model_id,
                    "onnx predict failed; emitting neutral probabilities"
                );
                Probs {
                    p_long: 0.5,
                    p_short: 0.5,
                    p_take: 0.5,
                    calibrated: 0.5,
                }
            }
        }
    }

    fn n_features(&self) -> usize {
        self.n_features
    }

    fn id(&self) -> &str {
        &self.model_id
    }
}

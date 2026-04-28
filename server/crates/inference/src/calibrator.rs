//! Probability calibrator. Today the Python research layer bakes the
//! Platt/sigmoid calibration into the ONNX subgraph (via
//! sklearn's `CalibratedClassifierCV`), so this module is mostly a
//! pass-through. Kept as a separate module so that:
//!   1. If we later move calibration outside ONNX, we can drop the
//!      Platt parameters into `Calibrator` without changing the
//!      `Predictor` trait.
//!   2. The OnnxPredictor still has a place to apply final clipping
//!      before publishing probabilities to the trader.

#[derive(Clone, Copy, Debug)]
pub enum Calibrator {
    /// No-op (calibration applied inside the ONNX graph).
    Identity,
    /// Platt scaling: `1 / (1 + exp(slope * x + intercept))`. Available
    /// for future use; not on the live path today.
    Platt { slope: f64, intercept: f64 },
}

impl Default for Calibrator {
    fn default() -> Self {
        Calibrator::Identity
    }
}

impl Calibrator {
    /// Apply calibration to a raw probability. Always clips to
    /// [eps, 1-eps] to avoid log(0) downstream.
    pub fn apply(&self, p: f64) -> f64 {
        let eps = 1.0e-9;
        let q = match *self {
            Calibrator::Identity => p,
            Calibrator::Platt { slope, intercept } => {
                let logit = slope * p + intercept;
                1.0 / (1.0 + (-logit).exp())
            }
        };
        q.clamp(eps, 1.0 - eps)
    }
}

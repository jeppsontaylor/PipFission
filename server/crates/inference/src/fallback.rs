//! Fallback predictor. The existing online logistic regression in the
//! `strategy` crate stays loaded so the live engine never goes silent
//! when the ONNX champion fails to load or its predictions are stale.
//!
//! For now this is a deterministic neutral predictor: it emits 0.5
//! probabilities for every bar so the trader's thresholds will pass on
//! it. When the strategy crate exposes its `OnlineLogReg` predict
//! method publicly we can swap this out for a real wrapper without
//! changing the trait surface — the registry already handles the swap.

use crate::{Predictor, Probs};

pub const FALLBACK_ID: &str = "fallback-neutral";

#[derive(Clone, Copy, Debug)]
pub struct FallbackPredictor {
    n_features: usize,
}

impl FallbackPredictor {
    pub fn new(n_features: usize) -> Self {
        Self { n_features }
    }
}

impl Predictor for FallbackPredictor {
    fn predict(&self, _features: &[f64]) -> Probs {
        // Neutral probabilities — trader thresholds will skip every bar.
        // Better than emitting noise on a faulty / missing model.
        Probs {
            p_long: 0.5,
            p_short: 0.5,
            p_take: 0.5,
            calibrated: 0.5,
        }
    }

    fn n_features(&self) -> usize {
        self.n_features
    }

    fn id(&self) -> &str {
        FALLBACK_ID
    }
}

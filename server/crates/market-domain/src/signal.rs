//! Strategy signals + ML fitness wire types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Direction the model predicts for the next forecast horizon.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalDirection {
    Long,
    Flat,
    Short,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategySignal {
    pub instrument: String,
    pub time: DateTime<Utc>,
    pub direction: SignalDirection,
    /// Probability of the chosen class (0.33 ≈ uniform; 1.0 = certain).
    pub confidence: f64,
    /// Per-class probabilities in order [long, flat, short].
    pub probs: [f64; 3],
    pub model_id: String,
    pub model_version: u32,
}

/// One row of fitness metrics on a contiguous window (train or OOS).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FitnessMetrics {
    pub samples: usize,
    pub accuracy: f64,
    pub log_loss: f64,
    /// Directional sharpe of `sign(prediction) * forward_return`. May be 0
    /// for the train window in v1 (we only score OOS).
    pub sharpe: f64,
    /// Per-class confusion: [predicted_long_count, predicted_flat_count,
    /// predicted_short_count]. For sanity-check.
    pub class_distribution: [usize; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFitness {
    pub instrument: String,
    pub model_id: String,
    pub model_version: u32,
    pub trained_at: DateTime<Utc>,
    pub train: FitnessMetrics,
    pub oos: FitnessMetrics,
    /// Number of samples seen total for this instrument.
    pub samples_seen: usize,
    /// `[train_start, train_end)` and `[oos_start, oos_end)` sample indexes.
    pub train_window: (usize, usize),
    pub oos_window: (usize, usize),
}

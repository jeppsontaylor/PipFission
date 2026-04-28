//! Champion-model live signals + hot-swap notifications. Runs in
//! parallel with the legacy `StrategySignal` so dashboards can show
//! both, or compare them side-by-side during the rollout window.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Per-bar prediction from the currently-loaded ONNX champion (or the
/// fallback predictor when no champion is loaded). Emitted on the
/// broadcast bus by the `live-inference` crate; persisted to the
/// `signals` table by the writer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChampionSignal {
    pub instrument: String,
    pub time: DateTime<Utc>,
    pub p_long: f64,
    pub p_short: f64,
    pub p_take: f64,
    pub calibrated: f64,
    pub model_id: String,
    /// `"onnx"` or `"fallback"`. Lets the dashboard render different
    /// chips for the two predictor families without parsing model_id.
    pub kind: String,
}

/// Emitted whenever the inference registry swaps to a new champion.
/// Surfaced on the WebSocket so the dashboard can show a banner.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChampionChanged {
    pub model_id: String,
    pub n_features: usize,
    pub kind: String,
}

/// Emitted when a champion-load attempt fails. Useful for the
/// dashboard's "champion is stale" warning.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChampionLoadFailed {
    pub reason: String,
}

/// Live decision emitted by the trader state machine running on top of
/// the champion's predictions. Distinct from the legacy `Signal`
/// pipeline — this is the bar-level, risk-gated, state-machine-driven
/// decision flow that the user wants for production.
///
/// `action` is one of `"open_long"`, `"open_short"`, `"close"`,
/// `"skip"`. `realized_r` is `Some` only on close events.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraderDecision {
    pub instrument: String,
    pub time: chrono::DateTime<chrono::Utc>,
    pub bar_idx: u32,
    pub action: String,
    pub reason: String,
    /// Entry / exit price.
    pub price: f64,
    /// Net realised return after costs. Only set on close events.
    pub realized_r: Option<f64>,
    /// Currently-active TraderParams id (links back to `trader_metrics`).
    pub params_id: String,
    /// Currently-active model id (links back to `model_metrics` /
    /// `model_artifacts`).
    pub model_id: String,
}

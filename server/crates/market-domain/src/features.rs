//! Feature vector emitted by `feature-engine` for each tick once the warmup
//! period has been reached.
//!
//! Wire shape: serializes as `{ instrument, time, version, vector }`. The
//! `vector` is a fixed-length array; the names are documented in
//! `FEATURE_NAMES` for both wire consumers and golden-test assertions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Number of features emitted per tick. Must match the length of
/// [`FEATURE_NAMES`]. Bumped to 24 in Sprint 2 / A2 to add 6 orderbook
/// features. Forex pairs (no orderbook) get zeros in slots 18..24.
pub const FEATURE_DIM: usize = 24;

/// Stable, ordered names of the features. Index = position in the
/// [`FeatureVector::vector`] array.
pub const FEATURE_NAMES: [&str; FEATURE_DIM] = [
    // 0..18: L1-derived features (forex + crypto)
    "log_return_1tick",
    "log_return_10tick",
    "log_return_60tick",
    "realized_vol_60tick",
    "realized_vol_300tick",
    "spread_bp",
    "spread_zscore_300tick",
    "mid_minus_sma60_over_mid",
    "mid_minus_sma300_over_mid",
    "rsi_14_on_60tick_bars",
    "bollinger_pct_b_300tick",
    "atr_14_on_60tick_bars",
    "macd_line_60_300",
    "macd_signal_minus_line",
    "tick_velocity_60",
    "range_60tick_over_mid",
    "minute_of_hour_sin",
    "minute_of_hour_cos",
    // 18..24: orderbook features (crypto only — zeros for forex)
    "ob_top_imbalance",          // (bid_size - ask_size) / (bid_size + ask_size) at top of book
    "ob_depth_weighted_mid_bp",  // (depth-weighted mid - mid) / mid * 10_000
    "ob_bid_wall_ratio",         // max(bid_size) / mean(bid_size) over top 20
    "ob_ask_wall_ratio",         // max(ask_size) / mean(ask_size) over top 20
    "ob_cum_depth_10bp",         // sum of sizes within ±10bp of mid (both sides)
    "ob_cum_depth_50bp",         // sum of sizes within ±50bp of mid (both sides)
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureVector {
    pub instrument: String,
    pub time: DateTime<Utc>,
    /// Schema version. Bump when the feature set changes.
    pub version: u32,
    pub vector: [f64; FEATURE_DIM],
}

impl FeatureVector {
    pub fn name_at(idx: usize) -> Option<&'static str> {
        FEATURE_NAMES.get(idx).copied()
    }
}

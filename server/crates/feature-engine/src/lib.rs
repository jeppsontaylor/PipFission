//! Feature engine: tick → 18-feature vector with incremental O(1) rolling state.
//!
//! See `market_domain::features::FEATURE_NAMES` for the canonical ordered
//! list. Per-instrument state lives in [`incremental::IncrementalFeatures`];
//! the bus runner in [`runner`] subscribes to `Event::Price` and emits
//! `Event::Features` once warmup (300 ticks) has completed.

#![deny(unsafe_code)]

pub mod incremental;
pub mod indicators;
pub mod runner;

pub use incremental::IncrementalFeatures;
pub use runner::spawn;

/// Number of ticks each instrument's state must observe before emitting
/// non-NaN features. Picked at >= max indicator window (300-tick spread
/// z-score / SMA / Bollinger).
pub const WARMUP_TICKS: usize = 300;

pub const FEATURE_VECTOR_DIM: usize = market_domain::FEATURE_DIM;

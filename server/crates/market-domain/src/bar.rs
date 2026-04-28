//! 10-second OHLCV bars produced by the bar-aggregator.
//!
//! Bars are the trading timeframe for the new ML pipeline:
//! triple-barrier labels, classifier inputs, and trader decisions all
//! operate on `Bar10s`. Raw `PriceTick`s remain the source of truth and
//! never leave `price_ticks`; bars are derived and persisted to
//! `bars_10s` for downstream consumers.

use serde::{Deserialize, Serialize};

/// One closed 10-second OHLCV bar.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Bar10s {
    pub instrument_id: u32,
    /// Bar-close timestamp in ms-since-epoch (the last tick to fall in
    /// the [t, t+10s) window stamps the bar; bars older than this are
    /// closed and emitted).
    pub ts_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    /// Number of ticks that contributed to the bar. Volume proxy when
    /// real volume is unavailable (forex).
    pub n_ticks: u32,
    /// Mean spread in basis points across the bar's ticks.
    pub spread_bp_avg: f64,
}

/// Wire-friendly variant carrying the instrument symbol. The bar bus
/// uses the int-keyed form for cheap dispatch; persistence and
/// dashboard surfaces use this string-keyed form.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bar10sNamed {
    pub instrument: String,
    pub ts_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub n_ticks: u32,
    pub spread_bp_avg: f64,
}

impl Bar10sNamed {
    pub fn from_bar(bar: Bar10s, instrument: String) -> Self {
        Self {
            instrument,
            ts_ms: bar.ts_ms,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            n_ticks: bar.n_ticks,
            spread_bp_avg: bar.spread_bp_avg,
        }
    }
}

/// Bucket size in milliseconds. 10s = 10_000ms.
pub const BAR_INTERVAL_MS: i64 = 10_000;

/// Floor `ts_ms` to its 10-second bucket boundary.
pub fn bucket_floor(ts_ms: i64) -> i64 {
    ts_ms - ts_ms.rem_euclid(BAR_INTERVAL_MS)
}

//! DuckDB-backed persistence for the live trading server.
//!
//! Migrated from rusqlite/SQLite to DuckDB for better analytical-query
//! throughput (the Python research layer reads this database directly via
//! its `duckdb` Python binding) and better concurrent-read behaviour.
//!
//! Design:
//! - One writer task owns the write-side connection. Reads share the same
//!   connection through a `parking_lot::Mutex` (DuckDB releases the lock
//!   internally during query execution, so the bottleneck is fine for our
//!   row volumes — at most 10k per instrument per table).
//! - Time is stored as `BIGINT` ms-since-epoch. The dashboard handles
//!   conversion at the wire boundary.
//! - HARD per-instrument FIFO cap of 10_000 rows on every time-series
//!   table. Locked by user requirement; never exceeded.
//! - No persisted derivative features. The strategy / Python research
//!   layer recomputes features from `price_ticks` + `bars_10s` on demand.
//!
//! What's persisted:
//! - `price_ticks`        every PriceTick (raw, source-of-truth)
//! - `bars_10s`           closed 10s OHLCV bars (trading timeframe)
//! - `labels`             ideal buy/sell entries from the label optimizer
//! - `oof_predictions`    out-of-fold classifier probabilities
//! - `signals`            live signals from the deployed champion
//! - `paper_fills`        audit trail (append-only, not capped)
//! - `trade_ledger`       round-trip trades (append-only, not capped) — read via `recent_trade_ledger`
//! - `model_metrics`      classifier scoring (append-only, not capped)
//! - `trader_metrics`     trader-param scoring (append-only, not capped)
//! - `optimizer_trials`   every Optuna trial (append-only, not capped)
//! - `lockbox_results`    sealed 100-bar holdout (write-once)
//! - `model_artifacts`    champion ONNX bytes (append-only, not capped)
//! - `fitness`            legacy logreg fitness (kept for the fallback)
//! - `account_snapshots`  every OANDA account poll (small)

#![deny(unsafe_code)]

mod connection;
mod dashboard;
mod retention;
mod schema;
mod writer;

pub use connection::Db;
pub use dashboard::{
    ChampionSignalRow, LabelRow, LockboxRow, ModelCandidateRow, ModelDeploymentGateRow,
    ModelMetricsRow, OptimizerTrialRow, PipelineRunRow, TradeLedgerInsert, TradeLedgerRow,
    TraderMetricsRow,
};
pub use retention::{spawn_rolloff, RetentionPolicy};
pub use writer::spawn_writer;

use serde::Serialize;

/// How many recent rows the api-server tries to hydrate at startup.
/// Sized to comfortably cover the strategy's MAX_BUFFER (8000). Bounded
/// by the HARD per-instrument cap (10_000).
pub const HYDRATE_FEATURES_PER_INSTRUMENT: usize = 8_000;

/// Hard cap on /api/history `limit` so a malformed query can't slam the
/// DB. Matches the per-instrument FIFO cap.
pub const MAX_HISTORY_LIMIT: usize = 10_000;

/// Re-export of the per-instrument row cap for downstream telemetry.
pub use schema::MAX_ROWS_PER_INSTRUMENT;

/// Compact wire shape for a single historical price tick. Time is sent
/// as integer ms-since-epoch so the dashboard can pass it straight to
/// `Date.parse` and recharts without further parsing.
#[derive(Clone, Debug, Serialize)]
pub struct PriceHistoryPoint {
    pub time_ms: i64,
    pub bid: f64,
    pub ask: f64,
    pub mid: f64,
    pub spread: f64,
    pub status: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SignalHistoryPoint {
    pub time_ms: i64,
    pub direction: String,
    pub confidence: f64,
    pub prob_long: f64,
    pub prob_flat: f64,
    pub prob_short: f64,
    pub model_id: String,
    pub model_version: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct FillHistoryPoint {
    pub instrument: String,
    pub time_ms: i64,
    pub units: i64,
    pub price: f64,
    pub fee: f64,
    pub mode: String,
    pub order_id: String,
}

//! Order types: intents, fills, routing modes, and inbound client commands.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Routing mode for outbound orders.
///
/// `Internal` — strategy's `OrderIntent`s are filled by the in-process
/// `PaperRouter` only; nothing is sent to OANDA.
///
/// `OandaPractice` — intents are sent to BOTH the internal `PaperRouter`
/// and OANDA's practice account, so the dashboard can compare them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingMode {
    #[default]
    Internal,
    OandaPractice,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderIntent {
    pub instrument: String,
    /// Signed units (positive = buy, negative = sell).
    pub units: i64,
    pub time: DateTime<Utc>,
    /// Model that produced this intent (for traceability).
    pub model_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaperFillEvent {
    pub mode: RoutingMode,
    pub order_id: String,
    pub instrument: String,
    pub units: i64,
    pub price: f64,
    pub fee: f64,
    pub time: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaperPosition {
    pub instrument: String,
    pub units: i64,
    pub avg_price: f64,
    pub realized: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaperBookSnapshot {
    pub mode: RoutingMode,
    /// Monotonic version stamp so the UI can detect lossy resync.
    pub version: u64,
    pub time: DateTime<Utc>,
    pub cash: f64,
    pub equity: f64,
    pub realized_pl: f64,
    pub unrealized_pl: f64,
    pub positions: Vec<PaperPosition>,
}

/// Reconciliation event: emitted in OandaPractice mode comparing the
/// internal paper book against OANDA's reported state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reconciliation {
    pub time: DateTime<Utc>,
    pub mode: RoutingMode,
    pub internal_paper_equity: f64,
    pub oanda_actual_equity: f64,
    /// internal - oanda, in basis points of OANDA NAV.
    pub oanda_minus_internal_bp: f64,
}

/// Inbound command from the WS client.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientCommand {
    SetMode { mode: RoutingMode },
    /// Place a manual order. Routes through the same per-venue dispatch
    /// as strategy-emitted intents — the only difference is provenance.
    /// `units` is signed (positive = buy, negative = sell). For Alpaca
    /// crypto the router floors at $10 notional regardless of units.
    ManualOrder { instrument: String, units: i64 },
}

/// Server's reply to a SetMode command.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModeAck {
    pub mode: RoutingMode,
    pub effective_at: DateTime<Utc>,
    /// `Some(reason)` if the requested mode was rejected.
    pub error: Option<String>,
}

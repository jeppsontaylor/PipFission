//! Account snapshot wire type. Serializes byte-identically to the pre-M1 shape.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccountSnapshot {
    pub time: DateTime<Utc>,
    pub nav: f64,
    pub balance: f64,
    pub unrealized_pl: f64,
    pub realized_pl: f64,
    pub margin_used: f64,
    pub margin_available: f64,
    pub open_position_count: i64,
    pub open_trade_count: i64,
    pub pending_order_count: i64,
    pub leverage: f64,
    pub currency: String,
}

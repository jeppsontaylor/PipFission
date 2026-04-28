//! Estimate tick: estimated vs actual NAV with drift.
//! Serializes byte-identically to the pre-M1 shape.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EstimateTick {
    pub time: DateTime<Utc>,
    /// What our internal book thinks the account is worth right now (mark-to-market
    /// against the latest mid for each open position).
    pub estimated_balance: f64,
    /// What OANDA's account summary said the NAV was at the last poll.
    pub actual_balance: f64,
    /// estimated - actual.
    pub drift: f64,
    /// drift / actual_balance, in bps. Useful for spotting slippage / fees.
    pub drift_bps: f64,
}

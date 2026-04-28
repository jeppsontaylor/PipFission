//! Snapshot wire type returned on /api/state and inside the WS Hello.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::account::AccountSnapshot;
use crate::conn::ConnStatus;
use crate::estimate::EstimateTick;
use crate::price::PriceTick;
use crate::transaction::TransactionEvent;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub server_time: DateTime<Utc>,
    pub prices: Vec<PriceTick>,
    pub account: Option<AccountSnapshot>,
    pub estimate: Option<EstimateTick>,
    pub recent_transactions: Vec<TransactionEvent>,
    pub connections: ConnStatus,
    /// Alpaca paper account snapshot (cash, equity, buying_power).
    /// `None` until ALPACA_KEY/ALPACA_SECRET are configured AND the
    /// first poll succeeds.
    #[serde(default)]
    pub alpaca: Option<AlpacaAccountSnapshot>,
}

/// Subset of Alpaca's `/v2/account` payload that we care about for the
/// dashboard. Field names match Alpaca's JSON exactly so we can json
/// the venue-specific snapshot through without rewriting.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlpacaAccountSnapshot {
    pub time: DateTime<Utc>,
    pub equity: f64,
    pub cash: f64,
    pub buying_power: f64,
    pub portfolio_value: f64,
    pub status: String,
    pub currency: String,
    pub long_market_value: f64,
    pub short_market_value: f64,
}

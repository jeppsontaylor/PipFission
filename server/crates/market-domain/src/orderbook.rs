//! Order book wire types. Top-N depth snapshots used by venues that
//! provide them (Alpaca crypto today; OANDA via a separate REST endpoint
//! later in M7).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// One side of a depth snapshot. Levels are sorted best-first
/// (highest bid, lowest ask). Each `(price, size)` pair.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OrderBookSide {
    pub levels: Vec<(f64, f64)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub instrument: String,
    pub time: DateTime<Utc>,
    pub bids: OrderBookSide,
    pub asks: OrderBookSide,
}

impl OrderBookSnapshot {
    pub fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids.levels.first().copied()
    }
    pub fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks.levels.first().copied()
    }
    pub fn mid(&self) -> Option<f64> {
        let (b, _) = self.best_bid()?;
        let (a, _) = self.best_ask()?;
        Some((b + a) / 2.0)
    }
}

//! Append-only ring buffer of (time, value) pairs.

use std::collections::VecDeque;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Bound on how many points we keep in memory for sparklines / chart history
/// per series.
pub const HISTORY_LIMIT: usize = 2000;

/// Bound on transaction log retention.
pub const TRANSACTION_LIMIT: usize = 500;

/// Capacity of the broadcast channel. Slow WS clients that fall behind will
/// be lagged out and resync'd via a fresh Hello.
pub const BROADCAST_CAPACITY: usize = 4096;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct History {
    pub points: VecDeque<HistoryPoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryPoint {
    pub time: DateTime<Utc>,
    pub value: f64,
}

impl History {
    pub fn push(&mut self, time: DateTime<Utc>, value: f64) {
        if self.points.len() >= HISTORY_LIMIT {
            self.points.pop_front();
        }
        self.points.push_back(HistoryPoint { time, value });
    }
}

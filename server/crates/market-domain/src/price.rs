//! Price tick wire type. Serializes byte-identically to the pre-M1 shape.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PriceTick {
    pub instrument: String,
    pub time: DateTime<Utc>,
    pub bid: f64,
    pub ask: f64,
    pub mid: f64,
    pub spread: f64,
    pub closeout_bid: Option<f64>,
    pub closeout_ask: Option<f64>,
    pub status: Option<String>,
}

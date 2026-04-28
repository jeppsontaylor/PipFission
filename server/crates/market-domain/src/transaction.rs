//! Transaction event wire type. Serializes byte-identically to the pre-M1 shape.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransactionEvent {
    pub id: String,
    pub time: DateTime<Utc>,
    pub kind: String,
    pub instrument: Option<String>,
    pub units: Option<f64>,
    pub price: Option<f64>,
    pub pl: Option<f64>,
    pub reason: Option<String>,
    pub raw: serde_json::Value,
}

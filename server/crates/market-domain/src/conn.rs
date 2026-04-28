//! Connection / stream health wire types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ConnStatus {
    pub pricing_stream: StreamHealth,
    pub transaction_stream: StreamHealth,
    pub account_poll: StreamHealth,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StreamHealth {
    pub connected: bool,
    pub last_message: Option<DateTime<Utc>>,
    pub messages_received: u64,
    pub reconnects: u64,
    pub last_error: Option<String>,
}

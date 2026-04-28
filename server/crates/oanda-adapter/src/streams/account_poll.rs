//! Polls /v3/accounts/{id}/summary at a fast interval (default 250 ms).
//! OANDA's REST limit is ~100 req/s per token, so 4 Hz uses ~4% of budget.
//!
//! Emits [`AccountMessage`] into the supplied sink.

use std::time::Duration;

use chrono::Utc;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;
use tokio::time::{interval, MissedTickBehavior};

use market_domain::AccountSnapshot;

use crate::client::Client;

#[derive(Clone, Debug)]
pub enum AccountMessage {
    Snapshot(AccountSnapshot),
    Error(String),
}

pub fn spawn(
    client: Client,
    account_id: String,
    poll_ms: u64,
    sink: UnboundedSender<AccountMessage>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move { run(client, account_id, poll_ms, sink).await })
}

async fn run(
    client: Client,
    account_id: String,
    poll_ms: u64,
    sink: UnboundedSender<AccountMessage>,
) {
    let period = Duration::from_millis(poll_ms);
    let mut tick = interval(period);
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    loop {
        tick.tick().await;
        if sink.is_closed() {
            tracing::info!("account sink closed; account_poll task exiting");
            return;
        }
        match client.account_summary(&account_id).await {
            Ok(v) => {
                if let Some(snap) = parse_summary(&v) {
                    let _ = sink.send(AccountMessage::Snapshot(snap));
                } else {
                    tracing::debug!("could not parse account summary: {v}");
                }
            }
            Err(e) => {
                tracing::warn!("account_summary failed: {e}");
                let _ = sink.send(AccountMessage::Error(e.to_string()));
            }
        }
    }
}

/// Parse OANDA's `/summary` payload into an [`AccountSnapshot`].
/// Public for unit tests.
pub fn parse_summary(v: &Value) -> Option<AccountSnapshot> {
    fn fnum(v: &Value, k: &str) -> f64 {
        v.get(k)
            .and_then(|x| x.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0)
    }
    fn inum(v: &Value, k: &str) -> i64 {
        v.get(k).and_then(|x| x.as_i64()).unwrap_or(0)
    }
    let currency = v
        .get("currency")
        .and_then(|x| x.as_str())
        .unwrap_or("USD")
        .to_string();
    let leverage = {
        let mr = fnum(v, "marginRate");
        if mr > 0.0 {
            (1.0 / mr).round()
        } else {
            0.0
        }
    };
    Some(AccountSnapshot {
        time: Utc::now(),
        nav: fnum(v, "NAV"),
        balance: fnum(v, "balance"),
        unrealized_pl: fnum(v, "unrealizedPL"),
        realized_pl: fnum(v, "pl"),
        margin_used: fnum(v, "marginUsed"),
        margin_available: fnum(v, "marginAvailable"),
        open_position_count: inum(v, "openPositionCount"),
        open_trade_count: inum(v, "openTradeCount"),
        pending_order_count: inum(v, "pendingOrderCount"),
        leverage,
        currency,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_summary_with_string_floats() {
        let raw = serde_json::json!({
            "currency": "USD",
            "NAV": "100123.45",
            "balance": "100100.10",
            "unrealizedPL": "23.35",
            "pl": "100.10",
            "marginUsed": "200.00",
            "marginAvailable": "99800.00",
            "openPositionCount": 1,
            "openTradeCount": 1,
            "pendingOrderCount": 0,
            "marginRate": "0.05"
        });
        let snap = parse_summary(&raw).expect("should parse");
        assert_eq!(snap.currency, "USD");
        assert!((snap.nav - 100_123.45).abs() < 1e-6);
        assert!((snap.balance - 100_100.10).abs() < 1e-6);
        assert!((snap.unrealized_pl - 23.35).abs() < 1e-6);
        assert_eq!(snap.open_position_count, 1);
        // marginRate 0.05 => leverage = 20 (1/0.05)
        assert!((snap.leverage - 20.0).abs() < 1e-9);
    }

    #[test]
    fn missing_fields_default_to_zero_or_usd() {
        let raw = serde_json::json!({});
        let snap = parse_summary(&raw).expect("should parse");
        assert_eq!(snap.currency, "USD");
        assert_eq!(snap.nav, 0.0);
        assert_eq!(snap.leverage, 0.0); // marginRate=0 -> leverage 0
    }
}

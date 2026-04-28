//! Long-lived transactions stream from OANDA. Same NDJSON pattern as pricing.
//! Emits [`TransactionMessage`] into the supplied sink.

use bytes::BytesMut;
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;
use tokio::time::sleep;

use market_domain::TransactionEvent;

use crate::backoff::Backoff;
use crate::client::Client;

#[derive(Clone, Debug)]
pub enum TransactionMessage {
    Transaction(TransactionEvent),
    Heartbeat,
    Connected,
    Error(String),
    Reconnecting { attempt: u64 },
}

pub fn spawn(
    client: Client,
    account_id: String,
    sink: UnboundedSender<TransactionMessage>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move { run(client, account_id, sink).await })
}

async fn run(client: Client, account_id: String, sink: UnboundedSender<TransactionMessage>) {
    let mut backoff = Backoff::new();
    let mut attempt: u64 = 0;
    loop {
        tracing::info!("opening transactions stream");
        match client.transactions_stream(&account_id).await {
            Ok(mut stream) => {
                let _ = sink.send(TransactionMessage::Connected);
                backoff.reset();
                let mut buf = BytesMut::with_capacity(32 * 1024);
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            buf.extend_from_slice(bytes.as_ref());
                            while let Some(idx) = buf.iter().position(|&b| b == b'\n') {
                                let line = buf.split_to(idx + 1);
                                let line = &line[..line.len() - 1];
                                if line.is_empty() {
                                    continue;
                                }
                                handle_line(line, &sink);
                            }
                        }
                        Err(e) => {
                            tracing::warn!("transactions stream chunk error: {e}");
                            let _ = sink.send(TransactionMessage::Error(e.to_string()));
                            break;
                        }
                    }
                }
                tracing::warn!("transactions stream ended; reconnecting...");
            }
            Err(e) => {
                tracing::error!("transactions stream open failed: {e:#}");
                let _ = sink.send(TransactionMessage::Error(format!("{e:#}")));
            }
        }
        attempt = attempt.saturating_add(1);
        let _ = sink.send(TransactionMessage::Reconnecting { attempt });
        sleep(backoff.next_delay()).await;
        if sink.is_closed() {
            tracing::info!("tx sink closed; transactions stream task exiting");
            return;
        }
    }
}

fn handle_line(line: &[u8], sink: &UnboundedSender<TransactionMessage>) {
    let v: Value = match serde_json::from_slice(line) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!("skipping malformed tx line ({e})");
            return;
        }
    };
    let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("UNKNOWN");

    if kind == "HEARTBEAT" {
        let _ = sink.send(TransactionMessage::Heartbeat);
        return;
    }

    if let Some(ev) = parse_transaction(&v, kind) {
        let _ = sink.send(TransactionMessage::Transaction(ev));
    }
}

/// Parse a single OANDA transaction JSON message. Public for tests.
pub fn parse_transaction(v: &Value, kind: &str) -> Option<TransactionEvent> {
    let id = v
        .get("id")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let time_str = v.get("time").and_then(|x| x.as_str()).unwrap_or("");
    let time: DateTime<Utc> = DateTime::parse_from_rfc3339(time_str)
        .map(|d| d.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());

    let instrument = v
        .get("instrument")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());
    let units = v
        .get("units")
        .and_then(|x| x.as_str())
        .and_then(|s| s.parse::<f64>().ok());
    let price = v
        .get("price")
        .and_then(|x| x.as_str())
        .and_then(|s| s.parse::<f64>().ok());
    let pl = v
        .get("pl")
        .and_then(|x| x.as_str())
        .and_then(|s| s.parse::<f64>().ok());
    let reason = v
        .get("reason")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());

    Some(TransactionEvent {
        id,
        time,
        kind: kind.to_string(),
        instrument,
        units,
        price,
        pl,
        reason,
        raw: v.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_order_fill() {
        let raw = serde_json::json!({
            "id": "12345",
            "type": "ORDER_FILL",
            "time": "2026-04-23T15:00:00.000000000Z",
            "instrument": "EUR_USD",
            "units": "100",
            "price": "1.10010",
            "pl": "0.5",
            "reason": "MARKET_ORDER"
        });
        let ev = parse_transaction(&raw, "ORDER_FILL").expect("should parse");
        assert_eq!(ev.id, "12345");
        assert_eq!(ev.kind, "ORDER_FILL");
        assert_eq!(ev.instrument.as_deref(), Some("EUR_USD"));
        assert_eq!(ev.units, Some(100.0));
        assert_eq!(ev.price, Some(1.10010));
        assert_eq!(ev.pl, Some(0.5));
        assert_eq!(ev.reason.as_deref(), Some("MARKET_ORDER"));
    }

    #[test]
    fn parses_with_missing_optionals() {
        let raw = serde_json::json!({
            "id": "999",
            "type": "DAILY_FINANCING",
            "time": "2026-04-23T15:00:00Z",
            "pl": "0.01"
        });
        let ev = parse_transaction(&raw, "DAILY_FINANCING").expect("should parse");
        assert_eq!(ev.kind, "DAILY_FINANCING");
        assert_eq!(ev.instrument, None);
        assert_eq!(ev.units, None);
        assert_eq!(ev.price, None);
        assert_eq!(ev.pl, Some(0.01));
    }
}

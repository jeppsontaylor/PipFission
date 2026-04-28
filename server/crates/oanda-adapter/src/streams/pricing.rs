//! Long-lived pricing stream from OANDA. NDJSON: split on `\n`, parse each
//! line as `serde_json::Value`, dispatch on `type`, emit a [`PricingMessage`]
//! into the supplied sink.
//!
//! No knowledge of `AppState`. The api-server runs a forwarder that maps
//! these messages to bus events.

use bytes::BytesMut;
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;
use tokio::time::sleep;

use market_domain::PriceTick;

use crate::backoff::Backoff;
use crate::client::Client;

/// Output messages from the pricing stream task.
#[derive(Clone, Debug)]
pub enum PricingMessage {
    Price(PriceTick),
    Heartbeat,
    Connected,
    Error(String),
    /// Emitted just before the next backoff sleep. Carries the reconnect count.
    Reconnecting {
        attempt: u64,
    },
}

pub fn spawn(
    client: Client,
    account_id: String,
    instruments: Vec<String>,
    sink: UnboundedSender<PricingMessage>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move { run(client, account_id, instruments, sink).await })
}

async fn run(
    client: Client,
    account_id: String,
    instruments: Vec<String>,
    sink: UnboundedSender<PricingMessage>,
) {
    let mut backoff = Backoff::new();
    let mut attempt: u64 = 0;
    loop {
        tracing::info!(instruments = instruments.len(), "opening pricing stream");
        match client.pricing_stream(&account_id, &instruments).await {
            Ok(mut stream) => {
                let _ = sink.send(PricingMessage::Connected);
                backoff.reset();
                let mut buf = BytesMut::with_capacity(64 * 1024);
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
                            tracing::warn!("pricing stream chunk error: {e}");
                            let _ = sink.send(PricingMessage::Error(e.to_string()));
                            break;
                        }
                    }
                }
                tracing::warn!("pricing stream ended; reconnecting...");
            }
            Err(e) => {
                tracing::error!("pricing stream open failed: {e:#}");
                let _ = sink.send(PricingMessage::Error(format!("{e:#}")));
            }
        }
        attempt = attempt.saturating_add(1);
        let _ = sink.send(PricingMessage::Reconnecting { attempt });
        sleep(backoff.next_delay()).await;
        // If the receiver has been dropped, exit cleanly.
        if sink.is_closed() {
            tracing::info!("pricing sink closed; pricing stream task exiting");
            return;
        }
    }
}

fn handle_line(line: &[u8], sink: &UnboundedSender<PricingMessage>) {
    let v: Value = match serde_json::from_slice(line) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(
                "skipping malformed pricing line ({e}): {}",
                String::from_utf8_lossy(line)
            );
            return;
        }
    };
    let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match kind {
        "PRICE" => {
            if let Some(tick) = parse_price(&v) {
                let _ = sink.send(PricingMessage::Price(tick));
            }
        }
        "HEARTBEAT" => {
            let _ = sink.send(PricingMessage::Heartbeat);
        }
        _ => {
            tracing::trace!("unknown pricing message type: {kind}");
        }
    }
}

/// Parse a single OANDA `PRICE` JSON message into a [`PriceTick`].
/// Public for unit + golden tests.
pub fn parse_price(v: &Value) -> Option<PriceTick> {
    let instrument = v.get("instrument")?.as_str()?.to_string();
    let time_str = v.get("time")?.as_str()?;
    let time: DateTime<Utc> = DateTime::parse_from_rfc3339(time_str)
        .ok()?
        .with_timezone(&Utc);

    // Best bid/ask are in the first elements of bids/asks arrays.
    let bid = v
        .get("bids")?
        .as_array()?
        .first()?
        .get("price")?
        .as_str()?
        .parse::<f64>()
        .ok()?;
    let ask = v
        .get("asks")?
        .as_array()?
        .first()?
        .get("price")?
        .as_str()?
        .parse::<f64>()
        .ok()?;
    let closeout_bid = v
        .get("closeoutBid")
        .and_then(|x| x.as_str())
        .and_then(|s| s.parse().ok());
    let closeout_ask = v
        .get("closeoutAsk")
        .and_then(|x| x.as_str())
        .and_then(|s| s.parse().ok());
    let status = v
        .get("status")
        .and_then(|s| s.as_str())
        .map(|s| s.to_string());

    let mid = (bid + ask) / 2.0;
    let spread = ask - bid;

    Some(PriceTick {
        instrument,
        time,
        bid,
        ask,
        mid,
        spread,
        closeout_bid,
        closeout_ask,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_price_extracts_bid_ask_mid_spread() {
        let raw = serde_json::json!({
            "type": "PRICE",
            "time": "2026-04-23T15:00:00.123456789Z",
            "instrument": "EUR_USD",
            "bids": [{"price": "1.10000"}],
            "asks": [{"price": "1.10020"}],
            "closeoutBid": "1.09995",
            "closeoutAsk": "1.10025",
            "status": "tradeable"
        });
        let tick = parse_price(&raw).expect("should parse");
        assert_eq!(tick.instrument, "EUR_USD");
        assert!((tick.bid - 1.10000).abs() < 1e-9);
        assert!((tick.ask - 1.10020).abs() < 1e-9);
        assert!((tick.mid - 1.10010).abs() < 1e-9);
        assert!((tick.spread - 0.00020).abs() < 1e-9);
        assert_eq!(tick.closeout_bid, Some(1.09995));
        assert_eq!(tick.status.as_deref(), Some("tradeable"));
    }

    #[test]
    fn parse_price_returns_none_on_missing_fields() {
        let raw = serde_json::json!({"type": "PRICE", "instrument": "EUR_USD"});
        assert!(parse_price(&raw).is_none());
    }

    #[test]
    fn parse_price_returns_none_on_unparseable_price() {
        let raw = serde_json::json!({
            "type": "PRICE",
            "time": "2026-04-23T15:00:00Z",
            "instrument": "EUR_USD",
            "bids": [{"price": "not-a-number"}],
            "asks": [{"price": "1.10020"}],
        });
        assert!(parse_price(&raw).is_none());
    }
}

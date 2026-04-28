//! Forwarder for the Alpaca crypto data feed.
//!
//! Owns the Alpaca WS task's receiver and translates each `AlpacaMessage`
//! into AppState updates + bus events. Mirrors the pattern in
//! `runners::pricing_forwarder`.

use std::sync::Arc;

use chrono::Utc;
use tokio::sync::mpsc;

use alpaca_adapter::AlpacaMessage;
use market_domain::Event;

use crate::state::AppState;

pub fn spawn_forwarder(
    state: Arc<AppState>,
    mut rx: mpsc::UnboundedReceiver<AlpacaMessage>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            match msg {
                AlpacaMessage::Price(tick) => {
                    state.touch_conn(|c| {
                        c.pricing_stream.last_message = Some(Utc::now());
                        c.pricing_stream.messages_received =
                            c.pricing_stream.messages_received.saturating_add(1);
                        c.pricing_stream.connected = true;
                    });
                    state.record_price(tick);
                }
                AlpacaMessage::OrderBook(ob) => {
                    let _ = state.bus.send(Event::OrderBook(ob));
                }
                AlpacaMessage::Trade { .. } => {
                    // Trades aren't surfaced today — quotes give us the
                    // live price already; trade tape will become useful
                    // when we add execution analytics later.
                }
                AlpacaMessage::Connected => {
                    tracing::info!("alpaca: stream connected + subscribed");
                }
                AlpacaMessage::Disconnected(err) => {
                    tracing::warn!(%err, "alpaca: stream disconnected");
                }
                AlpacaMessage::Reconnecting { attempt } => {
                    tracing::info!(attempt, "alpaca: reconnecting");
                }
            }
        }
    })
}

/// Periodic GET /v2/account against Alpaca paper. Updates AppState's
/// dedicated `alpaca_account` slot and broadcasts a status event.
pub fn spawn_account_poller(
    state: Arc<AppState>,
    router: Arc<alpaca_adapter::AlpacaRouter>,
    poll_ms: u64,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_millis(poll_ms));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tick.tick().await;
            match router.fetch_account().await {
                Ok(v) => {
                    state.update_alpaca_account(v);
                }
                Err(e) => {
                    tracing::debug!("alpaca: account poll failed: {e:#}");
                }
            }
        }
    })
}

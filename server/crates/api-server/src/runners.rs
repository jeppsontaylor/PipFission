//! Forwarders that consume oanda-adapter stream messages and translate
//! them into AppState updates + bus broadcasts. Each runner owns one
//! `tokio::sync::mpsc::Receiver` from the corresponding stream task.

use std::sync::Arc;

use chrono::Utc;
use tokio::sync::mpsc;

use oanda_adapter::streams::{
    account_poll::AccountMessage, pricing::PricingMessage, transactions::TransactionMessage,
};

use crate::state::AppState;

pub fn pricing_forwarder(
    state: Arc<AppState>,
    mut rx: mpsc::UnboundedReceiver<PricingMessage>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            match msg {
                PricingMessage::Price(tick) => {
                    state.touch_conn(|c| {
                        c.pricing_stream.last_message = Some(Utc::now());
                        c.pricing_stream.messages_received =
                            c.pricing_stream.messages_received.saturating_add(1);
                        c.pricing_stream.connected = true;
                    });
                    state.record_price(tick);
                }
                PricingMessage::Heartbeat => {
                    state.touch_conn(|c| {
                        c.pricing_stream.last_message = Some(Utc::now());
                        c.pricing_stream.messages_received =
                            c.pricing_stream.messages_received.saturating_add(1);
                        c.pricing_stream.connected = true;
                    });
                }
                PricingMessage::Connected => {
                    state.touch_conn(|c| {
                        c.pricing_stream.connected = true;
                        c.pricing_stream.last_error = None;
                    });
                }
                PricingMessage::Error(e) => {
                    state.touch_conn(|c| {
                        c.pricing_stream.connected = false;
                        c.pricing_stream.last_error = Some(e);
                    });
                }
                PricingMessage::Reconnecting { attempt } => {
                    state.touch_conn(|c| {
                        c.pricing_stream.reconnects = attempt;
                    });
                }
            }
        }
    })
}

pub fn transaction_forwarder(
    state: Arc<AppState>,
    mut rx: mpsc::UnboundedReceiver<TransactionMessage>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            match msg {
                TransactionMessage::Transaction(ev) => {
                    state.touch_conn(|c| {
                        c.transaction_stream.last_message = Some(Utc::now());
                        c.transaction_stream.messages_received =
                            c.transaction_stream.messages_received.saturating_add(1);
                        c.transaction_stream.connected = true;
                    });
                    state.record_transaction(ev);
                }
                TransactionMessage::Heartbeat => {
                    state.touch_conn(|c| {
                        c.transaction_stream.last_message = Some(Utc::now());
                        c.transaction_stream.messages_received =
                            c.transaction_stream.messages_received.saturating_add(1);
                        c.transaction_stream.connected = true;
                    });
                }
                TransactionMessage::Connected => {
                    state.touch_conn(|c| {
                        c.transaction_stream.connected = true;
                        c.transaction_stream.last_error = None;
                    });
                }
                TransactionMessage::Error(e) => {
                    state.touch_conn(|c| {
                        c.transaction_stream.connected = false;
                        c.transaction_stream.last_error = Some(e);
                    });
                }
                TransactionMessage::Reconnecting { attempt } => {
                    state.touch_conn(|c| {
                        c.transaction_stream.reconnects = attempt;
                    });
                }
            }
        }
    })
}

pub fn account_forwarder(
    state: Arc<AppState>,
    mut rx: mpsc::UnboundedReceiver<AccountMessage>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            match msg {
                AccountMessage::Snapshot(snap) => {
                    state.touch_conn(|c| {
                        c.account_poll.connected = true;
                        c.account_poll.last_message = Some(Utc::now());
                        c.account_poll.messages_received =
                            c.account_poll.messages_received.saturating_add(1);
                        c.account_poll.last_error = None;
                    });
                    state.record_account(snap);
                }
                AccountMessage::Error(e) => {
                    state.touch_conn(|c| {
                        c.account_poll.connected = false;
                        c.account_poll.last_error = Some(e);
                    });
                }
            }
        }
    })
}

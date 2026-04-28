//! WebSocket handler. On connect, sends a `Hello` event with the full
//! current snapshot, then forwards every event from the broadcast bus.
//! Lagged clients are resync'd with a fresh Hello rather than disconnected.

use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::broadcast::error::RecvError;

use market_domain::Event;

use crate::state::AppState;

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle(socket, state))
}

async fn handle(socket: WebSocket, state: Arc<AppState>) {
    let (mut tx_ws, mut rx_ws) = socket.split();

    // 1. Send initial hello.
    let hello = Event::Hello {
        account_id: state.account_id.clone(),
        instruments: state.cfg.instruments.clone(),
        environment: state.cfg.environment.clone(),
        snapshot: Box::new(state.snapshot()),
    };
    if let Ok(json) = serde_json::to_string(&hello) {
        if tx_ws.send(Message::Text(json)).await.is_err() {
            return;
        }
    }

    // 2. Subscribe to the bus and forward.
    let mut rx_bus = state.bus.subscribe();

    loop {
        tokio::select! {
            recv = rx_bus.recv() => {
                match recv {
                    Ok(ev) => {
                        match serde_json::to_string(&ev) {
                            Ok(s) => {
                                if tx_ws.send(Message::Text(s)).await.is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                tracing::warn!("ws serialize error: {e}");
                            }
                        }
                    }
                    Err(RecvError::Lagged(n)) => {
                        tracing::warn!("ws client lagged {n} events; resyncing");
                        let resync = Event::Hello {
                            account_id: state.account_id.clone(),
                            instruments: state.cfg.instruments.clone(),
                            environment: state.cfg.environment.clone(),
                            snapshot: Box::new(state.snapshot()),
                        };
                        if let Ok(s) = serde_json::to_string(&resync) {
                            let _ = tx_ws.send(Message::Text(s)).await;
                        }
                    }
                    Err(RecvError::Closed) => break,
                }
            }
            incoming = rx_ws.next() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(p))) => {
                        let _ = tx_ws.send(Message::Pong(p)).await;
                    }
                    Some(Ok(Message::Text(t))) => {
                        crate::commands::handle_client_text(&state, &t);
                    }
                    Some(Ok(_)) => { /* binary/pong: ignore */ }
                    Some(Err(e)) => {
                        tracing::debug!("ws recv error: {e}");
                        break;
                    }
                }
            }
        }
    }
}

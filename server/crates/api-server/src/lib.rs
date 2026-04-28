//! api-server: axum HTTP + WebSocket orchestration.
//!
//! Wires the `oanda-adapter` channel-based stream tasks into `AppState`
//! and the broadcast bus, mounts the WS handler, and runs the estimator
//! tick loop.

#![deny(unsafe_code)]

pub mod alpaca_runner;
pub mod auto_retrain;
pub mod champion_router;
pub mod commands;
pub mod estimator;
pub mod http;
pub mod pipeline_trigger;
pub mod runners;
pub mod state;
pub mod ws;

pub use state::AppState;

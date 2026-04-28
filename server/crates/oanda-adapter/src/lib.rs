//! OANDA v20 REST + streaming adapter.
//!
//! Only crate that uses `reqwest`. Wraps:
//! - REST `GET /v3/accounts` (account discovery)
//! - REST `GET /v3/accounts/{id}/summary` (hot-poll, default 4 Hz)
//! - REST `POST /v3/accounts/{id}/orders` (place_market_order, M6 wires it
//!   into `OandaRouter`)
//! - Streaming `GET /v3/accounts/{id}/pricing/stream`
//! - Streaming `GET /v3/accounts/{id}/transactions/stream`
//!
//! Long-lived streams use a separate reqwest::Client with no timeout and
//! gzip disabled, so chunked NDJSON arrives line-aligned. Reconnects use
//! exponential backoff (250ms → 30s) via [`Backoff`].

#![deny(unsafe_code)]

pub mod backoff;
pub mod client;
pub mod replay;
pub mod router;
pub mod streams;
pub mod synthetic;

pub use router::OandaRouter;

pub use backoff::Backoff;
pub use client::{ByteStream, Client};

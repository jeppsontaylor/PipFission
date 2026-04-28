//! Alpaca Markets adapter — crypto data WebSocket + paper trading REST.
//!
//! - Data: `wss://stream.data.alpaca.markets/v1beta3/crypto/us`
//!   - Authenticated via JSON message after connect.
//!   - Subscribes to `trades`, `quotes`, `bars`, `orderbooks` for the
//!     configured symbols (BTC/USD, ETH/USD by default).
//! - Trading: `https://paper-api.alpaca.markets/v2`
//!   - Paper account; same instruments tradable, separate from OANDA.
//!
//! Sprint 2 milestones A1 (data) and A4 (router).

#![deny(unsafe_code)]

pub mod config;
pub mod router;
pub mod stream;

pub use config::AlpacaConfig;
pub use router::AlpacaRouter;
pub use stream::AlpacaMessage;
// Re-export market-domain's canonical OrderBookSnapshot so callers don't
// have to import from two places.
pub use market_domain::{OrderBookSide, OrderBookSnapshot};

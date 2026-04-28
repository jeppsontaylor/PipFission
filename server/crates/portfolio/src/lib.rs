//! Portfolio: synthetic paper book + OANDA-actual reconciler.
//!
//! ## Roles
//!
//! - [`PaperBook`] — VWAP-tracked positions per instrument, cash, and a
//!   mark-to-market estimator against a price oracle. Used in two places:
//!     1. As the "OANDA-actual reconciler" (M1): seeds cash from the first
//!        OANDA account snapshot, applies OANDA `ORDER_FILL` transactions,
//!        emits an [`EstimateTick`] every tick comparing estimated vs actual.
//!     2. As the "internal paper book" (M5): same logic, but fills come from
//!        a `PaperRouter` synthesizing fills against current best bid/ask.
//!
//! - [`reconciliation`] — 3-way reconciliation between StrategyExpected,
//!   InternalPaperActual, and OandaActual. Stub in M1; populated in M6.

#![deny(unsafe_code)]

pub mod paper_book;
pub mod reconciliation;
pub mod router;

pub use paper_book::{PaperBook, Position};
pub use router::{OrderRouter, PaperRouter, PriceOracle, RouterError};

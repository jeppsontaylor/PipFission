//! Deterministic trader state machine.
//!
//! Consumes per-bar `Probs` (from the inference crate or from OOF
//! parquet during research) and emits `TradeEvent`s. The same code runs
//! in the live engine and inside the backtester, so the policy that
//! gets optimised in research is exactly the policy that trades on
//! OANDA practice.
//!
//! The state machine is deterministic given `TraderParams`; the only
//! optional non-determinism comes from `RiskGates::stale_data` (clock
//! comparisons), which the backtester skips. Every transition writes
//! its `Reason` so the dashboard can show why a trade was taken or
//! skipped.

#![deny(unsafe_code)]

pub mod params;
pub mod risk;
pub mod state;

pub use params::{Probs, TraderParams};
pub use risk::{RiskGates, RiskOutcome};
pub use state::{Reason, Side, State, TradeEvent, Trader};

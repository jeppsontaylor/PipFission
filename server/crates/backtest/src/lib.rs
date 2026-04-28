//! Deterministic event-driven backtester.
//!
//! Inputs:
//!   - `bars: &[Bar10s]` — closed 10-second OHLCV bars in ascending time.
//!   - `probs: &[Probs]` — per-bar OOF probabilities aligned 1-1 with bars.
//!   - `sigma: &[f64]` — per-bar volatility (typically EWMA σ from the
//!     labeling crate).
//!   - `params: TraderParams` — what the optimiser tunes.
//!   - `costs: Costs` — commission + slippage assumptions applied at fill.
//!
//! Output:
//!   - `Report { summary: metrics::Summary, ledger: Vec<TradeRecord>, equity: Vec<f64> }`.
//!
//! Determinism: same `(bars, probs, sigma, params, costs)` → byte-identical
//! `Report`. The trader has no internal RNG; the only sequence-dependent
//! piece is `RiskGates::day_pnl_bp` which advances purely on realised P&L.

#![deny(unsafe_code)]

pub mod fill;
pub mod replay;

pub use fill::Costs;
pub use replay::{run_backtest, Report, TradeRecord};

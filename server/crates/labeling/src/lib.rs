//! Triple-barrier labeling, event sampling, and label optimisation.
//!
//! The labeling pipeline turns a `[Bar10s]` window into a set of
//! `LabelRow` records that the classifier trains against. The pipeline
//! deliberately runs only on the trailing 1000-bar window — the user's
//! locked invariant — to keep the optimiser cheap and to avoid hindsight
//! bias creeping in from earlier regimes.
//!
//! ## Stages
//! 1. **Volatility estimate** — EWMA of bar returns; barrier widths are
//!    `pt_atr × σ_t` and `sl_atr × σ_t`, so the barriers adapt to regime.
//! 2. **Event sampling** — CUSUM-vol filter selects candidate event bars.
//!    A breakout filter is also provided.
//! 3. **Triple-barrier labelling** — for each event bar, walk forward
//!    until upper barrier (long-side TP) or lower barrier (long-side SL)
//!    is hit, or the vertical horizon expires. The first-hit barrier and
//!    the realised return determine `(side, meta_y, t1_ms, realized_r)`.
//! 4. **Label optimiser** — constrained interval scheduler that picks a
//!    non-overlapping subset of `LabelRow`s maximising a regularised
//!    objective (Sortino - drawdown - turnover - imbalance).
//!
//! ## What's intentionally NOT in here
//! - Feature recompute. That stays in `feature-engine`.
//! - CV splitting. That's the `cv` crate.
//! - Backtesting. That's the `backtest` crate.
//!
//! Everything is pure-Rust; no I/O. The Python research layer calls into
//! these functions via PyO3 bindings (added in a later milestone).

#![deny(unsafe_code)]

pub mod events;
pub mod label_optimizer;
pub mod meta;
pub mod triple_barrier;
pub mod volatility;

pub use events::{breakout_events, cusum_filter, EventConfig};
pub use label_optimizer::{optimise_labels, LabelOptimiserConfig};
pub use meta::meta_label;
pub use triple_barrier::{triple_barrier, BarrierConfig, BarrierHit, LabelRow};
pub use volatility::{atr, ewma_volatility};

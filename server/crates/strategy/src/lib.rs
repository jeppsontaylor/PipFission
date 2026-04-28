//! Strategy crate: online **binary** logistic regression (long vs short),
//! walk-forward fitness, signal emission. Sprint 2 milestones M4 and M5.
//! Binary burn-down complete — flat is gone from the model layer; the
//! `SignalDirection::Flat` enum variant is kept on the wire only for
//! pre-burn-down signals already in the database.

#![deny(unsafe_code)]
// Numeric/ML code is more readable with explicit indices than with
// `iter_mut().enumerate()`. Crate-wide allow.
#![allow(clippy::needless_range_loop)]

pub mod intent_emitter;
pub mod online_logreg;
pub mod runner;
pub mod walk_forward;

pub use intent_emitter::IntentEmitter;
pub use online_logreg::{LogReg, LogRegConfig};
pub use runner::{spawn, spawn_with_prefill, MAX_BUFFER, RETRAIN_EVERY, TRAIN_AFTER};
pub use walk_forward::LABEL_HORIZON_TICKS;

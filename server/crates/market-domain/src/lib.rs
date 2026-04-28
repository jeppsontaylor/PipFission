//! Pure domain types for the OANDA live trading system.
//!
//! No I/O, no tokio, no reqwest. This crate is the leaf of the workspace
//! DAG and is depended on by every other crate.
//!
//! ## Conventions
//!
//! - Wire types (`PriceTick`, `AccountSnapshot`, `TransactionEvent`,
//!   `EstimateTick`, `Snapshot`, `ConnStatus`, `StreamHealth`, `History`)
//!   serialize byte-identically to what the dashboard expected before the
//!   M1 refactor. Do not change field names or shapes without bumping the
//!   wire schema and updating `dashboard/src/types.ts`.
//!
//! - Typed IDs (`Instrument`, `AccountId`, `OrderId`, `TransactionId`)
//!   use `#[serde(transparent)]` so they remain wire-compatible with the
//!   plain `String` / number representations.
//!
//! - Validated constructors live next to the type they construct.
//!   See `ids::Instrument::try_new` for the canonical example.

#![deny(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod account;
pub mod bar;
pub mod champion;
pub mod config;
pub mod conn;
pub mod errors;
pub mod estimate;
pub mod event;
pub mod features;
pub mod history;
pub mod ids;
pub mod order;
pub mod orderbook;
pub mod price;
pub mod signal;
pub mod snapshot;
pub mod transaction;

pub use account::AccountSnapshot;
pub use bar::{bucket_floor, Bar10s, Bar10sNamed, BAR_INTERVAL_MS};
pub use champion::{ChampionChanged, ChampionLoadFailed, ChampionSignal, TraderDecision};
pub use config::{Config, DEFAULT_INSTRUMENTS};
pub use conn::{ConnStatus, StreamHealth};
pub use errors::DomainError;
pub use estimate::EstimateTick;
pub use event::Event;
pub use features::{FeatureVector, FEATURE_DIM, FEATURE_NAMES};
pub use history::{History, HistoryPoint, BROADCAST_CAPACITY, HISTORY_LIMIT, TRANSACTION_LIMIT};
pub use ids::{AccountId, Instrument, OrderId, TransactionId, Units};
pub use order::{
    ClientCommand, ModeAck, OrderIntent, PaperBookSnapshot, PaperFillEvent, PaperPosition,
    Reconciliation, RoutingMode,
};
pub use orderbook::{OrderBookSide, OrderBookSnapshot};
pub use price::PriceTick;
pub use signal::{FitnessMetrics, ModelFitness, SignalDirection, StrategySignal};
pub use snapshot::{AlpacaAccountSnapshot, Snapshot};
pub use transaction::TransactionEvent;

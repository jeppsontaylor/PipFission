//! Bus / WebSocket event enum. Serializes byte-identically to the pre-M1 shape:
//! `{ "type": "snake_case", ... }` per `#[serde(tag = "type", rename_all = "snake_case")]`.

use serde::{Deserialize, Serialize};

use crate::account::AccountSnapshot;
use crate::bar::Bar10sNamed;
use crate::champion::{ChampionChanged, ChampionLoadFailed, ChampionSignal, TraderDecision};
use crate::conn::ConnStatus;
use crate::estimate::EstimateTick;
use crate::features::FeatureVector;
use crate::order::{ModeAck, PaperBookSnapshot, PaperFillEvent, Reconciliation};
use crate::orderbook::OrderBookSnapshot;
use crate::price::PriceTick;
use crate::signal::{ModelFitness, StrategySignal};
use crate::snapshot::Snapshot;
use crate::transaction::TransactionEvent;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// Initial state dump on connect. Snapshot is boxed so the enum stays
    /// small in memory (avoids `clippy::large_enum_variant`); serde
    /// serializes Box<T> transparently so the wire shape is unchanged.
    Hello {
        account_id: String,
        instruments: Vec<String>,
        environment: String,
        snapshot: Box<Snapshot>,
    },
    Price(PriceTick),
    Account(AccountSnapshot),
    Transaction(TransactionEvent),
    Estimate(EstimateTick),
    Status(ConnStatus),
    /// Per-tick feature vector. Emitted only after the per-instrument
    /// warmup window. Sprint 2 milestone M3.
    Features(FeatureVector),
    /// Per-tick model output: predicted next-30s direction and confidence.
    /// Emitted once a model has been trained for the instrument. Sprint 2 M4.
    Signal(StrategySignal),
    /// Model-retrain event: emitted whenever a fresh model lands. Carries
    /// honest train + OOS metrics. Sprint 2 M4.
    Fitness(ModelFitness),
    /// Internal paper-book snapshot. Streamed on every fill + on a tick.
    /// Boxed for size symmetry with Hello.
    PaperBook(Box<PaperBookSnapshot>),
    /// Single internal paper fill event.
    PaperFill(PaperFillEvent),
    /// 3-way reconciliation between internal paper book and OANDA actuals.
    /// Emitted only when in OandaPractice mode.
    Reconciliation(Reconciliation),
    /// Server's response to a `ClientCommand::SetMode`.
    ModeAck(ModeAck),
    /// Top-N depth snapshot for an instrument (provided by Alpaca crypto).
    OrderBook(OrderBookSnapshot),
    /// Closed 10-second OHLCV bar emitted by the bar-aggregator. The
    /// trading timeframe for the new ML pipeline.
    Bar10s(Bar10sNamed),
    /// Per-bar prediction from the currently-active ONNX champion (or
    /// the fallback predictor). Distinct from `Signal`, which carries
    /// the legacy logreg's tick-level outputs.
    ChampionSignal(ChampionSignal),
    /// New champion loaded; dashboard banner.
    ChampionChanged(ChampionChanged),
    /// Champion load attempt failed; dashboard warning.
    ChampionLoadFailed(ChampionLoadFailed),
    /// Live trader state-machine decision (open/close/skip) on a closed
    /// bar. Distinct from the legacy `Signal` so the existing intent
    /// runner doesn't double-route.
    TraderDecision(TraderDecision),
}

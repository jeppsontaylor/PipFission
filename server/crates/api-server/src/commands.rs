//! Inbound `ClientCommand` handler + outbound intent runner.
//!
//! Two responsibilities:
//!
//! 1. **Inbound:** parse a JSON `ClientCommand` from a WS text frame,
//!    apply it to AppState (`SetMode`), and emit a `ModeAck` event.
//!
//! 2. **Outbound (intent_runner):** subscribe to `Event::Signal`, run
//!    them through `IntentEmitter`, call the active router, apply the
//!    fill to the internal PaperBook, and broadcast `PaperFill` +
//!    `PaperBook` snapshots.
//!
//! Sprint 2 milestones M5 (Internal routing) and M6 (OANDA routing).

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use parking_lot::Mutex;
use tokio::sync::broadcast;
use tokio::time::{interval, MissedTickBehavior};

use market_domain::{
    ClientCommand, Event, ModeAck, OrderIntent, PaperBookSnapshot, PaperPosition, Reconciliation,
    RoutingMode,
};
use portfolio::OrderRouter;
use strategy::IntentEmitter;

use crate::state::AppState;

/// Coarse venue inference from instrument symbol. Used for per-instrument
/// router dispatch. Wrong only if the user adds an instrument whose name
/// doesn't follow either convention — currently impossible in v1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Venue {
    Oanda,
    Alpaca,
}

fn classify_venue(instrument: &str) -> Venue {
    if instrument.contains('/') {
        Venue::Alpaca
    } else if matches!(
        instrument,
        "BTC_USD" | "ETH_USD" | "SOL_USD" | "DOGE_USD" | "LTC_USD"
    ) {
        Venue::Alpaca
    } else {
        Venue::Oanda
    }
}

/// Choose the right router for a (mode, venue) pair, with a safe fallback
/// to the always-present PaperRouter.
fn pick_router(state: &Arc<AppState>, mode: RoutingMode, venue: Venue) -> Arc<dyn OrderRouter> {
    match mode {
        RoutingMode::Internal => Arc::clone(&state.router.load_full()),
        RoutingMode::OandaPractice => match venue {
            Venue::Oanda => state
                .oanda_router
                .clone()
                .unwrap_or_else(|| Arc::clone(&state.router.load_full())),
            Venue::Alpaca => state
                .alpaca_router
                .clone()
                .unwrap_or_else(|| Arc::clone(&state.router.load_full())),
        },
    }
}

pub fn handle_client_text(state: &Arc<AppState>, text: &str) {
    let cmd: ClientCommand = match serde_json::from_str(text) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("invalid ClientCommand JSON: {e}; raw={text:.200}");
            return;
        }
    };
    match cmd {
        ClientCommand::SetMode { mode } => set_mode(state, mode),
        ClientCommand::ManualOrder { instrument, units } => {
            // Spawn so we don't block the WS reader on the HTTP roundtrip.
            let st = state.clone();
            tokio::spawn(async move {
                manual_order(st, instrument, units).await;
            });
        }
    }
}

/// Fire a manual order through the same per-venue dispatch as strategy
/// signals. Convenience for testing the live-trading path before the ML
/// model has warmed up. Result is broadcast as a `PaperFill` event on
/// success or logged as a router rejection.
async fn manual_order(state: Arc<AppState>, instrument: String, units: i64) {
    if units == 0 {
        tracing::warn!("manual_order with zero units ignored");
        return;
    }
    let venue = classify_venue(&instrument);
    let active_mode = *state.mode.load_full();
    let router = pick_router(&state, active_mode, venue);
    let mode = router.mode();
    tracing::info!(
        instrument = %instrument,
        units,
        ?venue,
        ?mode,
        "manual_order: submitting"
    );
    let intent = OrderIntent {
        instrument: instrument.clone(),
        units,
        time: Utc::now(),
        model_id: "manual".into(),
    };
    match router.submit(intent).await {
        Ok(fill) => {
            {
                let mut pb = state.paper_book.lock();
                if !pb.is_seeded() {
                    pb.seed_cash(100_000.0);
                }
                pb.apply_paper_fill(&fill.instrument, fill.units, fill.price);
            }
            tracing::info!(
                instrument = %fill.instrument,
                units = fill.units,
                price = fill.price,
                ?mode,
                "manual_order: fill"
            );
            let _ = state.bus.send(Event::PaperFill(fill));
            broadcast_paper_book(&state, mode);
        }
        Err(e) => {
            tracing::warn!(error = %e, instrument = %instrument, units, "manual_order: rejected");
        }
    }
}

fn set_mode(state: &Arc<AppState>, requested: RoutingMode) {
    // OandaPractice requires the operator-set safety gate.
    let allow_oanda = std::env::var("ALLOW_OANDA_ROUTING").as_deref() == Ok("true");
    let (effective, error): (RoutingMode, Option<String>) = match requested {
        RoutingMode::Internal => (RoutingMode::Internal, None),
        RoutingMode::OandaPractice => {
            if allow_oanda && state.oanda_router.is_some() {
                (RoutingMode::OandaPractice, None)
            } else if !allow_oanda {
                (
                    *state.mode.load_full(),
                    Some("ALLOW_OANDA_ROUTING is false on the server".into()),
                )
            } else {
                (
                    *state.mode.load_full(),
                    Some("OANDA router not initialized".into()),
                )
            }
        }
    };
    state.mode.store(Arc::new(effective));
    // Swap the active router pointer.
    match effective {
        RoutingMode::Internal => {
            state.router.store(state.paper_router.clone());
        }
        RoutingMode::OandaPractice => {
            if let Some(o) = &state.oanda_router {
                state.router.store(Arc::new(Arc::clone(o)));
            }
        }
    }
    tracing::info!(?effective, ?requested, "mode set");
    let ack = ModeAck {
        mode: effective,
        effective_at: Utc::now(),
        error,
    };
    let _ = state.bus.send(Event::ModeAck(ack));
}

/// Spawn the intent runner. Subscribes to Signal events, gates them
/// through IntentEmitter, fires the active router, and broadcasts fills
/// + PaperBook snapshots.
pub fn spawn_intent_runner(state: Arc<AppState>) -> tokio::task::JoinHandle<()> {
    let emitter = Arc::new(Mutex::new(IntentEmitter::new()));
    let mut rx = state.bus.subscribe();
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(Event::Signal(sig)) => {
                    let intent = emitter.lock().on_signal(&sig);
                    let Some(intent) = intent else { continue };
                    let order_intent = OrderIntent {
                        instrument: intent.instrument.clone(),
                        units: intent.units_delta,
                        time: sig.time,
                        model_id: intent.triggered_by_model.clone(),
                    };
                    // Per-instrument routing dispatch:
                    // - Internal mode → PaperRouter for everything.
                    // - Live (OandaPractice) mode → infer venue from
                    //   instrument name. Crypto symbols (contain "/" or
                    //   start with BTC/ETH/SOL/...) route to AlpacaRouter;
                    //   forex (XXX_YYY) routes to OandaRouter. Fallback
                    //   to PaperRouter when the venue's router isn't wired.
                    let active_mode = *state.mode.load_full();
                    let venue = classify_venue(&order_intent.instrument);
                    let router = pick_router(&state, active_mode, venue);
                    let mode = router.mode();
                    let result = router.submit(order_intent).await;
                    match result {
                        Ok(fill) => {
                            // Apply to internal paper book.
                            {
                                let mut pb = state.paper_book.lock();
                                if !pb.is_seeded() {
                                    pb.seed_cash(100_000.0);
                                }
                                pb.apply_paper_fill(&fill.instrument, fill.units, fill.price);
                                let new_pos = pb
                                    .position(&fill.instrument)
                                    .map(|p| p.units as i64)
                                    .unwrap_or(0);
                                emitter.lock().set_position(&fill.instrument, new_pos);
                            }
                            tracing::info!(
                                instrument = %fill.instrument,
                                units = fill.units,
                                price = fill.price,
                                ?mode,
                                "paper fill"
                            );
                            let _ = state.bus.send(Event::PaperFill(fill));
                            broadcast_paper_book(&state, mode);
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "router rejected intent");
                        }
                    }
                }
                Ok(_) => {}
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("intent runner lagged {n} events");
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    })
}

/// Spawn a periodic snapshot emitter so the dashboard sees the paper book
/// even when no fills are happening (mark-to-market changes).
pub fn spawn_paper_book_ticker(state: Arc<AppState>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = interval(Duration::from_millis(500));
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            tick.tick().await;
            let mode = *state.mode.load_full();
            broadcast_paper_book(&state, mode);
            // M6: when in OandaPractice mode, also emit a reconciliation tick.
            if mode == RoutingMode::OandaPractice {
                broadcast_reconciliation(&state, mode);
            }
        }
    })
}

fn broadcast_reconciliation(state: &Arc<AppState>, mode: RoutingMode) {
    let internal_equity = {
        let pb = state.paper_book.lock();
        if !pb.is_seeded() {
            return;
        }
        let cash = pb.cash();
        let mut unrealized = 0.0;
        for (inst, pos) in pb.positions() {
            let mid = state
                .latest_prices
                .get(inst)
                .map(|p| p.mid)
                .unwrap_or(pos.avg_price);
            unrealized += pos.units * (mid - pos.avg_price);
        }
        cash + unrealized
    };
    let oanda_nav = match state.account.read().clone() {
        Some(a) => a.nav,
        None => return,
    };
    let bp = if oanda_nav != 0.0 {
        (oanda_nav - internal_equity) / oanda_nav * 10_000.0
    } else {
        0.0
    };
    let r = Reconciliation {
        time: Utc::now(),
        mode,
        internal_paper_equity: internal_equity,
        oanda_actual_equity: oanda_nav,
        oanda_minus_internal_bp: bp,
    };
    let _ = state.bus.send(Event::Reconciliation(r));
}

fn broadcast_paper_book(state: &Arc<AppState>, mode: RoutingMode) {
    let snap = {
        let pb = state.paper_book.lock();
        if !pb.is_seeded() {
            return;
        }
        let cash = pb.cash();
        // Mark every open position to market.
        let mut unrealized = 0.0;
        let mut positions: Vec<PaperPosition> = Vec::new();
        for (inst, pos) in pb.positions() {
            let units = pos.units as i64;
            let avg = pos.avg_price;
            let realized = pos.realized;
            let mid = state.latest_prices.get(inst).map(|p| p.mid).unwrap_or(avg);
            unrealized += (pos.units) * (mid - avg);
            positions.push(PaperPosition {
                instrument: inst.clone(),
                units,
                avg_price: avg,
                realized,
            });
        }
        let realized_pl: f64 = pb.positions().values().map(|p| p.realized).sum();
        let equity = cash + unrealized;
        let version = state.paper_book_version.fetch_add(1, Ordering::Relaxed) + 1;
        PaperBookSnapshot {
            mode,
            version,
            time: Utc::now(),
            cash,
            equity,
            realized_pl,
            unrealized_pl: unrealized,
            positions,
        }
    };
    let _ = state.bus.send(Event::PaperBook(Box::new(snap)));
}

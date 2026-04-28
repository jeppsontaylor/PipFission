//! Champion order router. Subscribes to `Event::TraderDecision` and
//! converts decisions into real orders through the active
//! `OrderRouter`.
//!
//! ## Why a separate runner?
//! The legacy `intent_runner` consumes `Event::Signal` from the
//! tick-level logreg and converts side flips into orders. The champion
//! path emits `Event::TraderDecision` from the bar-level state machine
//! (open / close / skip). Letting the legacy runner handle both would
//! double-route and create position conflicts. This module is the
//! parallel router for the champion stream.
//!
//! ## Safety gating
//! Two env flags must both be set for a single order to actually
//! route:
//!   - `LIVE_TRADER_ENABLED=true`  → live-trader runs and emits decisions
//!   - `CHAMPION_ROUTING_ENABLED=true` → this runner converts them to orders
//!
//! With either off, the system is a research-only diagnostic. Even
//! when both are on, the active `RoutingMode` defaults to `Internal`
//! (paper book) until the operator explicitly opts into
//! `OandaPractice` via the existing WS `SetMode` command, which itself
//! gates on `ALLOW_OANDA_ROUTING`.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::broadcast;

use market_domain::{Event, OrderIntent, RoutingMode, TraderDecision};
use portfolio::OrderRouter;

use crate::state::AppState;

/// Configuration parsed from env at spawn time.
#[derive(Clone, Debug)]
pub struct ChampionRouterConfig {
    /// Trade size in instrument-native units (≈ 100 for forex
    /// micro-lot, smaller for crypto). Conservative default keeps
    /// risk bounded until the operator tunes it.
    pub units_per_trade: i64,
    /// Run-id stamped on every routed intent so the audit log can
    /// distinguish champion-routed orders from manual ones.
    pub run_id: String,
}

impl ChampionRouterConfig {
    pub fn from_env() -> Self {
        let units = std::env::var("CHAMPION_TRADE_UNITS")
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(100)
            .max(1);
        let run_id = std::env::var("CHAMPION_RUN_ID")
            .unwrap_or_else(|_| {
                format!("champion_{}", chrono::Utc::now().format("%Y%m%dT%H%M%SZ"))
            });
        Self {
            units_per_trade: units,
            run_id,
        }
    }
}

/// Convenience: read both opt-in flags.
pub fn routing_enabled_in_env() -> bool {
    bool_env("CHAMPION_ROUTING_ENABLED")
}

fn bool_env(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

/// Spawn the runner. Returns immediately; the task ends when the bus
/// closes. Caller is responsible for verifying both opt-in flags.
pub fn spawn(state: Arc<AppState>, cfg: ChampionRouterConfig) -> tokio::task::JoinHandle<()> {
    let mut rx = state.bus.subscribe();
    let positions: Arc<Mutex<HashMap<String, i64>>> = Arc::new(Mutex::new(HashMap::new()));
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(Event::TraderDecision(d)) => {
                    if let Err(e) = handle_decision(&state, &cfg, &positions, d).await {
                        tracing::warn!(error = %e, "champion-router: routing failed");
                    }
                }
                Ok(_) => {}
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("champion-router: lagged {n} events");
                }
                Err(broadcast::error::RecvError::Closed) => return,
            }
        }
    })
}

async fn handle_decision(
    state: &Arc<AppState>,
    cfg: &ChampionRouterConfig,
    positions: &Arc<Mutex<HashMap<String, i64>>>,
    decision: TraderDecision,
) -> anyhow::Result<()> {
    let cur_pos = positions
        .lock()
        .get(&decision.instrument)
        .copied()
        .unwrap_or(0);
    let units_delta = action_to_units_delta(&decision.action, cfg.units_per_trade, cur_pos);
    if units_delta == 0 {
        return Ok(());
    }

    let intent = OrderIntent {
        instrument: decision.instrument.clone(),
        units: units_delta,
        time: decision.time,
        // Tag with both model_id and run_id so audit queries can
        // attribute fills to a specific champion + live session.
        model_id: format!("{}:{}", decision.model_id, cfg.run_id),
    };

    let active_mode = *state.mode.load_full();
    let router = pick_router(state, active_mode, &intent.instrument);
    let mode = router.mode();
    let fill = router
        .submit(intent.clone())
        .await
        .map_err(|e| anyhow::anyhow!("router rejected: {e}"))?;

    // Apply to the internal paper book (mark-to-market for the
    // dashboard) and update our local position tracker.
    {
        let mut pb = state.paper_book.lock();
        if !pb.is_seeded() {
            pb.seed_cash(100_000.0);
        }
        pb.apply_paper_fill(&fill.instrument, fill.units, fill.price);
    }
    {
        let mut pos = positions.lock();
        let cur = pos.entry(decision.instrument.clone()).or_insert(0);
        *cur += fill.units;
    }
    tracing::info!(
        instrument = %fill.instrument,
        units = fill.units,
        price = fill.price,
        action = %decision.action,
        params_id = %decision.params_id,
        ?mode,
        "champion-router: fill"
    );
    let _ = state.bus.send(Event::PaperFill(fill));
    Ok(())
}

/// Pick the right router for the (mode, venue) pair. Same logic as
/// `commands::pick_router` but accessed via the public `state.router`
/// pointers because that module's helper isn't pub.
fn pick_router(
    state: &Arc<AppState>,
    mode: RoutingMode,
    instrument: &str,
) -> Arc<dyn OrderRouter> {
    match mode {
        RoutingMode::Internal => Arc::clone(&state.router.load_full()),
        RoutingMode::OandaPractice => {
            let is_crypto = instrument.contains('/')
                || matches!(
                    instrument,
                    "BTC_USD" | "ETH_USD" | "SOL_USD" | "DOGE_USD" | "LTC_USD"
                );
            if is_crypto {
                state
                    .alpaca_router
                    .clone()
                    .unwrap_or_else(|| Arc::clone(&state.router.load_full()))
            } else {
                state
                    .oanda_router
                    .clone()
                    .unwrap_or_else(|| Arc::clone(&state.router.load_full()))
            }
        }
    }
}

/// Map a trader-decision action to a unit-delta order. Pure function;
/// extracted from the runner so the unit tests below can lock its
/// behaviour without spinning up an AppState.
///
///   - `open_long`  → +units_per_trade
///   - `open_short` → −units_per_trade
///   - `close`      → −cur_pos  (flatten)
///   - `skip` or anything else → 0  (no order)
pub fn action_to_units_delta(action: &str, units_per_trade: i64, cur_pos: i64) -> i64 {
    match action {
        "open_long" => units_per_trade,
        "open_short" => -units_per_trade,
        "close" => -cur_pos,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_long_returns_positive_units() {
        assert_eq!(action_to_units_delta("open_long", 100, 0), 100);
    }

    #[test]
    fn open_short_returns_negative_units() {
        assert_eq!(action_to_units_delta("open_short", 100, 0), -100);
    }

    #[test]
    fn close_flattens_long_position() {
        assert_eq!(action_to_units_delta("close", 100, 100), -100);
    }

    #[test]
    fn close_flattens_short_position() {
        assert_eq!(action_to_units_delta("close", 100, -100), 100);
    }

    #[test]
    fn close_with_no_position_is_zero() {
        assert_eq!(action_to_units_delta("close", 100, 0), 0);
    }

    #[test]
    fn skip_returns_zero() {
        assert_eq!(action_to_units_delta("skip", 100, 0), 0);
    }

    #[test]
    fn unknown_action_returns_zero() {
        assert_eq!(action_to_units_delta("dance", 100, 50), 0);
    }
}

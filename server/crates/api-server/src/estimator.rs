//! Estimator runner.
//!
//! Owns a [`PaperBook`], updates it from OANDA `Account` (seed) and
//! `Transaction` (apply ORDER_FILL) bus events, and emits an
//! `Event::Estimate` on a regular tick after marking-to-market against
//! `state.latest_prices`.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::broadcast;
use tokio::time::{interval, MissedTickBehavior};

use market_domain::Event;
use portfolio::PaperBook;

use crate::state::AppState;

pub fn spawn(state: Arc<AppState>) {
    let book = Arc::new(Mutex::new(PaperBook::new()));

    // Listener task: consume the bus and update the book as transactions arrive.
    {
        let state = state.clone();
        let book = book.clone();
        tokio::spawn(async move {
            let mut rx = state.bus.subscribe();
            loop {
                match rx.recv().await {
                    Ok(Event::Transaction(tx)) => {
                        book.lock().apply_oanda_fill(&tx);
                    }
                    Ok(Event::Account(acct)) => {
                        let mut b = book.lock();
                        if !b.is_seeded() {
                            b.seed_cash(acct.balance);
                            tracing::info!(
                                seeded = acct.balance,
                                "estimator seeded cash from first account snapshot"
                            );
                        }
                    }
                    Ok(_) => {}
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("estimator lagged {n} events; continuing");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }

    // Tick task: emit Estimate events on a regular cadence.
    {
        let state = state.clone();
        let book = book.clone();
        tokio::spawn(async move {
            let period = Duration::from_millis(state.cfg.estimator_tick_ms);
            let mut tick = interval(period);
            tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
            loop {
                tick.tick().await;
                let actual_nav = match state.account.read().clone() {
                    Some(a) => a.nav,
                    None => continue, // wait until we have at least one account poll
                };
                let est = {
                    let b = book.lock();
                    b.mark_to_market(actual_nav, |inst| {
                        state.latest_prices.get(inst).map(|p| p.mid)
                    })
                };
                if let Some(est) = est {
                    state.record_estimate(est);
                }
            }
        });
    }
}

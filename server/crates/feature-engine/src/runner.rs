//! Bus runner: subscribes to `Event::Price`, maintains per-instrument
//! incremental state, and emits `Event::Features` once warmup completes.
//!
//! Owned by `api-server::main`. The runner takes a `broadcast::Sender<Event>`
//! and a `broadcast::Receiver<Event>` (subscribed before this task starts
//! so we don't miss prices).

use std::collections::HashMap;

use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::broadcast;

use market_domain::{Event, FeatureVector};

use crate::IncrementalFeatures;

pub fn spawn(bus: broadcast::Sender<Event>) -> tokio::task::JoinHandle<()> {
    let mut rx = bus.subscribe();
    let states: Arc<Mutex<HashMap<String, IncrementalFeatures>>> =
        Arc::new(Mutex::new(HashMap::new()));
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(Event::Price(tick)) => {
                    let inst = tick.instrument.clone();
                    let fv: Option<FeatureVector> = {
                        let mut map = states.lock();
                        let entry = map
                            .entry(inst.clone())
                            .or_insert_with(|| IncrementalFeatures::new(inst));
                        entry.push(tick.mid, tick.bid, tick.ask, tick.time)
                    };
                    if let Some(v) = fv {
                        let _ = bus.send(Event::Features(v));
                    }
                }
                Ok(Event::OrderBook(ob)) => {
                    // Cache the snapshot on per-instrument state. Read on
                    // the next price tick.
                    let inst = ob.instrument.clone();
                    let mut map = states.lock();
                    let entry = map
                        .entry(inst.clone())
                        .or_insert_with(|| IncrementalFeatures::new(inst));
                    entry.set_orderbook(ob);
                }
                Ok(_) => {}
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("feature-engine lagged {n} events");
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    })
}

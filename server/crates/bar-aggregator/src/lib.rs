//! Tick → 10-second OHLCV bar aggregator.
//!
//! Subscribes to `Event::Price` on the broadcast bus, maintains a
//! per-instrument open-bar state, and emits `Event::Bar10s` whenever a
//! bar boundary is crossed (the next tick falls into a later 10s
//! bucket than the current open bar).
//!
//! The bar's stamped `ts_ms` is the bar-CLOSE timestamp (the start of
//! the next 10s bucket), matching the convention that the bar
//! represents the interval `[ts_ms - 10_000, ts_ms)`.
//!
//! No I/O of its own. `persistence::spawn_writer` already subscribes to
//! `Event::Bar10s` and writes the bars to `bars_10s`.

#![deny(unsafe_code)]

use dashmap::DashMap;
use tokio::sync::broadcast;

use market_domain::{bucket_floor, Bar10sNamed, Event, BAR_INTERVAL_MS};

/// Open-bar state for one instrument. `bucket_start_ms` is the floor of
/// the 10s window the open bar belongs to. The bar closes when a new
/// tick arrives whose bucket floor is strictly greater.
#[derive(Clone, Debug)]
struct OpenBar {
    bucket_start_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    n_ticks: u32,
    spread_bp_sum: f64,
}

impl OpenBar {
    fn new(bucket_start_ms: i64, mid: f64, spread_bp: f64) -> Self {
        Self {
            bucket_start_ms,
            open: mid,
            high: mid,
            low: mid,
            close: mid,
            n_ticks: 1,
            spread_bp_sum: spread_bp,
        }
    }

    fn extend(&mut self, mid: f64, spread_bp: f64) {
        if mid > self.high {
            self.high = mid;
        }
        if mid < self.low {
            self.low = mid;
        }
        self.close = mid;
        self.n_ticks += 1;
        self.spread_bp_sum += spread_bp;
    }

    fn close_to_named(&self, instrument: String) -> Bar10sNamed {
        Bar10sNamed {
            instrument,
            // Bar-close stamp = start of the NEXT bucket.
            ts_ms: self.bucket_start_ms + BAR_INTERVAL_MS,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            n_ticks: self.n_ticks,
            spread_bp_avg: if self.n_ticks > 0 {
                self.spread_bp_sum / self.n_ticks as f64
            } else {
                0.0
            },
        }
    }
}

/// Spawn the aggregator. Runs forever; ends when the bus closes.
pub fn spawn(bus: broadcast::Sender<Event>) -> tokio::task::JoinHandle<()> {
    let mut rx = bus.subscribe();
    let open_bars: DashMap<String, OpenBar> = DashMap::new();
    let tx = bus.clone();
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    if let Event::Price(p) = event {
                        let mid = p.mid;
                        let spread_bp =
                            if p.mid > 0.0 { p.spread / p.mid * 10_000.0 } else { 0.0 };
                        let ts_ms = p.time.timestamp_millis();
                        let bucket = bucket_floor(ts_ms);
                        let mut to_emit: Option<Bar10sNamed> = None;
                        let inst = p.instrument.clone();

                        // DashMap entry API gives us per-key locking.
                        let mut entry = open_bars.entry(inst.clone()).or_insert_with(|| {
                            OpenBar::new(bucket, mid, spread_bp)
                        });
                        if bucket > entry.bucket_start_ms {
                            // Bar boundary crossed: close the open bar, emit, and
                            // start a new one with this tick.
                            to_emit = Some(entry.close_to_named(inst.clone()));
                            *entry = OpenBar::new(bucket, mid, spread_bp);
                        } else {
                            entry.extend(mid, spread_bp);
                        }
                        drop(entry);

                        if let Some(bar) = to_emit {
                            // Send is best-effort; if no subscribers are present
                            // we don't care.
                            let _ = tx.send(Event::Bar10s(bar));
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("bar-aggregator: lagged {n} events");
                }
                Err(broadcast::error::RecvError::Closed) => return,
            }
        }
    })
}

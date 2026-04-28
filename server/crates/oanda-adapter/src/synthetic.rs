//! Synthetic random-walk pricing source.
//!
//! Used for offline demos (forex weekend) and tests. Emits the same
//! `PricingMessage` variants as the real OANDA pricing stream so
//! downstream consumers don't need to know which source they're reading.
//!
//! **Disabled by default.** Enable via `SYNTHETIC_TICKS=true` env var
//! handled in `api-server::main`. Never used in production.

use std::collections::HashMap;
use std::time::Duration;

use chrono::Utc;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use tokio::sync::mpsc::UnboundedSender;
use tokio::time::{interval, MissedTickBehavior};

use market_domain::PriceTick;

use crate::streams::pricing::PricingMessage;

/// Pip size per quoted-currency convention. JPY pairs use 0.01; everything
/// else uses 0.0001. Returned in price units (so a "1 pip" move on EUR_USD
/// is `+0.0001`).
fn pip_size(instrument: &str) -> f64 {
    if instrument.ends_with("_JPY") {
        0.01
    } else {
        0.0001
    }
}

/// Baked starting mid prices for the 7 majors. Picked to be "vaguely
/// realistic for early 2026" — actual numbers don't matter for the demo,
/// the random walk is what matters.
fn starting_mid(instrument: &str) -> f64 {
    match instrument {
        "EUR_USD" => 1.0900,
        "USD_JPY" => 150.00,
        "GBP_USD" => 1.2700,
        "AUD_USD" => 0.6500,
        "USD_CAD" => 1.3500,
        "USD_CHF" => 0.9000,
        "NZD_USD" => 0.6000,
        "XAU_USD" => 2050.0,
        _ => 1.0000,
    }
}

struct InstState {
    instrument: String,
    mid: f64,
    pip: f64,
}

impl InstState {
    fn new(instrument: String) -> Self {
        let pip = pip_size(&instrument);
        let mid = starting_mid(&instrument);
        Self {
            instrument,
            mid,
            pip,
        }
    }

    fn step(&mut self, rng: &mut SmallRng, dist: &Uniform<f64>) -> PriceTick {
        // Random-walk increment: uniform in ±1 pip with a slight drift.
        // Uniform is fine for a demo; the strategy doesn't need realistic
        // returns, just *some* signal it can learn (or fail to learn).
        let step = dist.sample(rng) * self.pip;
        self.mid = (self.mid + step).max(self.pip * 100.0); // floor far from zero
        let half_spread = self.pip; // ~2 pip total spread
        let bid = self.mid - half_spread;
        let ask = self.mid + half_spread;
        let mid = (bid + ask) / 2.0;
        let spread = ask - bid;
        PriceTick {
            instrument: self.instrument.clone(),
            time: Utc::now(),
            bid,
            ask,
            mid,
            spread,
            closeout_bid: Some(bid),
            closeout_ask: Some(ask),
            status: Some("tradeable".into()),
        }
    }
}

/// Spawn a synthetic pricing task. `hz` is the per-instrument tick rate.
/// At 100Hz across 7 instruments we generate 700 ticks/sec total, which
/// hits the 1000-tick training threshold per instrument in roughly 10s.
pub fn spawn(
    instruments: Vec<String>,
    sink: UnboundedSender<PricingMessage>,
    hz: f64,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move { run(instruments, sink, hz).await })
}

async fn run(instruments: Vec<String>, sink: UnboundedSender<PricingMessage>, hz: f64) {
    let mut states: HashMap<String, InstState> = instruments
        .into_iter()
        .map(|i| (i.clone(), InstState::new(i)))
        .collect();
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    // Each tick, increment mid by a uniform draw in [-1.5, +1.5] pip.
    let dist = Uniform::new_inclusive(-1.5_f64, 1.5_f64);

    // Convert hz → period. At 100Hz per instrument across 7 instruments
    // we sleep 1/(100*7) seconds between sends.
    let inst_count = states.len().max(1) as f64;
    let period_us = (1_000_000.0 / (hz * inst_count)).max(50.0);
    let mut tick = interval(Duration::from_micros(period_us as u64));
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let _ = sink.send(PricingMessage::Connected);
    tracing::info!(
        instruments = states.len(),
        hz,
        period_us = period_us as u64,
        "synthetic ticker started"
    );

    // Round-robin across instruments so each gets ticks at the same rate.
    let order: Vec<String> = {
        let mut v: Vec<String> = states.keys().cloned().collect();
        v.sort();
        v
    };
    let mut idx: usize = 0;
    loop {
        tick.tick().await;
        if sink.is_closed() {
            tracing::info!("synthetic sink closed; ticker exiting");
            return;
        }
        let inst = &order[idx % order.len()];
        idx = idx.wrapping_add(1);
        if let Some(st) = states.get_mut(inst) {
            let pricetick = st.step(&mut rng, &dist);
            let _ = sink.send(PricingMessage::Price(pricetick));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc::unbounded_channel;
    use tokio::time::{timeout, Duration as TokioDur};

    #[test]
    fn pip_size_jpy_vs_other() {
        assert_eq!(pip_size("USD_JPY"), 0.01);
        assert_eq!(pip_size("EUR_JPY"), 0.01);
        assert_eq!(pip_size("EUR_USD"), 0.0001);
    }

    #[test]
    fn starting_mid_known_majors() {
        assert!((starting_mid("EUR_USD") - 1.09).abs() < 1e-9);
        assert!((starting_mid("USD_JPY") - 150.0).abs() < 1e-9);
        assert_eq!(starting_mid("UNKNOWN_PAIR"), 1.0);
    }

    #[test]
    fn step_keeps_spread_positive_and_mid_in_band() {
        let mut st = InstState::new("EUR_USD".into());
        let mut rng = SmallRng::seed_from_u64(42);
        let dist = Uniform::new_inclusive(-1.5_f64, 1.5_f64);
        let starting = st.mid;
        for _ in 0..1000 {
            let t = st.step(&mut rng, &dist);
            assert!(t.spread > 0.0);
            assert!(t.bid < t.ask);
            assert!((t.mid - (t.bid + t.ask) / 2.0).abs() < 1e-12);
        }
        // After 1000 random steps in [-1.5, +1.5] pips, we shouldn't have drifted more
        // than ~3000 pips (extreme); typically much less.
        let drift = (st.mid - starting).abs();
        assert!(drift < 1000.0 * 0.0001);
    }

    #[tokio::test]
    async fn spawn_emits_connected_then_prices() {
        let (tx, mut rx) = unbounded_channel::<PricingMessage>();
        let handle = spawn(vec!["EUR_USD".into(), "USD_JPY".into()], tx, 200.0);

        // First message should be Connected.
        let first = timeout(TokioDur::from_secs(1), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(first, PricingMessage::Connected));

        // Then we should see at least 30 prices in 250ms.
        let mut count = 0;
        let deadline = TokioDur::from_millis(500);
        let started = std::time::Instant::now();
        while started.elapsed() < deadline {
            if let Ok(Some(msg)) = timeout(TokioDur::from_millis(100), rx.recv()).await {
                if matches!(msg, PricingMessage::Price(_)) {
                    count += 1;
                }
            }
        }
        assert!(count >= 30, "expected >=30 prices, got {count}");
        handle.abort();
    }
}

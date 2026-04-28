//! Integration test: feed `Event::Bar10s` events into the bus, assert
//! `Event::ChampionSignal` events come back out for each bar that
//! arrives after the rolling buffer warms up.

use std::sync::Arc;
use std::time::Duration;

use inference::PredictorRegistry;
use market_domain::{Bar10sNamed, Event, FEATURE_DIM};
use tokio::sync::broadcast;

fn make_bar(ts_ms: i64, close: f64) -> Bar10sNamed {
    Bar10sNamed {
        instrument: "EUR_USD".to_string(),
        ts_ms,
        open: close,
        high: close + 1e-4,
        low: close - 1e-4,
        close,
        n_ticks: 1,
        spread_bp_avg: 0.5,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_inference_emits_champion_signal_after_warmup() {
    let (bus, _initial_rx) = broadcast::channel::<Event>(256);
    let registry = Arc::new(PredictorRegistry::new(FEATURE_DIM));

    // Subscribe BEFORE spawn so we don't race past early bars.
    let mut sink = bus.subscribe();

    let _join = live_inference::spawn(bus.clone(), registry.clone());

    // Feed 5 bars — should produce no ChampionSignal (need >= 2 bars
    // for a return, but recompute_last needs only 2). Send 60 to be
    // safely above all rolling windows used by features.
    for i in 0..60_i64 {
        let close = 1.0 + (i as f64) * 1e-4;
        bus.send(Event::Bar10s(make_bar(i * 10_000, close))).expect("send");
    }

    // Drain events with a generous timeout. Expect at least one
    // ChampionSignal whose model_id is the fallback.
    let mut got_signal = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while tokio::time::Instant::now() < deadline {
        let recv = tokio::time::timeout(Duration::from_millis(50), sink.recv()).await;
        match recv {
            Ok(Ok(Event::ChampionSignal(c))) => {
                assert!(c.p_long.is_finite());
                assert!(c.p_short.is_finite());
                assert!((c.p_long + c.p_short - 1.0).abs() < 1e-6);
                assert_eq!(c.kind, "fallback");
                got_signal = true;
                break;
            }
            Ok(Ok(_)) => continue,
            Ok(Err(_)) => break,
            Err(_) => continue, // timeout, loop
        }
    }
    assert!(got_signal, "expected at least one ChampionSignal event");
}

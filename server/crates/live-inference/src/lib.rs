//! Live ONNX inference runner.
//!
//! Subscribes to `Event::Bar10s` on the broadcast bus. For each closed
//! bar:
//!   1. Append it to a per-instrument rolling buffer (last 600 bars,
//!      enough to power every rolling window in `bar-features`).
//!   2. Recompute the 24-dim feature vector for the bar via
//!      `bar_features::recompute_last`.
//!   3. Call `inference::PredictorRegistry::current()` and run inference.
//!   4. Emit `Event::ChampionSignal` on the bus.
//!
//! Skipped silently while the buffer is too short to produce stable
//! features (< 2 bars). When the registry has only the neutral
//! fallback loaded, predictions are still emitted with `kind="fallback"`
//! so the dashboard can show that state.

#![deny(unsafe_code)]

use std::collections::VecDeque;
use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};
use dashmap::DashMap;
use parking_lot::Mutex;
use tokio::sync::broadcast;

use bar_features::{recompute_last, FEATURE_NAMES, N_FEATURES};
use inference::{Predictor, PredictorRegistry};
use market_domain::{Bar10s, Bar10sNamed, ChampionSignal, Event, FEATURE_DIM};

/// Maximum bars retained per instrument for the rolling feature
/// recompute. Picked so the largest window (`drawdown_300` at 300 bars)
/// still fits with headroom.
const ROLLING_WINDOW_BARS: usize = 600;

/// Spawn the runner. Returns immediately; the task runs forever and
/// ends when the bus closes.
pub fn spawn(
    bus: broadcast::Sender<Event>,
    registry: Arc<PredictorRegistry>,
) -> tokio::task::JoinHandle<()> {
    // Compile-time invariant: the live registry's expected feature
    // dimension must equal `bar-features::N_FEATURES` and
    // `market_domain::FEATURE_DIM`.
    debug_assert_eq!(N_FEATURES, FEATURE_DIM);

    let buffers: Arc<DashMap<String, Mutex<VecDeque<Bar10s>>>> = Arc::new(DashMap::new());
    let mut rx = bus.subscribe();
    let tx = bus.clone();

    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(Event::Bar10s(named)) => {
                    on_bar(&buffers, &registry, &tx, named);
                }
                Ok(_) => {}
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("live-inference: lagged {n} events");
                }
                Err(broadcast::error::RecvError::Closed) => return,
            }
        }
    })
}

fn on_bar(
    buffers: &DashMap<String, Mutex<VecDeque<Bar10s>>>,
    registry: &PredictorRegistry,
    bus: &broadcast::Sender<Event>,
    named: Bar10sNamed,
) {
    // Append to the per-instrument rolling buffer.
    let entry = buffers
        .entry(named.instrument.clone())
        .or_insert_with(|| Mutex::new(VecDeque::with_capacity(ROLLING_WINDOW_BARS + 1)));
    let buffer_lock = entry.value();
    let mut buf = buffer_lock.lock();
    let bar = Bar10s {
        instrument_id: 0,
        ts_ms: named.ts_ms,
        open: named.open,
        high: named.high,
        low: named.low,
        close: named.close,
        n_ticks: named.n_ticks,
        spread_bp_avg: named.spread_bp_avg,
    };
    buf.push_back(bar);
    while buf.len() > ROLLING_WINDOW_BARS {
        buf.pop_front();
    }
    // Cheap snapshot for the recomputer (it needs &[Bar10s], not VecDeque).
    let bars: Vec<Bar10s> = buf.iter().copied().collect();
    drop(buf);

    let Some(feat) = recompute_last(&bars) else {
        return; // not enough bars yet
    };
    if feat.len() != registry.expected_n_features() {
        tracing::error!(
            got = feat.len(),
            want = registry.expected_n_features(),
            "live-inference: feature dim mismatch — skipping bar"
        );
        return;
    }

    let predictor = registry.current();
    let probs = predictor.predict(&feat);
    let id = predictor.id().to_string();
    let kind = if id == inference::fallback::FALLBACK_ID {
        "fallback".to_string()
    } else {
        "onnx".to_string()
    };

    let time = bar_close_to_datetime(named.ts_ms);
    let signal = ChampionSignal {
        instrument: named.instrument.clone(),
        time,
        p_long: probs.p_long,
        p_short: probs.p_short,
        p_take: probs.p_take,
        calibrated: probs.calibrated,
        model_id: id.clone(),
        kind: kind.clone(),
    };
    let _ = bus.send(Event::ChampionSignal(signal));

    // Append the per-bar feature vector + champion output to the
    // agent-readable JSONL preview. Failures are logged + swallowed;
    // this is a review artifact, not load-bearing.
    if let Err(e) = write_features_jsonl(&named, &feat, probs, &id, &kind) {
        tracing::debug!(error = %e, "live-inference: features.jsonl append failed");
    }
}

/// Top-level dir used for `trade_logs/<v>/<ticker>/features.jsonl`.
/// Matches `live-trader`'s convention.
fn trade_logs_root() -> std::path::PathBuf {
    std::env::var("TRADE_LOGS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("./trade_logs"))
}

const RELEASE_VERSION: &str = env!("CARGO_PKG_VERSION");

fn write_features_jsonl(
    bar: &Bar10sNamed,
    feat: &[f64],
    probs: inference::Probs,
    model_id: &str,
    kind: &str,
) -> anyhow::Result<()> {
    use serde_json::json;
    // Round to 5 decimals to keep the line short. Pair each feature
    // with its name so an agent can read it without referring to the
    // schema.
    let named: serde_json::Map<String, serde_json::Value> = FEATURE_NAMES
        .iter()
        .zip(feat.iter())
        .map(|(name, value)| {
            (
                (*name).to_string(),
                serde_json::Value::from((value * 100_000.0).round() / 100_000.0),
            )
        })
        .collect();
    let record = json!({
        "v": format!("v{RELEASE_VERSION}"),
        "instrument": bar.instrument,
        "ts_ms": bar.ts_ms,
        "close": bar.close,
        "n_ticks": bar.n_ticks,
        "spread_bp_avg": bar.spread_bp_avg,
        "model_id": model_id,
        "kind": kind,
        "p_long": probs.p_long,
        "p_short": probs.p_short,
        "calibrated": probs.calibrated,
        "features": serde_json::Value::Object(named),
    });
    let safe_inst = bar.instrument.replace('/', "_");
    let sub = format!("{safe_inst}/features.jsonl");
    live_trader::jsonl_log::append(&trade_logs_root(), RELEASE_VERSION, &sub, &record)
}

fn bar_close_to_datetime(ts_ms: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(ts_ms).single().unwrap_or_else(Utc::now)
}

//! Strategy runner: subscribes to `Event::Features` (and `Event::Price` for
//! mid lookups), maintains a per-instrument labeled buffer, retrains every
//! 100 features past warmup of 1000, and emits `Event::Signal` per tick
//! after the first model lands plus `Event::Fitness` on each retrain.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::broadcast;

use market_domain::{
    Event, FeatureVector, FitnessMetrics, ModelFitness, SignalDirection, StrategySignal,
    FEATURE_DIM,
};

use crate::online_logreg::{LogReg, LogRegConfig, CLASS_LONG, CLASS_SHORT, NUM_CLASSES};
use crate::walk_forward::{build_labeled, train_walk_forward, LABEL_HORIZON_TICKS};

/// Number of features observed per instrument before the first training pass.
pub const TRAIN_AFTER: usize = 1000;
/// Retrain every N additional features after the first model.
pub const RETRAIN_EVERY: usize = 100;
/// Maximum buffer per instrument. Bounded to avoid unbounded memory.
pub const MAX_BUFFER: usize = 8000;

#[derive(Clone)]
struct InstState {
    feats: Vec<[f64; FEATURE_DIM]>,
    mids: Vec<f64>,
    seen_for_retrain: usize,
    model: Option<LogReg>,
    model_id: String,
    model_version: u32,
}

impl InstState {
    fn new() -> Self {
        Self {
            feats: Vec::with_capacity(2048),
            mids: Vec::with_capacity(2048),
            seen_for_retrain: 0,
            model: None,
            model_id: String::new(),
            model_version: 0,
        }
    }
}

pub fn spawn(bus: broadcast::Sender<Event>) -> tokio::task::JoinHandle<()> {
    spawn_with_prefill(bus, HashMap::new())
}

/// Like `spawn`, but pre-seeds each instrument's feature buffer with
/// historical (feature_vector, mid_price) pairs loaded from the
/// persistence layer. Pairs must be in chronological order, oldest
/// first. After prefill the runner immediately attempts to train each
/// instrument that has reached `TRAIN_AFTER`, then resumes its normal
/// per-tick subscriber loop.
pub fn spawn_with_prefill(
    bus: broadcast::Sender<Event>,
    prefill: HashMap<String, Vec<(FeatureVector, f64)>>,
) -> tokio::task::JoinHandle<()> {
    let mut rx = bus.subscribe();
    let states: Arc<Mutex<HashMap<String, InstState>>> = Arc::new(Mutex::new(HashMap::new()));
    // Cache last mid per instrument so we can pair it with each feature.
    let mids: Arc<Mutex<HashMap<String, f64>>> = Arc::new(Mutex::new(HashMap::new()));

    // Apply the prefill before subscribing-by-tick. Bound each
    // instrument to MAX_BUFFER so a generous DB doesn't blow memory.
    {
        let mut map = states.lock();
        let mut cached = mids.lock();
        for (instrument, pairs) in prefill {
            if pairs.is_empty() {
                continue;
            }
            let st = map.entry(instrument.clone()).or_insert_with(InstState::new);
            for (fv, mid) in pairs {
                st.feats.push(fv.vector);
                st.mids.push(mid);
            }
            if st.feats.len() > MAX_BUFFER {
                let drop_n = st.feats.len() - MAX_BUFFER;
                st.feats.drain(0..drop_n);
                st.mids.drain(0..drop_n);
            }
            // Seed last-mid cache so the first live Feature event after
            // boot has a paired price even before the first Price event
            // arrives.
            if let Some(last_mid) = st.mids.last().copied() {
                cached.insert(instrument.clone(), last_mid);
            }
            // Immediately fire one training pass per prefilled instrument
            // so the model is ready before the next live tick. Skips
            // gracefully when there isn't enough data yet.
            if st.feats.len() >= TRAIN_AFTER {
                if let Some(fit) = train_inst(&instrument, st) {
                    let _ = bus.send(Event::Fitness(fit));
                    tracing::info!(
                        instrument = %instrument,
                        samples = st.feats.len(),
                        "strategy: prefill-trained model on startup"
                    );
                }
            } else {
                tracing::info!(
                    instrument = %instrument,
                    prefilled = st.feats.len(),
                    needed = TRAIN_AFTER,
                    "strategy: prefilled buffer (still warming up)"
                );
            }
        }
    }

    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(Event::Price(t)) => {
                    mids.lock().insert(t.instrument.clone(), t.mid);
                }
                Ok(Event::Features(fv)) => {
                    let inst = fv.instrument.clone();
                    let mid = match mids.lock().get(&inst).copied() {
                        Some(m) => m,
                        None => continue, // shouldn't happen — a price always precedes a feature
                    };

                    // Update buffer + maybe retrain. Hold lock briefly.
                    let (signal_event, fitness_event): (
                        Option<StrategySignal>,
                        Option<ModelFitness>,
                    ) = {
                        let mut map = states.lock();
                        let st = map.entry(inst.clone()).or_insert_with(InstState::new);
                        st.feats.push(fv.vector);
                        st.mids.push(mid);
                        if st.feats.len() > MAX_BUFFER {
                            // Drop the oldest 10% to keep things bounded.
                            let drop_n = MAX_BUFFER / 10;
                            st.feats.drain(0..drop_n);
                            st.mids.drain(0..drop_n);
                        }
                        st.seen_for_retrain += 1;

                        // Decide whether to retrain.
                        let want_retrain = match (st.model.is_some(), st.feats.len()) {
                            (false, n) if n >= TRAIN_AFTER => true,
                            (true, _) if st.seen_for_retrain >= RETRAIN_EVERY => true,
                            _ => false,
                        };
                        let mut fitness_out: Option<ModelFitness> = None;
                        if want_retrain {
                            fitness_out = train_inst(&inst, st);
                        }

                        // Emit a signal if we have a model. Binary classifier:
                        // probs is [short, long] internally; the wire format
                        // is [long, flat, short] for backwards compat with
                        // the signals table + dashboard. Flat is wired as
                        // 0.0 (no flat class) and the model never emits
                        // SignalDirection::Flat anymore.
                        let signal_out = st.model.as_ref().map(|m| {
                            let bp = m.predict_probs(&fv.vector);
                            let argmax = (0..NUM_CLASSES)
                                .max_by(|&a, &b| bp[a].partial_cmp(&bp[b]).unwrap())
                                .unwrap();
                            let dir = if argmax == CLASS_LONG {
                                SignalDirection::Long
                            } else {
                                SignalDirection::Short
                            };
                            let wire_probs = [bp[CLASS_LONG], 0.0, bp[CLASS_SHORT]];
                            StrategySignal {
                                instrument: inst.clone(),
                                time: fv.time,
                                direction: dir,
                                confidence: bp[argmax],
                                probs: wire_probs,
                                model_id: st.model_id.clone(),
                                model_version: st.model_version,
                            }
                        });
                        (signal_out, fitness_out)
                    };

                    if let Some(s) = signal_event {
                        let _ = bus.send(Event::Signal(s));
                    }
                    if let Some(f) = fitness_event {
                        let _ = bus.send(Event::Fitness(f));
                    }
                    // Silence "unused" warning for FitnessMetrics import on test path.
                    let _ = std::mem::size_of::<FitnessMetrics>();
                    let _ = std::mem::size_of::<FeatureVector>();
                }
                Ok(_) => {}
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("strategy runner lagged {n} events");
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    })
}

/// Run one walk-forward training pass against an instrument's current
/// buffer, mutating `st` to install the new model + version. Returns
/// the `ModelFitness` to broadcast on success, or `None` if the
/// trainer rejected the data (typically not enough samples).
fn train_inst(inst: &str, st: &mut InstState) -> Option<ModelFitness> {
    let labeled = build_labeled(&st.feats, &st.mids);
    let n = labeled.len();
    let train_end = (n as f64 * 0.8) as usize;
    let purge = LABEL_HORIZON_TICKS + 5;
    let oos_end = n;
    let cfg = LogRegConfig::default();
    let res = train_walk_forward(&labeled, train_end, purge, oos_end, &cfg)?;
    st.model = Some(res.model);
    st.model_version += 1;
    st.model_id = format!("{}-v{}", inst, st.model_version);
    st.seen_for_retrain = 0;
    let fit = ModelFitness {
        instrument: inst.to_string(),
        model_id: st.model_id.clone(),
        model_version: st.model_version,
        trained_at: chrono::Utc::now(),
        train: res.train,
        oos: res.oos,
        samples_seen: st.feats.len(),
        train_window: res.train_window,
        oos_window: res.oos_window,
    };
    tracing::info!(
        instrument = %inst,
        version = st.model_version,
        n = labeled.len(),
        oos_acc = fit.oos.accuracy,
        "model trained"
    );
    Some(fit)
}

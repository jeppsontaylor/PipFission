//! Bar-driven automatic retrain orchestrator.
//!
//! Subscribes to `Event::Bar10s`. For each instrument we are tracking,
//! count closed bars; when the per-instrument counter hits the
//! threshold, fire the same `python -m research pipeline run` we
//! expose at `POST /api/pipeline/run`. The flight slot is shared so
//! the auto-retrain task and a manual REST trigger never both spawn
//! a Python subprocess at the same time.
//!
//! Opt-in via `AUTO_RETRAIN_ENABLED=true`. Without that env, the
//! task spawns and counts but never fires — useful for collecting
//! telemetry (`/api/pipeline/auto-retrain`) before turning the
//! actual fire on.
//!
//! Behaviour:
//!  * counter resets the moment we successfully spawn (not on
//!    completion), so a long Python run doesn't queue up dozens of
//!    "should-fire" decisions behind it. While the flight slot is
//!    held the counter stops incrementing — bars during a run are
//!    effectively absorbed by the run we just kicked off.
//!  * if a spawn attempt fails (slot busy, log-file error,
//!    PIPELINE_TRIGGER_ENABLED off), the counter is *not* reset — the
//!    next bar will try again.
//!  * one task watches all instruments. Counters are independent.

use std::sync::Arc;

use chrono::Utc;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::Serialize;
use tokio::sync::broadcast;

use market_domain::Event;

use crate::pipeline_trigger::{
    spawn_pipeline_subprocess, PipelineRunRequest, PipelineRunResponse, PipelineSpawnError,
};
use crate::state::AppState;

#[derive(Clone, Debug)]
pub struct AutoRetrainConfig {
    /// Master switch. When false, bars are still counted but no
    /// subprocess fires.
    pub enabled: bool,
    /// Bars per instrument before we fire.
    pub bars_threshold: u32,
    /// Instruments to track. Bars for any other instrument are
    /// ignored entirely (no counter, not surfaced).
    pub instruments: Vec<String>,
}

impl AutoRetrainConfig {
    /// Pull from env. Defaults: disabled, threshold 100, no instruments.
    /// `AUTO_RETRAIN_INSTRUMENTS` is comma-separated, e.g.
    /// `EUR_USD,USD_JPY,BTC_USD`.
    pub fn from_env() -> Self {
        let enabled = std::env::var("AUTO_RETRAIN_ENABLED").as_deref() == Ok("true");
        let bars_threshold = std::env::var("AUTO_RETRAIN_BARS_THRESHOLD")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(100)
            .max(1);
        let instruments = std::env::var("AUTO_RETRAIN_INSTRUMENTS")
            .ok()
            .map(|s| {
                s.split(',')
                    .map(|t| t.trim().to_string())
                    .filter(|t| !t.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        Self {
            enabled,
            bars_threshold,
            instruments,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CounterDecision {
    HoldOff,
    ShouldFire,
}

/// Per-instrument bar counter. Pure, unit-testable — the broadcast
/// loop just calls `record_bar` and reacts to the decision.
#[derive(Default)]
pub struct AutoRetrainCounters {
    counters: DashMap<String, u32>,
    last_fired_ms: DashMap<String, i64>,
    last_fire_skipped_reason: DashMap<String, String>,
    threshold: u32,
    instruments: Vec<String>,
}

impl AutoRetrainCounters {
    pub fn new(threshold: u32, instruments: Vec<String>) -> Self {
        let counters = DashMap::new();
        for inst in &instruments {
            counters.insert(inst.clone(), 0u32);
        }
        Self {
            counters,
            last_fired_ms: DashMap::new(),
            last_fire_skipped_reason: DashMap::new(),
            threshold: threshold.max(1),
            instruments,
        }
    }

    /// Whether this instrument is on the watchlist. Bars for other
    /// instruments are ignored.
    pub fn tracks(&self, instrument: &str) -> bool {
        self.instruments.iter().any(|i| i == instrument)
    }

    /// Increment and decide. Caller is expected to honour the
    /// decision: on `ShouldFire`, attempt a spawn; on success, call
    /// `mark_fired` to reset; on failure, leave counters alone so the
    /// next bar retries.
    pub fn record_bar(&self, instrument: &str) -> CounterDecision {
        if !self.tracks(instrument) {
            return CounterDecision::HoldOff;
        }
        let mut entry = self.counters.entry(instrument.to_string()).or_insert(0);
        *entry = entry.saturating_add(1);
        if *entry >= self.threshold {
            CounterDecision::ShouldFire
        } else {
            CounterDecision::HoldOff
        }
    }

    /// Reset the counter and stamp the last-fired time. Call this
    /// only after a successful spawn — it's the "we handed off to
    /// Python" milestone, not "Python finished".
    pub fn mark_fired(&self, instrument: &str) {
        if let Some(mut e) = self.counters.get_mut(instrument) {
            *e = 0;
        }
        self.last_fired_ms
            .insert(instrument.to_string(), Utc::now().timestamp_millis());
        self.last_fire_skipped_reason.remove(instrument);
    }

    /// Record why a should-fire was *not* honored (e.g. another run
    /// was already in flight). Surfaced on `/api/pipeline/auto-retrain`.
    pub fn mark_skipped(&self, instrument: &str, reason: &str) {
        self.last_fire_skipped_reason
            .insert(instrument.to_string(), reason.to_string());
    }

    pub fn snapshot(&self) -> Vec<AutoRetrainInstrumentStatus> {
        let mut out = Vec::with_capacity(self.instruments.len());
        for inst in &self.instruments {
            let bars = self.counters.get(inst).map(|v| *v).unwrap_or(0);
            let last = self.last_fired_ms.get(inst).map(|v| *v);
            let skipped = self
                .last_fire_skipped_reason
                .get(inst)
                .map(|v| v.clone());
            out.push(AutoRetrainInstrumentStatus {
                instrument: inst.clone(),
                bars_since_last_fire: bars,
                last_fired_ms: last,
                last_skip_reason: skipped,
            });
        }
        out
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct AutoRetrainInstrumentStatus {
    pub instrument: String,
    pub bars_since_last_fire: u32,
    pub last_fired_ms: Option<i64>,
    pub last_skip_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AutoRetrainStatus {
    pub enabled: bool,
    pub bars_threshold: u32,
    pub instruments: Vec<AutoRetrainInstrumentStatus>,
}

/// Shared runtime handle so the HTTP route can read counter state.
pub struct AutoRetrain {
    pub cfg: AutoRetrainConfig,
    pub counters: Arc<AutoRetrainCounters>,
    /// Last error / log message from a spawn attempt — surfaced on the
    /// status endpoint for debugging.
    pub last_status_message: RwLock<Option<String>>,
}

impl AutoRetrain {
    pub fn new(cfg: AutoRetrainConfig) -> Self {
        let counters = Arc::new(AutoRetrainCounters::new(
            cfg.bars_threshold,
            cfg.instruments.clone(),
        ));
        Self {
            cfg,
            counters,
            last_status_message: RwLock::new(None),
        }
    }

    pub fn status(&self) -> AutoRetrainStatus {
        AutoRetrainStatus {
            enabled: self.cfg.enabled,
            bars_threshold: self.cfg.bars_threshold,
            instruments: self.counters.snapshot(),
        }
    }

    /// Record the outcome of a spawn attempt and update counters.
    ///
    /// Decoupled from `spawn_pipeline_subprocess` so the broadcast loop
    /// is testable without forking a Python subprocess: tests construct
    /// fake `Result`s directly. Returns `true` on a successful spawn so
    /// callers can branch on the outcome (e.g. log differently).
    pub fn record_spawn_result(
        &self,
        instrument: &str,
        result: Result<&PipelineRunResponse, &PipelineSpawnError>,
    ) -> bool {
        match result {
            Ok(resp) => {
                self.counters.mark_fired(instrument);
                *self.last_status_message.write() =
                    Some(format!("fired run_id={} for {instrument}", resp.run_id));
                true
            }
            Err(e) => {
                let msg = format!("{e}");
                self.counters.mark_skipped(instrument, &msg);
                *self.last_status_message.write() =
                    Some(format!("{instrument}: skipped — {msg}"));
                false
            }
        }
    }
}

/// Spawn the broadcast subscriber. Returns the `JoinHandle` so main
/// can keep it alive (and drop on shutdown).
pub fn spawn(
    state: Arc<AppState>,
    auto: Arc<AutoRetrain>,
) -> tokio::task::JoinHandle<()> {
    let mut rx = state.bus.subscribe();
    let task_state = state.clone();
    tokio::spawn(async move {
        tracing::info!(
            enabled = auto.cfg.enabled,
            threshold = auto.cfg.bars_threshold,
            instruments = ?auto.cfg.instruments,
            "auto-retrain task started"
        );
        loop {
            match rx.recv().await {
                Ok(Event::Bar10s(bar)) => {
                    let decision = auto.counters.record_bar(&bar.instrument);
                    if decision == CounterDecision::ShouldFire && auto.cfg.enabled {
                        attempt_fire(&task_state, &auto, &bar.instrument).await;
                    }
                }
                Ok(_) => {}
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("auto-retrain lagged {n} events");
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
        tracing::info!("auto-retrain task exiting");
    })
}

async fn attempt_fire(state: &Arc<AppState>, auto: &Arc<AutoRetrain>, instrument: &str) {
    // Sentinel mode: when AUTO_RETRAIN_VIA_SENTINEL=true, write the
    // instrument to `data/.retrain-pending` and exit with code 75
    // (EX_TEMPFAIL). The supervisor (`scripts/api-server-supervisor.sh`)
    // sees the exit code, drains the sentinel, runs the Python pipeline
    // (which can now acquire the DuckDB write lock the api-server held),
    // then restarts the api-server. This is the permanent fix for the
    // "auto-retrain never succeeds because of the DB lock" bug.
    if std::env::var("AUTO_RETRAIN_VIA_SENTINEL").as_deref() == Ok("true") {
        let sentinel = std::path::PathBuf::from("./data/.retrain-pending");
        if let Some(parent) = sentinel.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let line = format!("{instrument}\n");
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&sentinel)
        {
            Ok(mut f) => {
                use std::io::Write;
                if let Err(e) = f.write_all(line.as_bytes()) {
                    tracing::error!(error = %e, "auto-retrain: sentinel write failed");
                    let err = PipelineSpawnError::SpawnFailed(format!("sentinel write: {e}"));
                    auto.record_spawn_result(instrument, Err(&err));
                    return;
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "auto-retrain: sentinel open failed");
                let err = PipelineSpawnError::SpawnFailed(format!("sentinel open: {e}"));
                auto.record_spawn_result(instrument, Err(&err));
                return;
            }
        }
        tracing::info!(instrument, "auto-retrain: sentinel written, exiting for handoff");
        // Schedule a clean exit on the next tick. The supervisor sees
        // exit code 75 and triggers the python handoff.
        tokio::spawn(async {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            std::process::exit(75);
        });
        return;
    }

    // In-process mode (default off): spawn python directly. Will fail
    // with DB lock conflict against the live api-server.
    let req = PipelineRunRequest {
        instrument: instrument.to_string(),
        n_bars: None,
        n_optuna_trials: None,
        n_trader_trials: None,
        seed: None,
    };
    let result = spawn_pipeline_subprocess(state, req).await;
    match &result {
        Ok(resp) => {
            tracing::info!(
                instrument = %instrument,
                run_id = %resp.run_id,
                "auto-retrain fired"
            );
        }
        Err(e) => {
            tracing::warn!(instrument = %instrument, error = %e, "auto-retrain skipped");
        }
    }
    auto.record_spawn_result(instrument, result.as_ref());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counters(threshold: u32, instruments: &[&str]) -> AutoRetrainCounters {
        AutoRetrainCounters::new(
            threshold,
            instruments.iter().map(|s| s.to_string()).collect(),
        )
    }

    #[test]
    fn tracks_only_listed_instruments() {
        let c = counters(3, &["EUR_USD"]);
        assert!(c.tracks("EUR_USD"));
        assert!(!c.tracks("USD_JPY"));
    }

    #[test]
    fn untracked_instruments_never_fire() {
        let c = counters(1, &["EUR_USD"]);
        for _ in 0..50 {
            assert_eq!(c.record_bar("USD_JPY"), CounterDecision::HoldOff);
        }
    }

    #[test]
    fn fires_at_threshold_for_tracked() {
        let c = counters(3, &["EUR_USD"]);
        assert_eq!(c.record_bar("EUR_USD"), CounterDecision::HoldOff);
        assert_eq!(c.record_bar("EUR_USD"), CounterDecision::HoldOff);
        assert_eq!(c.record_bar("EUR_USD"), CounterDecision::ShouldFire);
        // Counter keeps increasing until mark_fired.
        assert_eq!(c.record_bar("EUR_USD"), CounterDecision::ShouldFire);
    }

    #[test]
    fn mark_fired_resets_only_that_instrument() {
        let c = counters(2, &["EUR_USD", "USD_JPY"]);
        c.record_bar("EUR_USD");
        c.record_bar("EUR_USD");
        c.record_bar("USD_JPY");
        c.mark_fired("EUR_USD");

        let snap: Vec<_> = c.snapshot();
        let eu = snap.iter().find(|s| s.instrument == "EUR_USD").unwrap();
        let jp = snap.iter().find(|s| s.instrument == "USD_JPY").unwrap();
        assert_eq!(eu.bars_since_last_fire, 0);
        assert_eq!(jp.bars_since_last_fire, 1);
        assert!(eu.last_fired_ms.is_some());
        assert!(jp.last_fired_ms.is_none());
    }

    #[test]
    fn threshold_below_one_is_clamped() {
        let c = counters(0, &["EUR_USD"]);
        // Even with threshold=0 → clamped to 1, so the very first bar fires.
        assert_eq!(c.record_bar("EUR_USD"), CounterDecision::ShouldFire);
    }

    #[test]
    fn snapshot_lists_all_tracked_even_before_first_bar() {
        let c = counters(5, &["EUR_USD", "USD_JPY", "BTC_USD"]);
        let snap = c.snapshot();
        assert_eq!(snap.len(), 3);
        assert!(snap.iter().all(|s| s.bars_since_last_fire == 0));
        assert!(snap.iter().all(|s| s.last_fired_ms.is_none()));
    }

    #[test]
    fn mark_skipped_records_reason() {
        let c = counters(1, &["EUR_USD"]);
        c.record_bar("EUR_USD");
        c.mark_skipped("EUR_USD", "another run in flight");
        let snap = c.snapshot();
        assert_eq!(
            snap[0].last_skip_reason.as_deref(),
            Some("another run in flight")
        );
        // mark_fired clears the skip reason.
        c.mark_fired("EUR_USD");
        let snap = c.snapshot();
        assert!(snap[0].last_skip_reason.is_none());
    }

    #[test]
    fn counter_does_not_overflow_on_long_idle() {
        let c = counters(10, &["EUR_USD"]);
        // Saturating_add prevents wrap when the counter sits past
        // threshold without being marked fired (e.g. flight slot busy).
        for _ in 0..1000 {
            c.record_bar("EUR_USD");
        }
        let snap = c.snapshot();
        assert_eq!(snap[0].bars_since_last_fire, 1000);
    }

    #[test]
    fn record_spawn_result_success_resets_counter_and_sets_status() {
        let cfg = AutoRetrainConfig {
            enabled: true,
            bars_threshold: 5,
            instruments: vec!["EUR_USD".into()],
        };
        let auto = AutoRetrain::new(cfg);
        // Drive counter past threshold.
        for _ in 0..5 {
            auto.counters.record_bar("EUR_USD");
        }
        assert_eq!(
            auto.counters.snapshot()[0].bars_since_last_fire,
            5,
            "counter should be at threshold"
        );

        let resp = PipelineRunResponse {
            run_id: "test-run-abc".into(),
            instrument: "EUR_USD".into(),
            started_ms: 0,
            log_path: "/tmp/test.log".into(),
        };
        let fired = auto.record_spawn_result("EUR_USD", Ok(&resp));
        assert!(fired);

        let snap = auto.counters.snapshot();
        assert_eq!(snap[0].bars_since_last_fire, 0, "counter resets");
        assert!(snap[0].last_fired_ms.is_some(), "stamp set");
        assert!(snap[0].last_skip_reason.is_none(), "no skip reason");

        let msg = auto.last_status_message.read().clone().unwrap();
        assert!(msg.contains("test-run-abc"), "msg includes run_id");
        assert!(msg.contains("EUR_USD"), "msg includes instrument");
    }

    #[test]
    fn record_spawn_result_already_running_marks_skipped_no_reset() {
        let cfg = AutoRetrainConfig {
            enabled: true,
            bars_threshold: 5,
            instruments: vec!["EUR_USD".into()],
        };
        let auto = AutoRetrain::new(cfg);
        for _ in 0..7 {
            auto.counters.record_bar("EUR_USD");
        }
        let other_flight = crate::pipeline_trigger::PipelineFlight {
            run_id: "other".into(),
            instrument: "USD_JPY".into(),
            started_ms: 100,
            child_pid: None,
            log_path: None,
            finished_ms: None,
            exit_status: None,
        };
        let err = PipelineSpawnError::AlreadyRunning(other_flight);
        let fired = auto.record_spawn_result("EUR_USD", Err(&err));
        assert!(!fired);

        let snap = auto.counters.snapshot();
        assert_eq!(snap[0].bars_since_last_fire, 7, "counter unchanged");
        assert!(snap[0].last_fired_ms.is_none(), "no fire stamp");
        assert!(
            snap[0]
                .last_skip_reason
                .as_deref()
                .unwrap_or("")
                .contains("already running"),
            "skip reason mentions conflict"
        );

        let msg = auto.last_status_message.read().clone().unwrap();
        assert!(msg.contains("EUR_USD"));
        assert!(msg.contains("skipped"));
    }

    #[test]
    fn record_spawn_result_disabled_marks_skipped() {
        let cfg = AutoRetrainConfig {
            enabled: true,
            bars_threshold: 1,
            instruments: vec!["EUR_USD".into()],
        };
        let auto = AutoRetrain::new(cfg);
        auto.counters.record_bar("EUR_USD");
        let err = PipelineSpawnError::Disabled;
        assert!(!auto.record_spawn_result("EUR_USD", Err(&err)));
        let snap = auto.counters.snapshot();
        assert!(snap[0].last_skip_reason.is_some());
        assert!(snap[0].last_fired_ms.is_none());
    }

    #[test]
    fn config_from_env_parses_csv() {
        std::env::set_var("AUTO_RETRAIN_INSTRUMENTS", " EUR_USD ,USD_JPY,, ");
        std::env::set_var("AUTO_RETRAIN_BARS_THRESHOLD", "150");
        std::env::set_var("AUTO_RETRAIN_ENABLED", "true");
        let cfg = AutoRetrainConfig::from_env();
        std::env::remove_var("AUTO_RETRAIN_INSTRUMENTS");
        std::env::remove_var("AUTO_RETRAIN_BARS_THRESHOLD");
        std::env::remove_var("AUTO_RETRAIN_ENABLED");
        assert_eq!(cfg.bars_threshold, 150);
        assert!(cfg.enabled);
        assert_eq!(cfg.instruments, vec!["EUR_USD".to_string(), "USD_JPY".into()]);
    }
}

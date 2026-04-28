//! `POST /api/pipeline/run` + `GET /api/pipeline/status`.
//!
//! Lets the dashboard spawn one `python -m research pipeline run
//! --instrument <X>` invocation at a time. The spawned subprocess
//! writes its own `pipeline_runs` rows via `research.observability`,
//! so the dashboard's `PipelineRunsCard` (which polls
//! `/api/pipeline/runs`) shows progress without us doubling-up writes
//! from Rust.
//!
//! Single-flight semantics: while a run is in flight, additional POSTs
//! get a 409. The watcher task that owns the spawned `Child` clears
//! the flight slot on exit.
//!
//! Opt-in via `PIPELINE_TRIGGER_ENABLED=true`. Without it, POST /api/pipeline/run
//! returns 503 — keeps the route disabled by default in production
//! environments where shelling out to Python is undesirable.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};
use chrono::Utc;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::state::AppState;

#[derive(Clone, Debug, Serialize)]
pub struct PipelineFlight {
    pub run_id: String,
    pub instrument: String,
    pub started_ms: i64,
    pub child_pid: Option<u32>,
    /// Absolute path of the captured stdout/stderr log. `None` only
    /// during the brief window between flight acquisition and log file
    /// creation in `spawn_pipeline_subprocess`.
    pub log_path: Option<String>,
    /// Subprocess exit info — only set after the run completes. Lives
    /// here (not just on the slot) so the "last completed" snapshot
    /// the HTTP route returns carries it too.
    pub finished_ms: Option<i64>,
    pub exit_status: Option<String>,
}

#[derive(Default)]
pub struct PipelineFlightSlot {
    inner: Mutex<Option<PipelineFlight>>,
}

impl PipelineFlightSlot {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    pub fn current(&self) -> Option<PipelineFlight> {
        self.inner.lock().clone()
    }

    /// Try to claim the slot. Returns `Err(existing)` if a run is
    /// already in flight; the caller should turn that into a 409.
    pub fn try_acquire(&self, flight: PipelineFlight) -> Result<(), PipelineFlight> {
        let mut g = self.inner.lock();
        if let Some(existing) = g.as_ref() {
            return Err(existing.clone());
        }
        *g = Some(flight);
        Ok(())
    }

    /// Clear the slot only if the current flight matches `run_id`.
    /// Idempotent and concurrency-safe: if some other run has already
    /// taken over (impossible today, but defensive) we leave it alone.
    pub fn release(&self, run_id: &str) {
        let mut g = self.inner.lock();
        let matches = g.as_ref().map(|f| f.run_id == run_id).unwrap_or(false);
        if matches {
            *g = None;
        }
    }

    /// Build the "finalized" snapshot of the flight identified by
    /// `run_id`, leaving the slot intact. Caller is expected to call
    /// `release` immediately after persisting this snapshot.
    pub fn snapshot_finalized(
        &self,
        run_id: &str,
        finished_ms: i64,
        exit_status: Option<String>,
    ) -> Option<PipelineFlight> {
        let g = self.inner.lock();
        let f = g.as_ref()?;
        if f.run_id != run_id {
            return None;
        }
        Some(PipelineFlight {
            finished_ms: Some(finished_ms),
            exit_status,
            ..f.clone()
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct PipelineRunRequest {
    pub instrument: String,
    /// Override label/training window. Defaults match `PipelineConfig`.
    #[serde(default)]
    pub n_bars: Option<u32>,
    #[serde(default)]
    pub n_optuna_trials: Option<u32>,
    #[serde(default)]
    pub n_trader_trials: Option<u32>,
    #[serde(default)]
    pub seed: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct PipelineRunResponse {
    pub run_id: String,
    pub instrument: String,
    pub started_ms: i64,
    pub log_path: String,
}

pub fn pipeline_trigger_enabled() -> bool {
    std::env::var("PIPELINE_TRIGGER_ENABLED").as_deref() == Ok("true")
}

/// All the ways `spawn_pipeline_subprocess` can fail. The HTTP handler
/// maps each variant to a status code; the auto-retrain task just
/// logs and moves on.
#[derive(Debug)]
pub enum PipelineSpawnError {
    Disabled,
    EmptyInstrument,
    AlreadyRunning(PipelineFlight),
    LogFile(String),
    SpawnFailed(String),
}

impl std::fmt::Display for PipelineSpawnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disabled => write!(f, "PIPELINE_TRIGGER_ENABLED is not set"),
            Self::EmptyInstrument => write!(f, "instrument is required"),
            Self::AlreadyRunning(flight) => write!(
                f,
                "pipeline already running: instrument={} run_id={} started_ms={}",
                flight.instrument, flight.run_id, flight.started_ms
            ),
            Self::LogFile(msg) => write!(f, "log file: {msg}"),
            Self::SpawnFailed(msg) => write!(f, "spawn failed: {msg}"),
        }
    }
}

/// Core spawn routine. Both the REST handler and the auto-retrain
/// task call this; differences in error handling live at the call
/// site. Holds the flight slot for the lifetime of the spawned
/// subprocess (released by the watcher task on exit).
pub async fn spawn_pipeline_subprocess(
    state: &Arc<AppState>,
    request: PipelineRunRequest,
) -> Result<PipelineRunResponse, PipelineSpawnError> {
    if !pipeline_trigger_enabled() {
        return Err(PipelineSpawnError::Disabled);
    }
    if request.instrument.trim().is_empty() {
        return Err(PipelineSpawnError::EmptyInstrument);
    }

    let cfg = PipelineSpawnConfig::from_env();
    let run_id = next_run_id();
    let started_ms = Utc::now().timestamp_millis();

    // Try to claim the flight slot before doing any expensive work.
    let pending = PipelineFlight {
        run_id: run_id.clone(),
        instrument: request.instrument.clone(),
        started_ms,
        child_pid: None,
        log_path: None,
        finished_ms: None,
        exit_status: None,
    };
    if let Err(existing) = state.pipeline_flight.try_acquire(pending) {
        return Err(PipelineSpawnError::AlreadyRunning(existing));
    }

    // Prepare log file. Errors here release the flight before bailing.
    let log_path = prepare_log_file(&cfg.log_dir, &run_id).map_err(|e| {
        state.pipeline_flight.release(&run_id);
        PipelineSpawnError::LogFile(format!("could not prepare log file: {e}"))
    })?;
    let stdout_file = std::fs::File::create(&log_path).map_err(|e| {
        state.pipeline_flight.release(&run_id);
        PipelineSpawnError::LogFile(format!("could not open log file {}: {e}", log_path.display()))
    })?;
    let stderr_file = stdout_file.try_clone().map_err(|e| {
        state.pipeline_flight.release(&run_id);
        PipelineSpawnError::LogFile(format!("could not clone log file: {e}"))
    })?;

    // Spawn the Rust pipeline-orchestrator binary. It owns the staged
    // pipeline (label → train → finetune → lockbox → gate → export)
    // and only shells out to Python for the ML steps. Replaced the
    // legacy `python -m research pipeline run` direct spawn.
    let orchestrator = cfg
        .working_dir
        .join("server/target/release/pipeline-orchestrator");
    let mut cmd = tokio::process::Command::new(&orchestrator);
    cmd.arg("--instrument")
        .arg(&request.instrument)
        .current_dir(&cfg.working_dir)
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .stdin(Stdio::null());
    if let Some(n) = request.n_optuna_trials {
        cmd.arg("--side-trials").arg(n.to_string());
    }
    if let Some(n) = request.n_trader_trials {
        cmd.arg("--trader-trials").arg(n.to_string());
    }
    // pipeline-orchestrator doesn't currently expose --n-bars or
    // --seed; those flow through env / defaults.

    let mut child = cmd.spawn().map_err(|e| {
        state.pipeline_flight.release(&run_id);
        PipelineSpawnError::SpawnFailed(format!(
            "failed to spawn `{}`: {e}. Build it with `cargo build --release -p pipeline-orchestrator`.",
            orchestrator.display()
        ))
    })?;

    // Update the flight with the actual PID.
    {
        let mut g = state.pipeline_flight.inner.lock();
        if let Some(f) = g.as_mut() {
            f.child_pid = child.id();
        }
    }

    // Detach a watcher task that releases the flight slot on exit.
    let state_for_watcher = state.clone();
    let run_id_for_watcher = run_id.clone();
    let instrument_for_watcher = request.instrument.clone();
    tokio::spawn(async move {
        let exit_string = match child.wait().await {
            Ok(status) => {
                tracing::info!(
                    run_id = %run_id_for_watcher,
                    instrument = %instrument_for_watcher,
                    exit = %status,
                    "pipeline run exited"
                );
                Some(status.to_string())
            }
            Err(e) => {
                tracing::warn!(
                    run_id = %run_id_for_watcher,
                    error = %e,
                    "pipeline wait error"
                );
                Some(format!("wait error: {e}"))
            }
        };
        // Snapshot the final flight state into `last_completed` BEFORE
        // releasing the slot. Avoids a window where the slot is empty
        // and the dashboard sees neither current nor last.
        if let Some(final_flight) = state_for_watcher.pipeline_flight.snapshot_finalized(
            &run_id_for_watcher,
            Utc::now().timestamp_millis(),
            exit_string,
        ) {
            *state_for_watcher.last_completed_pipeline_flight.write() = Some(final_flight);
        }
        state_for_watcher.pipeline_flight.release(&run_id_for_watcher);
    });

    Ok(PipelineRunResponse {
        run_id,
        instrument: request.instrument,
        started_ms,
        log_path: log_path.display().to_string(),
    })
}

/// `POST /api/pipeline/run`. Body is `PipelineRunRequest`. On success
/// returns 202 + `PipelineRunResponse`.
pub async fn pipeline_run(
    State(state): State<Arc<AppState>>,
    Json(body): Json<PipelineRunRequest>,
) -> Result<(StatusCode, Json<PipelineRunResponse>), (StatusCode, String)> {
    match spawn_pipeline_subprocess(&state, body).await {
        Ok(resp) => Ok((StatusCode::ACCEPTED, Json(resp))),
        Err(PipelineSpawnError::Disabled) => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "PIPELINE_TRIGGER_ENABLED is not set; refusing to spawn".into(),
        )),
        Err(PipelineSpawnError::EmptyInstrument) => {
            Err((StatusCode::BAD_REQUEST, "instrument is required".into()))
        }
        Err(e @ PipelineSpawnError::AlreadyRunning(_)) => {
            Err((StatusCode::CONFLICT, format!("{e}")))
        }
        Err(e @ PipelineSpawnError::LogFile(_)) => {
            Err((StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
        }
        Err(e @ PipelineSpawnError::SpawnFailed(_)) => {
            Err((StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
        }
    }
}

#[derive(Serialize)]
pub struct PipelineStatusResponse {
    pub enabled: bool,
    pub current: Option<PipelineFlight>,
}

/// `GET /api/pipeline/status`. Surfaces whether the trigger is enabled
/// and what (if anything) is currently running. Used by the dashboard
/// to disable the "Run pipeline" button while a run is in flight.
pub async fn pipeline_status(
    State(state): State<Arc<AppState>>,
) -> Json<PipelineStatusResponse> {
    Json(PipelineStatusResponse {
        enabled: pipeline_trigger_enabled(),
        current: state.pipeline_flight.current(),
    })
}

/// `GET /api/pipeline/auto-retrain`. Per-instrument bar counters,
/// threshold, last-fired stamps, last skip reasons. The dashboard
/// uses this to render an "auto-retrain progress" indicator.
pub async fn auto_retrain_status(
    State(state): State<Arc<AppState>>,
) -> Json<crate::auto_retrain::AutoRetrainStatus> {
    let status = match &state.auto_retrain {
        Some(ar) => ar.status(),
        None => crate::auto_retrain::AutoRetrainStatus {
            enabled: false,
            bars_threshold: 0,
            instruments: vec![],
        },
    };
    Json(status)
}

#[derive(Deserialize)]
pub struct PipelineLogQuery {
    pub run_id: String,
    /// Tail size in bytes. Capped at 1 MB. Default 64 KB.
    #[serde(default)]
    pub tail: Option<usize>,
}

/// Allowed run_id alphabet: alphanumeric + `-` + `_`. Capped at 64 chars.
/// Used by both the homegrown Rust IDs (`<ms_hex>-<counter_hex>`) and
/// any future Python-supplied IDs that might piggy-back on this route.
/// Rejects path-traversal (`..`, `/`, `\`) by construction.
pub fn validate_run_id(s: &str) -> bool {
    if s.is_empty() || s.len() > 64 {
        return false;
    }
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// `GET /api/pipeline/log?run_id=<id>&tail=<bytes>`. Returns
/// `text/plain`. Validates `run_id` against the strict alphabet above
/// before joining it to the configured log directory — defends
/// against `..` traversal even though `run_id` is filtered upstream.
pub async fn pipeline_log(
    State(_state): State<Arc<AppState>>,
    axum::extract::Query(q): axum::extract::Query<PipelineLogQuery>,
) -> Result<axum::response::Response<axum::body::Body>, (StatusCode, String)> {
    if !validate_run_id(&q.run_id) {
        return Err((StatusCode::BAD_REQUEST, "invalid run_id format".into()));
    }
    let cfg = PipelineSpawnConfig::from_env();
    let log_path = cfg.log_dir.join(format!("{}.log", q.run_id));
    if !log_path.exists() {
        return Err((StatusCode::NOT_FOUND, "log not found".into()));
    }
    let tail = q.tail.unwrap_or(65_536).min(1_048_576);
    let body = match read_tail(&log_path, tail) {
        Ok(b) => b,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("read log {}: {e}", log_path.display()),
            ))
        }
    };
    Ok(axum::response::Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/plain; charset=utf-8")
        .header("cache-control", "no-store")
        .body(axum::body::Body::from(body))
        .expect("valid response"))
}

/// Read up to `tail` bytes from the end of `path`. Streams via
/// `seek` so we don't slurp a multi-GB log.
pub fn read_tail(path: &std::path::Path, tail: usize) -> std::io::Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len() as usize;
    let start = len.saturating_sub(tail);
    f.seek(SeekFrom::Start(start as u64))?;
    let mut buf = Vec::with_capacity(len.min(tail));
    f.take(tail as u64).read_to_end(&mut buf)?;
    // If we sliced into the middle of a UTF-8 sequence, drop bytes
    // until the next '\n' — keeps the response valid utf-8.
    if start > 0 {
        if let Some(nl) = buf.iter().position(|&b| b == b'\n') {
            buf.drain(..=nl);
        }
    }
    Ok(buf)
}

/// `GET /api/pipeline/last-completed`. Snapshot of the most-recent
/// finished pipeline subprocess (if any). Used by the dashboard to
/// keep "View log" available for ~the next run window.
pub async fn last_completed_pipeline(
    State(state): State<Arc<AppState>>,
) -> Json<Option<PipelineFlight>> {
    Json(state.last_completed_pipeline_flight.read().clone())
}

pub struct PipelineSpawnConfig {
    pub python_bin: PathBuf,
    pub working_dir: PathBuf,
    pub log_dir: PathBuf,
}

impl PipelineSpawnConfig {
    pub fn from_env() -> Self {
        // The Rust server runs from <repo>/server by convention; resolve
        // paths relative to the repo root so the venv + research/ live
        // at predictable locations.
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let repo_root = guess_repo_root(&cwd).unwrap_or(cwd);
        let python_bin = std::env::var("PIPELINE_PYTHON_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|_| repo_root.join(".venv").join("bin").join("python"));
        let working_dir = std::env::var("PIPELINE_RESEARCH_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| repo_root.join("research"));
        let log_dir = std::env::var("PIPELINE_LOG_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| repo_root.join("data").join("logs").join("pipeline"));
        Self {
            python_bin,
            working_dir,
            log_dir,
        }
    }
}

fn guess_repo_root(start: &std::path::Path) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        if ancestor.join("research").is_dir() && ancestor.join("server").is_dir() {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

fn prepare_log_file(log_dir: &std::path::Path, run_id: &str) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(log_dir)?;
    Ok(log_dir.join(format!("{run_id}.log")))
}

static RUN_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// `<unix_ms_hex>-<process_counter_hex>` — collision-free per process,
/// human-greppable, no extra dependencies. Good enough for run_ids
/// that the user pairs with a server-restart-bounded log file.
fn next_run_id() -> String {
    let ms = Utc::now().timestamp_millis();
    let n = RUN_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{ms:x}-{n:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flight(run_id: &str) -> PipelineFlight {
        PipelineFlight {
            run_id: run_id.into(),
            instrument: "EUR_USD".into(),
            started_ms: 0,
            child_pid: None,
            log_path: None,
            finished_ms: None,
            exit_status: None,
        }
    }

    #[test]
    fn try_acquire_succeeds_when_empty() {
        let slot = PipelineFlightSlot::new();
        assert!(slot.try_acquire(flight("a")).is_ok());
        let cur = slot.current().expect("must be set");
        assert_eq!(cur.run_id, "a");
    }

    #[test]
    fn try_acquire_fails_when_in_flight() {
        let slot = PipelineFlightSlot::new();
        slot.try_acquire(flight("a")).unwrap();
        let err = slot.try_acquire(flight("b")).unwrap_err();
        assert_eq!(err.run_id, "a");
        // Slot still belongs to "a", not "b".
        assert_eq!(slot.current().unwrap().run_id, "a");
    }

    #[test]
    fn release_clears_only_when_run_id_matches() {
        let slot = PipelineFlightSlot::new();
        slot.try_acquire(flight("a")).unwrap();
        // Releasing a different run_id is a no-op.
        slot.release("b");
        assert!(slot.current().is_some());
        slot.release("a");
        assert!(slot.current().is_none());
    }

    #[test]
    fn release_on_empty_slot_is_noop() {
        let slot = PipelineFlightSlot::new();
        slot.release("a"); // doesn't panic
        assert!(slot.current().is_none());
    }

    #[test]
    fn snapshot_finalized_returns_finished_metadata() {
        let slot = PipelineFlightSlot::new();
        slot.try_acquire(flight("abc")).unwrap();
        let snap = slot
            .snapshot_finalized("abc", 12345, Some("exit code: 0".into()))
            .expect("should match");
        assert_eq!(snap.finished_ms, Some(12345));
        assert_eq!(snap.exit_status.as_deref(), Some("exit code: 0"));
        assert_eq!(snap.run_id, "abc");
    }

    #[test]
    fn snapshot_finalized_returns_none_for_wrong_id() {
        let slot = PipelineFlightSlot::new();
        slot.try_acquire(flight("abc")).unwrap();
        assert!(slot.snapshot_finalized("xyz", 0, None).is_none());
    }

    #[test]
    fn validate_run_id_accepts_legitimate_ids() {
        assert!(validate_run_id("19a9b3c47f0-3e"));
        assert!(validate_run_id("abc_123"));
        assert!(validate_run_id(
            "0123456789abcdef0123456789abcdef"
        ));
    }

    #[test]
    fn validate_run_id_rejects_traversal() {
        assert!(!validate_run_id("../etc/passwd"));
        assert!(!validate_run_id("foo/bar"));
        assert!(!validate_run_id("foo\\bar"));
        assert!(!validate_run_id("foo bar"));
        assert!(!validate_run_id("foo;rm -rf /"));
    }

    #[test]
    fn validate_run_id_rejects_empty_and_oversized() {
        assert!(!validate_run_id(""));
        assert!(!validate_run_id(&"x".repeat(65)));
    }

    #[test]
    fn read_tail_returns_full_file_when_smaller_than_tail() {
        let tmp = std::env::temp_dir().join("rtk-tail-test-1.log");
        std::fs::write(&tmp, b"hello world").unwrap();
        let bytes = read_tail(&tmp, 1024).unwrap();
        assert_eq!(bytes, b"hello world");
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn read_tail_drops_partial_first_line_when_truncating() {
        let tmp = std::env::temp_dir().join("rtk-tail-test-2.log");
        std::fs::write(&tmp, b"line one\nline two\nline three\n").unwrap();
        // Ask for the last ~14 bytes — we expect the partial first line
        // (the tail of "line two") to be dropped, leaving "line three\n".
        let bytes = read_tail(&tmp, 14).unwrap();
        let s = String::from_utf8(bytes).unwrap();
        assert_eq!(s, "line three\n");
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn read_tail_handles_empty_file() {
        let tmp = std::env::temp_dir().join("rtk-tail-test-3.log");
        std::fs::write(&tmp, b"").unwrap();
        let bytes = read_tail(&tmp, 1024).unwrap();
        assert!(bytes.is_empty());
        std::fs::remove_file(&tmp).ok();
    }
}

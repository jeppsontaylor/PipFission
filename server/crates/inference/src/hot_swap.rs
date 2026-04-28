//! File-system watcher that reloads the champion when
//! `artifacts/models/live/manifest.json` changes. Debounces rapid
//! consecutive writes (the Python pipeline writes ONNX then manifest;
//! we want exactly one reload, not two).
//!
//! Returns a `tokio::sync::broadcast::Receiver<HotSwapEvent>` so the
//! api-server can surface ChampionChanged / ChampionFallback events on
//! the dashboard WS.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::{broadcast, mpsc};

use crate::registry::{live_champion_paths, PredictorRegistry};

/// Broadcast event emitted when the watcher swaps or fails to swap.
#[derive(Clone, Debug)]
pub enum HotSwapEvent {
    /// New champion loaded successfully. Carries the manifest's `model_id`.
    ChampionChanged { model_id: String },
    /// Tried to load but failed. Carries the error message.
    ChampionLoadFailed { reason: String },
    /// Fell back to neutral predictor (e.g. after repeated runtime errors).
    Fallback,
}

const DEBOUNCE_MS: u64 = 500;

/// Spawn the watcher. Returns a broadcast subscriber and a JoinHandle.
/// On startup it tries an immediate load so a server boot picks up the
/// most recent champion without waiting for a file change.
pub fn spawn_hot_swap_watcher(
    repo_root: PathBuf,
    registry: Arc<PredictorRegistry>,
) -> (broadcast::Receiver<HotSwapEvent>, tokio::task::JoinHandle<Result<()>>) {
    let (tx, rx) = broadcast::channel::<HotSwapEvent>(64);
    let tx_inner = tx.clone();
    let join = tokio::spawn(async move {
        let (_onnx_path, manifest_path) = live_champion_paths(&repo_root);
        let watch_dir = manifest_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| repo_root.clone());
        std::fs::create_dir_all(&watch_dir)
            .with_context(|| format!("create {}", watch_dir.display()))?;

        // Initial load attempt.
        attempt_swap(&registry, &manifest_path, &tx_inner);

        let (notif_tx, mut notif_rx) = mpsc::unbounded_channel::<Event>();
        let mut watcher: RecommendedWatcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
            if let Ok(ev) = res {
                let _ = notif_tx.send(ev);
            }
        })
        .with_context(|| "build notify watcher")?;
        watcher
            .watch(&watch_dir, RecursiveMode::NonRecursive)
            .with_context(|| format!("watch {}", watch_dir.display()))?;

        // Debounce loop: collect events for DEBOUNCE_MS, then attempt
        // a single reload if the manifest path was touched.
        loop {
            let ev = match notif_rx.recv().await {
                Some(ev) => ev,
                None => return Ok(()),
            };
            if !event_touches_manifest(&ev, &manifest_path) {
                continue;
            }
            // Drain any further events that arrive within the debounce
            // window so a paired (onnx, manifest) write triggers one swap.
            let drain_until = tokio::time::Instant::now() + Duration::from_millis(DEBOUNCE_MS);
            loop {
                let remaining = drain_until.saturating_duration_since(tokio::time::Instant::now());
                if remaining.is_zero() {
                    break;
                }
                let _ = tokio::time::timeout(remaining, notif_rx.recv()).await;
            }
            attempt_swap(&registry, &manifest_path, &tx_inner);
        }
    });
    (rx, join)
}

fn event_touches_manifest(ev: &Event, manifest_path: &std::path::Path) -> bool {
    ev.paths.iter().any(|p| p == manifest_path)
        || ev.paths.iter().any(|p| {
            // Some notify backends emit "create+rename" pairs on a different
            // tempfile path; treat any json under the watch dir as a trigger.
            p.extension().is_some_and(|e| e == "json") && p.file_name() == manifest_path.file_name()
        })
}

fn attempt_swap(
    registry: &PredictorRegistry,
    manifest_path: &std::path::Path,
    tx: &broadcast::Sender<HotSwapEvent>,
) {
    if !manifest_path.exists() {
        return;
    }
    match registry.try_load_onnx(manifest_path) {
        Ok(model_id) => {
            let _ = tx.send(HotSwapEvent::ChampionChanged { model_id });
        }
        Err(e) => {
            let reason = format!("{e:#}");
            tracing::warn!(reason = %reason, "inference: hot-swap failed; staying on prior predictor");
            let _ = tx.send(HotSwapEvent::ChampionLoadFailed { reason });
        }
    }
}

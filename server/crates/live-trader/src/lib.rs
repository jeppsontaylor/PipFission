//! Live trader state machine. Consumes `Event::Bar10s` and
//! `Event::ChampionSignal`, drives a per-instrument
//! `trader::Trader` configured from the latest `TraderParams` JSON
//! persisted by the research layer, and emits `Event::TraderDecision`
//! plus `trade_ledger` rows on close.
//!
//! ## Lifecycle
//!  - On startup: load the latest `trader_metrics` row, parse
//!    `params_json` as `TraderParams`. If no row exists, the runner
//!    stays idle (skips events) until one appears.
//!  - On `ChampionChanged`: reload `TraderParams` (the new champion
//!    likely has a fresh trader fine-tune attached).
//!  - On each `(Bar10s, ChampionSignal)` pair for the same
//!    `(instrument, ts_ms)`: compute the bar's sigma from a rolling
//!    buffer, advance the per-instrument `Trader`, emit a
//!    `TraderDecision` for *every* bar (including skips so the
//!    dashboard can render "trader is in cooldown / drawdown / etc").
//!  - On `Open` events: stash the entry context; emit a TraderDecision
//!    with `action="open_long"|"open_short"`.
//!  - On `Close` events: emit a TraderDecision with `action="close"`,
//!    insert a row into `trade_ledger`.
//!
//! The runner is **default off**. The api-server gates spawn behind
//! `LIVE_TRADER_ENABLED=true`. Until the user explicitly opts in, the
//! infrastructure runs in research-only mode.

#![deny(unsafe_code)]

pub mod forensics;
pub mod jsonl_log;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use chrono::{TimeZone, Utc};
use dashmap::DashMap;
use parking_lot::Mutex;
use tokio::sync::broadcast;

use market_domain::{Bar10s, Bar10sNamed, ChampionSignal, Event, TraderDecision};
use persistence::Db;
use trader::{Probs, Reason, Side, TradeEvent, Trader, TraderParams};

/// Largest rolling window the sigma estimator wants. Comfortably above
/// any window we care about for the EWMA σ, ~10 minutes at 10s bars.
const ROLLING_WINDOW_BARS: usize = 600;

/// Span used by the EWMA volatility estimate. Matches the live
/// inference/labelling defaults.
const SIGMA_SPAN: usize = 60;

/// Configuration for the live trader runner.
#[derive(Clone, Debug)]
pub struct LiveTraderConfig {
    /// Instruments the trader is allowed to act on. Bars/signals for
    /// other instruments are ignored. Empty = no filtering.
    pub instruments: Vec<String>,
    /// Tag every `trade_ledger` row with this run_id so trades from a
    /// given live session can be queried as a unit.
    pub run_id: String,
    /// How long to wait between TraderParams reload checks if the bus
    /// hasn't fired a `ChampionChanged`. Defensive against the DB being
    /// updated by a Python pipeline run that the live engine missed.
    pub reload_interval_secs: u64,
}

impl Default for LiveTraderConfig {
    fn default() -> Self {
        Self {
            instruments: Vec::new(),
            run_id: format!("live_{}", chrono::Utc::now().timestamp()),
            reload_interval_secs: 600,
        }
    }
}

/// Spawn the live trader runner. Returns immediately; the task ends
/// when the bus closes.
pub fn spawn(
    bus: broadcast::Sender<Event>,
    db: Db,
    cfg: LiveTraderConfig,
) -> tokio::task::JoinHandle<()> {
    let state = Arc::new(RuntimeState::new(cfg.run_id.clone()));
    let _ = state.try_reload_params(&db);
    let mut rx = bus.subscribe();
    let tx = bus.clone();
    let cfg2 = cfg.clone();
    let db2 = db.clone();
    let state2 = state.clone();

    // Periodic param reloader — defensive against missed
    // ChampionChanged events.
    tokio::spawn(async move {
        let interval = std::time::Duration::from_secs(cfg2.reload_interval_secs.max(30));
        loop {
            tokio::time::sleep(interval).await;
            let _ = state2.try_reload_params(&db2);
        }
    });

    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(Event::Bar10s(named)) => {
                    if !state.allows(&cfg, &named.instrument) {
                        continue;
                    }
                    state.record_bar(named);
                }
                Ok(Event::ChampionSignal(c)) => {
                    if !state.allows(&cfg, &c.instrument) {
                        continue;
                    }
                    if let Err(e) = on_champion_signal(&state, &db, &tx, c) {
                        tracing::warn!(error = %e, "live-trader: signal step failed");
                    }
                }
                Ok(Event::ChampionChanged(_)) => {
                    let _ = state.try_reload_params(&db);
                }
                Ok(_) => {}
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("live-trader: lagged {n} events");
                }
                Err(broadcast::error::RecvError::Closed) => return,
            }
        }
    })
}

// ----- runtime state ----------------------------------------------------

struct RuntimeState {
    run_id: String,
    params: Mutex<Option<LoadedParams>>,
    instruments: DashMap<String, InstrumentState>,
}

#[derive(Clone, Debug)]
struct LoadedParams {
    params: TraderParams,
    params_id: String,
    model_id: String,
    /// Inferred from `model_id` prefix (`side_lgbm_xxx` → `"lgbm"`).
    /// Used in the JSONL preview so reviewers can see at a glance what
    /// kind of classifier produced the signal.
    model_kind: String,
    /// Raw JSON of the trader params snapshot — preserved verbatim from
    /// `trader_metrics.params_json` so the JSONL preview can echo back
    /// the exact configuration the trader used.
    params_json: String,
    /// Best-effort fill from `model_metrics`. None when no row matched
    /// the active `model_id` (e.g. a fallback / cold-start model).
    oos_auc: Option<f64>,
    oos_log_loss: Option<f64>,
}

/// Parse `side_lgbm_1777266560_9df229` → `"lgbm"`. Falls back to the
/// full id when the convention isn't matched.
fn infer_model_kind(model_id: &str) -> String {
    let parts: Vec<&str> = model_id.splitn(3, '_').collect();
    if parts.len() >= 2 && parts[0] == "side" {
        parts[1].to_string()
    } else {
        model_id.to_string()
    }
}

struct InstrumentState {
    bars: Mutex<VecDeque<Bar10s>>,
    trader: Mutex<TraderEntry>,
    /// Bar idx since the last params reload. Used by the trader's
    /// min/max-hold counters.
    bar_idx: Mutex<u32>,
}

struct TraderEntry {
    trader: Trader,
    /// Open-position context: (entry_idx, entry_ts_ms, entry_price, side).
    /// Required to populate trade_ledger on close.
    open: Option<OpenLedgerCtx>,
}

/// Per-entry context the trader stashes on Open and reads back on
/// Close so the persisted `trade_ledger` row carries the full
/// decision context (Phase D enrichment). Public-but-crate-internal
/// because `forensics::write_trade_snapshot` borrows it.
#[derive(Clone, Debug)]
pub struct OpenLedgerCtx {
    entry_ts_ms: i64,
    entry_price: f64,
    side: Side,
    /// ChampionSignal probabilities at entry — captured directly off
    /// the signal that triggered the open. None only in the rare cold-
    /// start case where the trader opens before any signal arrived.
    entry_p_long: Option<f64>,
    entry_p_short: Option<f64>,
    entry_calibrated: Option<f64>,
    /// Bar-level context at entry: spread (bp) + ATR(14) computed from
    /// the rolling buffer.
    entry_spread_bp: Option<f64>,
    entry_atr_14: Option<f64>,
    /// Champion + params identifiers at entry. Decouples the trade
    /// from later hot-swaps so analysts know which model produced this
    /// trade even if the champion has rotated since.
    model_id: Option<String>,
    params_id: Option<String>,
    /// Decision rule label that fired at entry, e.g. "entry:p_long_clear".
    /// Filled out further on close to form the full decision_chain.
    entry_reason: String,
}

impl InstrumentState {
    fn new(params: &TraderParams) -> Self {
        Self {
            bars: Mutex::new(VecDeque::with_capacity(ROLLING_WINDOW_BARS + 1)),
            trader: Mutex::new(TraderEntry {
                trader: Trader::new(*params),
                open: None,
            }),
            bar_idx: Mutex::new(0),
        }
    }
}

impl RuntimeState {
    fn new(run_id: String) -> Self {
        Self {
            run_id,
            params: Mutex::new(None),
            instruments: DashMap::new(),
        }
    }

    fn allows(&self, cfg: &LiveTraderConfig, instrument: &str) -> bool {
        cfg.instruments.is_empty() || cfg.instruments.iter().any(|s| s == instrument)
    }

    fn record_bar(&self, named: Bar10sNamed) {
        let entry = self
            .instruments
            .entry(named.instrument.clone())
            .or_insert_with(|| {
                let p = self
                    .params
                    .lock()
                    .as_ref()
                    .map(|x| x.params)
                    .unwrap_or_default();
                InstrumentState::new(&p)
            });
        let mut buf = entry.value().bars.lock();
        buf.push_back(Bar10s {
            instrument_id: 0,
            ts_ms: named.ts_ms,
            open: named.open,
            high: named.high,
            low: named.low,
            close: named.close,
            n_ticks: named.n_ticks,
            spread_bp_avg: named.spread_bp_avg,
        });
        while buf.len() > ROLLING_WINDOW_BARS {
            buf.pop_front();
        }
    }

    /// Try loading the most recent trader_metrics row. Updates internal
    /// params if found; returns Ok(true) on a successful update,
    /// Ok(false) when nothing changed (no row, or same params_id).
    fn try_reload_params(&self, db: &Db) -> Result<bool> {
        let row = db
            .latest_trader_metrics()
            .context("query latest_trader_metrics")?;
        let Some(row) = row else {
            return Ok(false);
        };
        let parsed: TraderParams = serde_json::from_str(&row.params_json)
            .map_err(|e| anyhow!("parse params_json: {e}"))?;
        let mut slot = self.params.lock();
        let same = slot
            .as_ref()
            .map(|cur| cur.params_id == row.params_id)
            .unwrap_or(false);
        if same {
            return Ok(false);
        }
        // Best-effort lookup of model_metrics for OOS context. Logged
        // and ignored on failure — the JSONL preview just records None.
        let (oos_auc, oos_log_loss) = match db.model_metrics_by_id(&row.model_id) {
            Ok(Some(m)) => (Some(m.oos_auc), Some(m.oos_log_loss)),
            _ => (None, None),
        };
        let model_kind = infer_model_kind(&row.model_id);
        *slot = Some(LoadedParams {
            params: parsed,
            params_id: row.params_id.clone(),
            model_id: row.model_id.clone(),
            model_kind,
            params_json: row.params_json.clone(),
            oos_auc,
            oos_log_loss,
        });
        // Reset every per-instrument state so the new TraderParams
        // start from a clean slate (avoids stale stop-loss / cooldown
        // counters carrying across param swaps).
        for entry in self.instruments.iter() {
            *entry.value().trader.lock() = TraderEntry {
                trader: Trader::new(parsed),
                open: None,
            };
            *entry.value().bar_idx.lock() = 0;
        }
        tracing::info!(
            params_id = %row.params_id,
            model_id = %row.model_id,
            "live-trader: TraderParams reloaded"
        );
        Ok(true)
    }
}

// ----- per-signal step --------------------------------------------------

fn on_champion_signal(
    state: &RuntimeState,
    db: &Db,
    bus: &broadcast::Sender<Event>,
    signal: ChampionSignal,
) -> Result<()> {
    let loaded = state.params.lock().clone();
    let Some(loaded) = loaded else {
        // No TraderParams loaded yet — skip silently. The user hasn't
        // run the research pipeline yet, or the lockbox hasn't passed.
        return Ok(());
    };
    let entry = state
        .instruments
        .entry(signal.instrument.clone())
        .or_insert_with(|| InstrumentState::new(&loaded.params));
    // Pull the matching bar — the most recent buffered bar whose ts_ms
    // matches the signal's. live-inference emits ChampionSignal AFTER
    // emitting Bar10s for the same ts_ms, so the bar is virtually
    // always present at signal-time.
    let signal_ts_ms = signal.time.timestamp_millis();
    let bars: Vec<Bar10s> = entry.value().bars.lock().iter().copied().collect();
    let Some(bar) = bars.iter().rev().find(|b| b.ts_ms == signal_ts_ms).copied() else {
        // Bar might not have arrived yet (race), or it was already
        // shed. Either way, skipping this signal is safe — the next
        // bar's signal will produce a fresh decision.
        tracing::debug!(
            instrument = %signal.instrument,
            ts_ms = signal_ts_ms,
            "live-trader: no matching bar for signal"
        );
        return Ok(());
    };

    // Use bar-features::recompute_last as a sigma proxy is overkill;
    // a direct rolling-EWMA on close-to-close log returns matches the
    // labeling crate's default and keeps this module self-contained.
    let sigma = ewma_sigma(&bars, SIGMA_SPAN);

    let mut bar_idx_slot = entry.value().bar_idx.lock();
    let bar_idx = *bar_idx_slot;
    *bar_idx_slot += 1;
    drop(bar_idx_slot);

    let probs = Probs {
        p_long: signal.p_long,
        p_short: signal.p_short,
        p_take: signal.p_take,
        calibrated: signal.calibrated,
    };

    let mut tt = entry.value().trader.lock();
    let event = tt.trader.on_bar(bar_idx, &bar, &probs, sigma, 0);

    match event {
        TradeEvent::Open { side, bar_idx: bi, entry_px, reason } => {
            tt.open = Some(OpenLedgerCtx {
                entry_ts_ms: bar.ts_ms,
                entry_price: entry_px,
                side,
                entry_p_long: Some(signal.p_long),
                entry_p_short: Some(signal.p_short),
                entry_calibrated: Some(signal.calibrated),
                entry_spread_bp: Some(bar.spread_bp_avg),
                entry_atr_14: atr_14(&bars),
                model_id: Some(loaded.model_id.clone()),
                params_id: Some(loaded.params_id.clone()),
                entry_reason: reason_str(reason).to_string(),
            });
            emit_decision(
                bus,
                &signal.instrument,
                bar.ts_ms,
                bi,
                action_str(side),
                reason_str(reason),
                entry_px,
                None,
                &loaded,
            );
        }
        TradeEvent::Close { bar_idx: bi, exit_px, realized_r, reason } => {
            // Persist the round trip to trade_ledger using the open
            // context recorded at entry. Phase D3: also write a
            // per-trade JSON forensic snapshot so an analyst can
            // reconstruct WHY the model entered + how the market
            // moved during the hold without joining time-series
            // tables that may have been shed.
            if let Some(open) = tt.open.take() {
                let snapshot_path: Option<PathBuf> = match forensics::write_trade_snapshot(
                    &snapshots_root(),
                    &state.run_id,
                    &signal.instrument,
                    &open,
                    bar.ts_ms,
                    exit_px,
                    realized_r,
                    reason,
                    Some(signal.p_long),
                    Some(signal.p_short),
                    &bars,
                ) {
                    Ok(p) => Some(p),
                    Err(e) => {
                        tracing::warn!(error = %e, "live-trader: forensics snapshot failed");
                        None
                    }
                };
                let snap_str = snapshot_path
                    .as_ref()
                    .map(|p| p.display().to_string());
                if let Err(e) = insert_trade_ledger(
                    db,
                    &state.run_id,
                    &signal.instrument,
                    &open,
                    bar.ts_ms,
                    exit_px,
                    realized_r,
                    reason,
                    Some(signal.p_long),
                    Some(signal.p_short),
                    snap_str.as_deref(),
                ) {
                    tracing::warn!(error = %e, "live-trader: trade_ledger insert failed");
                }
                // Append the agent-readable JSONL preview row
                // (trade_logs/<v>/<ticker>/trades.jsonl). Rolls at 2k
                // lines. Heavy data already on disk in `data/trades/`.
                if let Err(e) = write_trade_jsonl(
                    &state.run_id,
                    &signal,
                    &open,
                    bar.ts_ms,
                    exit_px,
                    realized_r,
                    reason,
                    snap_str.as_deref(),
                    &loaded,
                ) {
                    tracing::warn!(error = %e, "live-trader: trades.jsonl append failed");
                }
                // Refresh summary.json + LATEST.md so an agent
                // browsing the repo always sees current per-ticker
                // state.
                if let Err(e) = update_ticker_summary(&signal.instrument) {
                    tracing::debug!(error = %e, "live-trader: summary update failed");
                }
            }
            emit_decision(
                bus,
                &signal.instrument,
                bar.ts_ms,
                bi,
                "close",
                reason_str(reason),
                exit_px,
                Some(realized_r),
                &loaded,
            );
        }
        TradeEvent::Skip { bar_idx: bi, reason } => {
            // Skips are still emitted so the dashboard can show
            // "trader is in cooldown" or "below threshold".
            emit_decision(
                bus,
                &signal.instrument,
                bar.ts_ms,
                bi,
                "skip",
                reason_str(reason),
                bar.close,
                None,
                &loaded,
            );
        }
    }
    Ok(())
}

fn emit_decision(
    bus: &broadcast::Sender<Event>,
    instrument: &str,
    ts_ms: i64,
    bar_idx: u32,
    action: &str,
    reason: &str,
    price: f64,
    realized_r: Option<f64>,
    loaded: &LoadedParams,
) {
    let time = Utc.timestamp_millis_opt(ts_ms).single().unwrap_or_else(Utc::now);
    let _ = bus.send(Event::TraderDecision(TraderDecision {
        instrument: instrument.to_string(),
        time,
        bar_idx,
        action: action.to_string(),
        reason: reason.to_string(),
        price,
        realized_r,
        params_id: loaded.params_id.clone(),
        model_id: loaded.model_id.clone(),
    }));
    // Mirror to the agent-readable JSONL previews:
    //   `actions.jsonl` — opens + closes only, full per-event record
    //   `skip_summary.jsonl` — compacted runs of consecutive skips
    // The split keeps signal-to-noise high so an agent reviewing the
    // file can see the decisions that actually moved the trader.
    let is_skip = action == "skip";
    if !is_skip {
        // First flush any accumulated skip-run for this ticker so the
        // skip_summary appears immediately before the action that
        // ended it.
        if let Some(rec) = SKIP_ACCUMULATOR.flush(instrument, ts_ms, loaded) {
            let _ = write_skip_summary_jsonl(&rec);
        }
        if let Err(e) = write_action_jsonl(
            instrument, ts_ms, action, reason, price, realized_r, loaded,
        ) {
            tracing::warn!(error = %e, "live-trader: actions.jsonl append failed");
        }
    } else {
        // Accumulate the skip; if the run is long enough flush a
        // checkpoint summary so we don't carry state forever.
        if let Some(rec) = SKIP_ACCUMULATOR.observe(instrument, ts_ms, reason, price, loaded) {
            let _ = write_skip_summary_jsonl(&rec);
        }
    }
}

/// Process-global skip accumulator. One `SkipRun` per ticker; each
/// observed skip extends the run, each non-skip flushes it. Runs flush
/// automatically every `SKIP_SUMMARY_FLUSH_AT` skips so we don't keep
/// state indefinitely if the trader is permanently parked.
const SKIP_SUMMARY_FLUSH_AT: u32 = 100;

#[derive(Clone, Debug)]
struct SkipRun {
    instrument: String,
    first_ts_ms: i64,
    last_ts_ms: i64,
    count: u32,
    by_reason: std::collections::BTreeMap<String, u32>,
    last_price: f64,
    model_id: String,
    params_id: String,
}

struct SkipAccumulator {
    inner: parking_lot::Mutex<std::collections::HashMap<String, SkipRun>>,
}

impl SkipAccumulator {
    fn new() -> Self {
        Self {
            inner: parking_lot::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Observe a new skip event. Returns `Some(SkipRun)` if the run
    /// should be flushed (count just hit `SKIP_SUMMARY_FLUSH_AT`).
    fn observe(
        &self,
        instrument: &str,
        ts_ms: i64,
        reason: &str,
        price: f64,
        loaded: &LoadedParams,
    ) -> Option<SkipRun> {
        let mut g = self.inner.lock();
        let entry = g.entry(instrument.to_string()).or_insert_with(|| SkipRun {
            instrument: instrument.to_string(),
            first_ts_ms: ts_ms,
            last_ts_ms: ts_ms,
            count: 0,
            by_reason: Default::default(),
            last_price: price,
            model_id: loaded.model_id.clone(),
            params_id: loaded.params_id.clone(),
        });
        entry.last_ts_ms = ts_ms;
        entry.count += 1;
        *entry.by_reason.entry(reason.to_string()).or_insert(0) += 1;
        entry.last_price = price;
        entry.model_id = loaded.model_id.clone();
        entry.params_id = loaded.params_id.clone();
        if entry.count >= SKIP_SUMMARY_FLUSH_AT {
            let snapshot = entry.clone();
            g.remove(instrument);
            Some(snapshot)
        } else {
            None
        }
    }

    /// Flush whatever skip-run is currently accumulated for `instrument`.
    fn flush(&self, instrument: &str, _now_ms: i64, _loaded: &LoadedParams) -> Option<SkipRun> {
        self.inner.lock().remove(instrument)
    }
}

static SKIP_ACCUMULATOR: std::sync::LazyLock<SkipAccumulator> =
    std::sync::LazyLock::new(SkipAccumulator::new);

#[allow(clippy::too_many_arguments)]
fn write_action_jsonl(
    instrument: &str,
    ts_ms: i64,
    action: &str,
    reason: &str,
    price: f64,
    realized_r: Option<f64>,
    loaded: &LoadedParams,
) -> Result<()> {
    use serde_json::json;
    let record = json!({
        "v": format!("v{RELEASE_VERSION}"),
        "instrument": instrument,
        "ts_ms": ts_ms,
        "action": action,
        "reason": reason,
        "price": price,
        "realized_r": realized_r,
        "model_id": loaded.model_id,
        "params_id": loaded.params_id,
    });
    let sub = ticker_sub_path(instrument, "actions.jsonl");
    jsonl_log::append(&trade_logs_root(), RELEASE_VERSION, &sub, &record)
}

fn write_skip_summary_jsonl(run: &SkipRun) -> Result<()> {
    use serde_json::json;
    let record = json!({
        "v": format!("v{RELEASE_VERSION}"),
        "instrument": run.instrument,
        "first_ts_ms": run.first_ts_ms,
        "last_ts_ms": run.last_ts_ms,
        "duration_ms": run.last_ts_ms - run.first_ts_ms,
        "count": run.count,
        "by_reason": &run.by_reason,
        "last_price": run.last_price,
        "model_id": run.model_id,
        "params_id": run.params_id,
    });
    let sub = ticker_sub_path(&run.instrument, "skip_summary.jsonl");
    jsonl_log::append(&trade_logs_root(), RELEASE_VERSION, &sub, &record)
}

/// Top-level dir into which `trade_logs/<v>/<ticker>/...jsonl` previews
/// are written. Anchored at the workspace root via
/// `CARGO_MANIFEST_DIR` so cargo tests / running from any subdir
/// doesn't pollute `server/crates/<crate>/trade_logs/`. Tests + ops
/// can still override with `TRADE_LOGS_DIR`.
fn trade_logs_root() -> PathBuf {
    if let Ok(dir) = std::env::var("TRADE_LOGS_DIR") {
        return PathBuf::from(dir);
    }
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .parent() // server/crates
        .and_then(|p| p.parent()) // server
        .and_then(|p| p.parent()) // repo root
        .map(|repo| repo.join("trade_logs"))
        .unwrap_or_else(|| PathBuf::from("./trade_logs"))
}

/// Cargo workspace version baked at compile time — the folder under
/// `trade_logs/` is named for it (e.g. `v1.0.1`). Bumping the workspace
/// version in `server/Cargo.toml` and rebuilding rolls future writes
/// into the new folder.
const RELEASE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build the relative sub-path for a per-ticker JSONL file. Forward
/// slashes in the ticker (BTC/USD on Alpaca) flatten to underscores so
/// they don't become directory separators.
fn ticker_sub_path(instrument: &str, file: &str) -> String {
    let safe = instrument.replace('/', "_");
    format!("{safe}/{file}")
}

#[allow(clippy::too_many_arguments)]
fn write_trade_jsonl(
    run_id: &str,
    signal: &ChampionSignal,
    open: &OpenLedgerCtx,
    exit_ts_ms: i64,
    exit_px: f64,
    realized_r: f64,
    reason: Reason,
    snapshot_path: Option<&str>,
    loaded: &LoadedParams,
) -> Result<()> {
    use serde_json::{json, Value};
    let side: i8 = match open.side {
        Side::Long => 1,
        Side::Short => -1,
    };
    let net_pnl = (side as f64) * (exit_px - open.entry_price);
    let trader_params: Value = serde_json::from_str(&loaded.params_json)
        .unwrap_or_else(|_| Value::String(loaded.params_json.clone()));
    let chain = vec![
        format!("entry:{}", open.entry_reason),
        format!("exit:{}", reason_str(reason)),
    ];
    let record = json!({
        "v": format!("v{RELEASE_VERSION}"),
        "instrument": signal.instrument,
        "run_id": run_id,
        "ts_in_ms": open.entry_ts_ms,
        "ts_out_ms": exit_ts_ms,
        "side": side,
        "qty": 1.0,
        "entry_px": open.entry_price,
        "exit_px": exit_px,
        "fee": 0.0,
        "slip": 0.0,
        "realized_r": realized_r,
        "net_pnl": net_pnl,
        "exit_reason": reason_str(reason),
        "model_id": loaded.model_id,
        "model_kind": loaded.model_kind,
        "params_id": loaded.params_id,
        "n_features": market_domain::FEATURE_DIM,
        "model_oos_auc": loaded.oos_auc,
        "model_oos_log_loss": loaded.oos_log_loss,
        "entry_p_long": open.entry_p_long,
        "entry_p_short": open.entry_p_short,
        "entry_calibrated": open.entry_calibrated,
        "entry_spread_bp": open.entry_spread_bp,
        "entry_atr_14": open.entry_atr_14,
        "exit_p_long": signal.p_long,
        "exit_p_short": signal.p_short,
        "decision_chain": chain,
        "trader_params": trader_params,
        "snapshot_path": snapshot_path,
    });
    let sub = ticker_sub_path(&signal.instrument, "trades.jsonl");
    jsonl_log::append(&trade_logs_root(), RELEASE_VERSION, &sub, &record)
}

fn action_str(side: Side) -> &'static str {
    match side {
        Side::Long => "open_long",
        Side::Short => "open_short",
    }
}

fn reason_str(reason: Reason) -> &'static str {
    match reason {
        Reason::Signal => "signal",
        Reason::StopLoss => "stop_loss",
        Reason::TakeProfit => "take_profit",
        Reason::TrailingStop => "trailing_stop",
        Reason::MaxHold => "max_hold",
        Reason::Reverse => "reverse",
        Reason::SpreadTooWide => "spread_too_wide",
        Reason::StaleData => "stale_data",
        Reason::DailyLossKill => "daily_loss_kill",
        Reason::DrawdownPause => "drawdown_pause",
        Reason::Cooldown => "cooldown",
        Reason::BelowThreshold => "below_threshold",
        Reason::MinHold => "min_hold",
    }
}

/// Persist a closed round-trip. Phase D: the row carries the entry
/// signal context (captured at open) plus exit signal probs (captured
/// from the signal that triggered the close).
#[allow(clippy::too_many_arguments)]
fn insert_trade_ledger(
    db: &Db,
    run_id: &str,
    instrument: &str,
    open: &OpenLedgerCtx,
    exit_ts_ms: i64,
    exit_px: f64,
    realized_r: f64,
    reason: Reason,
    exit_p_long: Option<f64>,
    exit_p_short: Option<f64>,
    snapshot_path: Option<&str>,
) -> Result<()> {
    let side: i8 = match open.side {
        Side::Long => 1,
        Side::Short => -1,
    };
    let exit_reason = reason_str(reason);
    let chain_json = build_decision_chain_json(&open.entry_reason, exit_reason);
    let row = persistence::TradeLedgerInsert {
        run_id,
        instrument,
        ts_in_ms: open.entry_ts_ms,
        ts_out_ms: exit_ts_ms,
        side,
        qty: 1.0,
        entry_px: open.entry_price,
        exit_px,
        fee: 0.0,
        slip: 0.0,
        realized_r,
        exit_reason,
        model_id: open.model_id.as_deref(),
        params_id: open.params_id.as_deref(),
        entry_p_long: open.entry_p_long,
        entry_p_short: open.entry_p_short,
        entry_calibrated: open.entry_calibrated,
        entry_spread_bp: open.entry_spread_bp,
        entry_atr_14: open.entry_atr_14,
        exit_p_long,
        exit_p_short,
        decision_chain: Some(&chain_json),
        snapshot_path,
    };
    db.insert_trade_ledger(&row)?;
    Ok(())
}

/// Compose the JSON-encoded decision chain from the entry + exit
/// reason labels. Free-form schema: `["entry:<reason>", "exit:<reason>"]`.
fn build_decision_chain_json(entry: &str, exit_reason: &str) -> String {
    let chain = [
        format!("entry:{entry}"),
        format!("exit:{exit_reason}"),
    ];
    serde_json::to_string(&chain).unwrap_or_else(|_| "[]".to_string())
}

/// Root directory for per-trade forensic JSON snapshots. Reads
/// `LIVE_TRADER_SNAPSHOT_DIR` env at call time so tests + ops can
/// override; defaults to `./data/trades` (relative to the api-server's
/// working directory).
/// Refresh the per-ticker `summary.json` and `LATEST.md` previews
/// after any trade-affecting event (open captures the entry context;
/// close commits the round-trip). Reads back from the just-written
/// trades.jsonl + actions.jsonl + features.jsonl so the artifacts
/// stay self-consistent.
///
/// Failures are logged + swallowed at the call site; this helper is
/// for review-time observability, not load-bearing.
fn update_ticker_summary(instrument: &str) -> Result<()> {
    let root = trade_logs_root().join(version_folder(RELEASE_VERSION));
    let safe = instrument.replace('/', "_");
    let ticker_dir = root.join(&safe);
    let trades_path = ticker_dir.join("trades.jsonl");
    let actions_path = ticker_dir.join("actions.jsonl");
    let features_path = ticker_dir.join("features.jsonl");

    let trades = read_jsonl_lines(&trades_path).unwrap_or_default();
    let actions = read_jsonl_lines(&actions_path).unwrap_or_default();
    let features = read_jsonl_lines(&features_path).unwrap_or_default();

    // Aggregate trade stats.
    let n_trades = trades.len();
    let mut wins = 0_usize;
    let mut losses = 0_usize;
    let mut cum_r = 0.0_f64;
    let mut by_reason: std::collections::BTreeMap<String, u32> =
        std::collections::BTreeMap::new();
    let mut latest_trade_ts: Option<i64> = None;
    for t in trades.iter() {
        let r = t.get("realized_r").and_then(|v| v.as_f64()).unwrap_or(0.0);
        cum_r += r;
        if r > 0.0 {
            wins += 1;
        } else if r < 0.0 {
            losses += 1;
        }
        if let Some(reason) = t.get("exit_reason").and_then(|v| v.as_str()) {
            *by_reason.entry(reason.to_string()).or_insert(0) += 1;
        }
        if let Some(ts) = t.get("ts_out_ms").and_then(|v| v.as_i64()) {
            latest_trade_ts = Some(latest_trade_ts.map_or(ts, |cur| cur.max(ts)));
        }
    }
    let hit_rate = if (wins + losses) > 0 {
        wins as f64 / (wins + losses) as f64
    } else {
        0.0
    };
    let latest_action = actions.last();
    let latest_feature = features.last();
    let model_id_now = latest_action
        .and_then(|a| a.get("model_id").and_then(|v| v.as_str()))
        .or_else(|| latest_feature.and_then(|f| f.get("model_id").and_then(|v| v.as_str())))
        .unwrap_or("(unknown)")
        .to_string();
    let p_long_now = latest_feature.and_then(|f| f.get("p_long").and_then(|v| v.as_f64()));
    let p_short_now =
        latest_feature.and_then(|f| f.get("p_short").and_then(|v| v.as_f64()));
    let close_now = latest_feature.and_then(|f| f.get("close").and_then(|v| v.as_f64()));
    let last_action_str = latest_action
        .and_then(|a| a.get("action").and_then(|v| v.as_str()))
        .unwrap_or("(none)")
        .to_string();

    let summary = serde_json::json!({
        "v": format!("v{RELEASE_VERSION}"),
        "instrument": instrument,
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "hit_rate": hit_rate,
        "cum_r": cum_r,
        "by_exit_reason": by_reason,
        "latest_trade_ts_ms": latest_trade_ts,
        "latest_action": last_action_str,
        "latest_close": close_now,
        "latest_p_long": p_long_now,
        "latest_p_short": p_short_now,
        "current_model_id": model_id_now,
        "summary_updated_ms": chrono::Utc::now().timestamp_millis(),
    });

    // Atomic write: tmp + rename.
    std::fs::create_dir_all(&ticker_dir)?;
    let summary_path = ticker_dir.join("summary.json");
    let summary_tmp = ticker_dir.join("summary.json.tmp");
    std::fs::write(&summary_tmp, serde_json::to_string_pretty(&summary)?)?;
    std::fs::rename(&summary_tmp, &summary_path)?;

    // Render LATEST.md from the same buffers.
    let latest_md = render_latest_md(instrument, &summary, &trades, &actions, &features);
    let md_path = ticker_dir.join("LATEST.md");
    let md_tmp = ticker_dir.join("LATEST.md.tmp");
    std::fs::write(&md_tmp, latest_md)?;
    std::fs::rename(&md_tmp, &md_path)?;
    Ok(())
}

fn read_jsonl_lines(path: &std::path::Path) -> Result<Vec<serde_json::Value>> {
    use std::io::{BufRead, BufReader};
    if !path.exists() {
        return Ok(Vec::new());
    }
    let f = std::fs::File::open(path)?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for line in reader.lines().map_while(|r| r.ok()) {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
            out.push(v);
        }
    }
    Ok(out)
}

fn version_folder(v: &str) -> String {
    if v.starts_with('v') {
        v.to_string()
    } else {
        format!("v{v}")
    }
}

fn render_latest_md(
    instrument: &str,
    summary: &serde_json::Value,
    trades: &[serde_json::Value],
    actions: &[serde_json::Value],
    features: &[serde_json::Value],
) -> String {
    let mut buf = String::new();
    buf.push_str(&format!("# {instrument} — latest\n\n"));
    buf.push_str(&format!(
        "_Auto-generated by live-trader at {}_\n\n",
        chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ")
    ));
    buf.push_str("## At a glance\n\n");
    buf.push_str(&format!(
        "- **Trades closed**: {}\n",
        summary.get("n_trades").and_then(|v| v.as_u64()).unwrap_or(0)
    ));
    buf.push_str(&format!(
        "- **Wins / losses**: {} / {} (hit rate {:.1}%)\n",
        summary.get("wins").and_then(|v| v.as_u64()).unwrap_or(0),
        summary.get("losses").and_then(|v| v.as_u64()).unwrap_or(0),
        100.0 * summary.get("hit_rate").and_then(|v| v.as_f64()).unwrap_or(0.0),
    ));
    buf.push_str(&format!(
        "- **Cumulative realised R**: {:+.6}\n",
        summary.get("cum_r").and_then(|v| v.as_f64()).unwrap_or(0.0)
    ));
    buf.push_str(&format!(
        "- **Current champion**: `{}`\n",
        summary
            .get("current_model_id")
            .and_then(|v| v.as_str())
            .unwrap_or("(none)")
    ));
    if let (Some(pl), Some(ps)) = (
        summary.get("latest_p_long").and_then(|v| v.as_f64()),
        summary.get("latest_p_short").and_then(|v| v.as_f64()),
    ) {
        buf.push_str(&format!(
            "- **Latest probs**: p_long={:.4}, p_short={:.4}\n",
            pl, ps
        ));
    }
    buf.push_str(&format!(
        "- **Latest action**: `{}`\n\n",
        summary
            .get("latest_action")
            .and_then(|v| v.as_str())
            .unwrap_or("(none)")
    ));

    // Last 5 trades pretty-printed.
    buf.push_str("## Last 5 closed trades\n\n");
    if trades.is_empty() {
        buf.push_str("_No trades yet._\n\n");
    } else {
        buf.push_str(
            "| ts_out (UTC) | side | entry | exit | realized_r | reason | model |\n",
        );
        buf.push_str(
            "| --- | --- | --- | --- | --- | --- | --- |\n",
        );
        let n = trades.len();
        for t in trades.iter().skip(n.saturating_sub(5)) {
            let ts = t.get("ts_out_ms").and_then(|v| v.as_i64()).unwrap_or(0);
            let side = t.get("side").and_then(|v| v.as_i64()).unwrap_or(0);
            let entry = t.get("entry_px").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let exit = t.get("exit_px").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let r = t.get("realized_r").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let reason = t
                .get("exit_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let model = t
                .get("model_kind")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let when = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ts)
                .map(|d| d.format("%H:%M:%S").to_string())
                .unwrap_or_else(|| ts.to_string());
            let side_str = match side {
                1 => "long",
                -1 => "short",
                _ => "?",
            };
            buf.push_str(&format!(
                "| {when} | {side_str} | {entry:.5} | {exit:.5} | {r:+.6} | {reason} | {model} |\n",
            ));
        }
        buf.push('\n');
    }

    // Last 5 actions.
    buf.push_str("## Last 5 actions (open / close)\n\n");
    if actions.is_empty() {
        buf.push_str("_No actions yet._\n\n");
    } else {
        buf.push_str("| ts (UTC) | action | reason | price | realized_r |\n");
        buf.push_str("| --- | --- | --- | --- | --- |\n");
        let n = actions.len();
        for a in actions.iter().skip(n.saturating_sub(5)) {
            let ts = a.get("ts_ms").and_then(|v| v.as_i64()).unwrap_or(0);
            let action = a.get("action").and_then(|v| v.as_str()).unwrap_or("?");
            let reason = a.get("reason").and_then(|v| v.as_str()).unwrap_or("?");
            let price = a.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let r = a
                .get("realized_r")
                .and_then(|v| v.as_f64())
                .map(|x| format!("{x:+.6}"))
                .unwrap_or_else(|| "—".to_string());
            let when = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ts)
                .map(|d| d.format("%H:%M:%S").to_string())
                .unwrap_or_else(|| ts.to_string());
            buf.push_str(&format!("| {when} | {action} | {reason} | {price:.5} | {r} |\n"));
        }
        buf.push('\n');
    }

    // Last 10 features summary (close + p_long).
    buf.push_str("## Last 10 feature snapshots\n\n");
    if features.is_empty() {
        buf.push_str("_No features yet._\n\n");
    } else {
        buf.push_str("| ts (UTC) | close | spread_bp | p_long | p_short | calibrated |\n");
        buf.push_str("| --- | --- | --- | --- | --- | --- |\n");
        let n = features.len();
        for f in features.iter().skip(n.saturating_sub(10)) {
            let ts = f.get("ts_ms").and_then(|v| v.as_i64()).unwrap_or(0);
            let close = f.get("close").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let spread = f
                .get("spread_bp_avg")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let p_long = f.get("p_long").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let p_short = f.get("p_short").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let cal = f.get("calibrated").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let when = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ts)
                .map(|d| d.format("%H:%M:%S").to_string())
                .unwrap_or_else(|| ts.to_string());
            buf.push_str(&format!(
                "| {when} | {close:.5} | {spread:.2} | {p_long:.4} | {p_short:.4} | {cal:.4} |\n"
            ));
        }
        buf.push('\n');
    }

    buf.push_str("---\n\n");
    buf.push_str("_See `summary.json` for the full at-a-glance object, or `trades.jsonl` / `features.jsonl` / `actions.jsonl` for full per-event records._\n");
    buf
}

fn snapshots_root() -> PathBuf {
    std::env::var("LIVE_TRADER_SNAPSHOT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./data/trades"))
}

/// Compute ATR(14) from a rolling-buffer view of recent bars. Mirrors
/// `bar_features::atr_14` but is local — `bar-features` isn't a
/// dependency of `live-trader`, and the formula is small enough to
/// inline.
fn atr_14(bars: &[Bar10s]) -> Option<f64> {
    let window = 14usize;
    if bars.len() < window + 1 {
        return None;
    }
    let n = bars.len();
    let mut sum_tr = 0.0_f64;
    for k in (n - window)..n {
        let prev_close = bars[k - 1].close;
        let h = bars[k].high;
        let l = bars[k].low;
        let tr = (h - l)
            .max((h - prev_close).abs())
            .max((l - prev_close).abs());
        sum_tr += tr;
    }
    Some(sum_tr / window as f64)
}

// ----- helpers ----------------------------------------------------------

fn ewma_sigma(bars: &[Bar10s], span: usize) -> f64 {
    if bars.len() < 2 || span < 2 {
        return 0.0;
    }
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut s2 = 0.0_f64;
    for i in 1..bars.len() {
        let r = (bars[i].close.max(1e-12) / bars[i - 1].close.max(1e-12)).ln();
        let r2 = r * r;
        s2 = if i == 1 { r2 } else { alpha * r2 + (1.0 - alpha) * s2 };
    }
    s2.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewma_sigma_zero_for_constant_series() {
        let bars: Vec<_> = (0..100)
            .map(|i| Bar10s {
                instrument_id: 0,
                ts_ms: i * 10_000,
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 1.0,
                n_ticks: 1,
                spread_bp_avg: 0.5,
            })
            .collect();
        assert_eq!(ewma_sigma(&bars, 60), 0.0);
    }

    #[test]
    fn ewma_sigma_positive_for_volatile_series() {
        let bars: Vec<_> = (0..100)
            .map(|i| {
                let p = 1.0 + (i as f64 * 0.01).sin() * 0.005;
                Bar10s {
                    instrument_id: 0,
                    ts_ms: i * 10_000,
                    open: p,
                    high: p,
                    low: p,
                    close: p,
                    n_ticks: 1,
                    spread_bp_avg: 0.5,
                }
            })
            .collect();
        assert!(ewma_sigma(&bars, 60) > 0.0);
    }

    #[test]
    fn trader_starts_flat() {
        // Sanity: a fresh Trader is Flat and configured from the
        // provided TraderParams.
        let t = Trader::new(TraderParams::default());
        match t.state {
            trader::State::Flat => {}
            other => panic!("expected Flat, got {:?}", other),
        }
    }
}

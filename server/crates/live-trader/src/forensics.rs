//! Per-trade forensic snapshot writer.
//!
//! On every closed round-trip, the live trader emits one JSON file at
//! `data/trades/<run_id>/<ts_out_ms>.json` that bundles:
//!
//!   * the trade row itself (entry/exit price, side, qty, realized R,
//!     exit reason),
//!   * a pre-trade window of bars (the 50 bars before entry — gives
//!     reviewers the regime context, "what did the market look like
//!     right before we entered?"),
//!   * an in-trade window of bars (every bar from entry through
//!     exit — answers "what did the trader actually see during the
//!     hold?"),
//!   * a post-trade window (10 bars after exit — answers "should we
//!     have held longer?"),
//!   * the model_id + params_id active at entry,
//!   * the entry signal (p_long, p_short, calibrated) and the exit
//!     signal probabilities,
//!   * the decision-rule chain (entry reason + exit reason).
//!
//! The file is written atomically (tmp + rename) so an in-flight
//! reader can't see a half-written snapshot. Files are gitignored
//! (`data/trades/` is in .gitignore as part of the broader `data/`
//! exclusion). They accumulate forever — operators prune by run_id
//! when needed.
//!
//! The snapshot path is recorded back into `trade_ledger.snapshot_path`
//! so the `/api/trade/context` endpoint can find it via a single DB
//! lookup.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};
use market_domain::Bar10s;
use serde::Serialize;
use trader::{Reason, Side};

use crate::OpenLedgerCtx;

/// How many bars to capture before/after the trade. These are
/// generous defaults — `pre_window=50` gives a reviewer 8 minutes of
/// 10s-bar context at entry, `post_window=10` gives 100 seconds of
/// "what happened after" so they can judge whether the exit was
/// premature.
pub const PRE_TRADE_WINDOW_BARS: usize = 50;
pub const POST_TRADE_WINDOW_BARS: usize = 10;

#[derive(Serialize)]
struct BarSnapshot {
    ts_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    n_ticks: u32,
    spread_bp_avg: f64,
}

impl From<&Bar10s> for BarSnapshot {
    fn from(b: &Bar10s) -> Self {
        Self {
            ts_ms: b.ts_ms,
            open: b.open,
            high: b.high,
            low: b.low,
            close: b.close,
            n_ticks: b.n_ticks,
            spread_bp_avg: b.spread_bp_avg,
        }
    }
}

#[derive(Serialize)]
struct TradeSnapshot<'a> {
    schema_version: u32,
    run_id: &'a str,
    instrument: &'a str,
    side: i8,
    /// Always 1.0 in paper mode today; reserved for future
    /// position-sized live runs.
    qty: f64,
    entry_ts_ms: i64,
    entry_price: f64,
    exit_ts_ms: i64,
    exit_price: f64,
    realized_r: f64,
    exit_reason: &'a str,
    /// ISO-8601 versions of the entry/exit timestamps (UTC) so the
    /// JSON is grep-friendly without the reviewer doing epoch math.
    entry_iso: String,
    exit_iso: String,
    /// Champion + params identifiers in effect at entry.
    model_id: Option<&'a str>,
    params_id: Option<&'a str>,
    /// Entry signal probabilities (None only for cold-start trades).
    entry_p_long: Option<f64>,
    entry_p_short: Option<f64>,
    entry_calibrated: Option<f64>,
    entry_spread_bp: Option<f64>,
    entry_atr_14: Option<f64>,
    /// Exit signal probabilities — captured at the bar that triggered
    /// the close. None when the close was driven by a non-signal
    /// barrier (stop, max hold) and no signal was current.
    exit_p_long: Option<f64>,
    exit_p_short: Option<f64>,
    /// Free-form decision rule chain, e.g. ["entry:signal", "exit:reverse"].
    decision_chain: Vec<String>,
    /// Pre-/in-/post-trade bar windows.
    pre_trade_bars: Vec<BarSnapshot>,
    in_trade_bars: Vec<BarSnapshot>,
    post_trade_bars: Vec<BarSnapshot>,
}

/// Build the snapshot file path for a (run_id, ts_out_ms) pair.
/// Returns `<root>/<run_id>/<ts_out_ms>.json`.
pub fn snapshot_path(root: &Path, run_id: &str, ts_out_ms: i64) -> PathBuf {
    root.join(run_id).join(format!("{ts_out_ms}.json"))
}

/// Write the per-trade JSON snapshot atomically. Returns the
/// destination path on success so the caller can record it on the
/// `trade_ledger` row.
///
/// Errors are logged but never crash the live trader — forensic
/// snapshots are observability, not load-bearing logic.
#[allow(clippy::too_many_arguments)]
pub fn write_trade_snapshot(
    root: &Path,
    run_id: &str,
    instrument: &str,
    open: &OpenLedgerCtx,
    exit_ts_ms: i64,
    exit_px: f64,
    realized_r: f64,
    exit_reason: Reason,
    exit_p_long: Option<f64>,
    exit_p_short: Option<f64>,
    bars: &[Bar10s],
) -> Result<PathBuf> {
    let exit_reason_str = reason_str(exit_reason);
    let chain = vec![
        format!("entry:{}", open.entry_reason),
        format!("exit:{exit_reason_str}"),
    ];

    // Slice the bars buffer into pre/in/post windows. The buffer is
    // ordered ascending by ts_ms; find the entry/exit indices.
    let entry_idx = bars
        .iter()
        .position(|b| b.ts_ms == open.entry_ts_ms);
    let exit_idx = bars
        .iter()
        .position(|b| b.ts_ms == exit_ts_ms);

    let (pre_bars, in_bars, post_bars) = match (entry_idx, exit_idx) {
        (Some(ei), Some(xi)) if xi >= ei => {
            let pre_lo = ei.saturating_sub(PRE_TRADE_WINDOW_BARS);
            let post_hi = (xi + POST_TRADE_WINDOW_BARS + 1).min(bars.len());
            (
                bars[pre_lo..ei].iter().map(BarSnapshot::from).collect(),
                bars[ei..=xi].iter().map(BarSnapshot::from).collect(),
                bars[xi + 1..post_hi]
                    .iter()
                    .map(BarSnapshot::from)
                    .collect(),
            )
        }
        // Entry bar already shed from the buffer or other edge case:
        // emit empty windows rather than failing the whole snapshot.
        _ => (Vec::new(), Vec::new(), Vec::new()),
    };

    let entry_iso = Utc
        .timestamp_millis_opt(open.entry_ts_ms)
        .single()
        .map(|t| t.to_rfc3339())
        .unwrap_or_default();
    let exit_iso = Utc
        .timestamp_millis_opt(exit_ts_ms)
        .single()
        .map(|t| t.to_rfc3339())
        .unwrap_or_default();
    let side: i8 = match open.side {
        Side::Long => 1,
        Side::Short => -1,
    };

    let snapshot = TradeSnapshot {
        schema_version: 1,
        run_id,
        instrument,
        side,
        qty: 1.0,
        entry_ts_ms: open.entry_ts_ms,
        entry_price: open.entry_price,
        exit_ts_ms,
        exit_price: exit_px,
        realized_r,
        exit_reason: exit_reason_str,
        entry_iso,
        exit_iso,
        model_id: open.model_id.as_deref(),
        params_id: open.params_id.as_deref(),
        entry_p_long: open.entry_p_long,
        entry_p_short: open.entry_p_short,
        entry_calibrated: open.entry_calibrated,
        entry_spread_bp: open.entry_spread_bp,
        entry_atr_14: open.entry_atr_14,
        exit_p_long,
        exit_p_short,
        decision_chain: chain,
        pre_trade_bars: pre_bars,
        in_trade_bars: in_bars,
        post_trade_bars: post_bars,
    };

    let path = snapshot_path(root, run_id, exit_ts_ms);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("create snapshot dir")?;
    }
    let tmp = path.with_extension("json.tmp");
    let body = serde_json::to_vec_pretty(&snapshot).context("serialize snapshot")?;
    fs::write(&tmp, &body).context("write tmp snapshot")?;
    fs::rename(&tmp, &path).context("rename snapshot into place")?;
    Ok(path)
}

fn reason_str(r: Reason) -> &'static str {
    // Mirror the canonical mapping in lib.rs::reason_str. Duplicated
    // here to keep this module self-contained — there's only one
    // "trade is closing" code path so drift is unlikely.
    match r {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn mk_bar(ts_ms: i64, c: f64) -> Bar10s {
        Bar10s {
            instrument_id: 0,
            ts_ms,
            open: c,
            high: c + 1e-4,
            low: c - 1e-4,
            close: c,
            n_ticks: 100,
            spread_bp_avg: 1.0,
        }
    }

    fn mk_open() -> OpenLedgerCtx {
        OpenLedgerCtx {
            entry_ts_ms: 100_000,
            entry_price: 1.10,
            side: Side::Long,
            entry_p_long: Some(0.7),
            entry_p_short: Some(0.3),
            entry_calibrated: Some(0.7),
            entry_spread_bp: Some(1.5),
            entry_atr_14: Some(0.0001),
            model_id: Some("test_model".into()),
            params_id: Some("test_params".into()),
            entry_reason: "signal".into(),
        }
    }

    #[test]
    fn snapshot_writes_atomically_and_round_trips() {
        let tmp = tempfile_path("trade_snap_atomic");
        let bars: Vec<_> = (0..200)
            .map(|i| mk_bar(i * 10_000 + 100_000 - 100 * 10_000, 1.10 + (i as f64) * 1e-5))
            .collect();
        let path = write_trade_snapshot(
            &tmp,
            "run-test",
            "EUR_USD",
            &mk_open(),
            150_000, // exit_ts after entry_ts
            1.1005,
            0.005,
            Reason::TakeProfit,
            Some(0.55),
            Some(0.45),
            &bars,
        )
        .expect("snapshot");
        assert!(path.exists());
        let body = fs::read_to_string(&path).unwrap();
        // Sanity checks on shape — schema_version present, key fields
        // round-tripped through serde.
        assert!(body.contains("\"schema_version\": 1"));
        assert!(body.contains("\"instrument\": \"EUR_USD\""));
        assert!(body.contains("\"exit_reason\": \"take_profit\""));
        assert!(body.contains("\"entry:signal\""));
        assert!(body.contains("\"exit:take_profit\""));
        // Tmp file should be gone (rename consumed it).
        assert!(!path.with_extension("json.tmp").exists());
        cleanup(&tmp);
    }

    #[test]
    fn snapshot_handles_missing_entry_in_buffer() {
        // entry_ts_ms not present in bars buffer (e.g. evicted) →
        // empty windows rather than failing.
        let tmp = tempfile_path("trade_snap_evicted");
        let bars: Vec<_> = (0..50)
            .map(|i| mk_bar(i * 10_000 + 1_000_000, 1.10))
            .collect();
        let path = write_trade_snapshot(
            &tmp,
            "run-test-2",
            "EUR_USD",
            &mk_open(),
            150_000,
            1.10,
            0.0,
            Reason::MaxHold,
            None,
            None,
            &bars,
        )
        .expect("snapshot");
        let body = fs::read_to_string(&path).unwrap();
        assert!(body.contains("\"pre_trade_bars\": []"));
        assert!(body.contains("\"in_trade_bars\": []"));
        cleanup(&tmp);
    }

    #[test]
    fn snapshot_path_layout() {
        let p = snapshot_path(Path::new("/tmp/x"), "myrun", 12345);
        assert_eq!(p, Path::new("/tmp/x/myrun/12345.json"));
    }

    fn tempfile_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rtk-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup(p: &Path) {
        let _ = fs::remove_dir_all(p);
    }
}

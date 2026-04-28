//! `pipeline-orchestrator` — Rust entry point for a full retrain run.
//!
//! Wraps the existing Python ML core (`python -m research pipeline
//! run`) with a Rust outer layer that owns:
//!
//!   * `pipeline_runs` row tracking via `observability::RunTracker`.
//!   * Post-flight JSONL artifact writes
//!     (`trade_logs/<v>/training_log.jsonl`,
//!      `deployment_gate.jsonl`, `champion_changes.jsonl`).
//!   * Reading back the `model_metrics`, `model_candidates`,
//!     `lockbox_results`, `model_deployment_gate` rows the Python
//!     core just wrote, so the JSONL preview reflects the same data
//!     the dashboard sees.
//!
//! The supervisor (`scripts/api-server-supervisor.sh`) calls this
//! binary instead of invoking Python directly. That gives us a single
//! Rust-owned entry point even while the heavy ML steps remain in
//! Python (LightGBM/XGBoost/CatBoost/sklearn/Optuna/skl2onnx — all
//! libs without first-class Rust equivalents).
//!
//! Usage:
//!
//! ```text
//! pipeline-orchestrator --instrument EUR_USD \
//!     [--side-trials 12] [--trader-trials 20] \
//!     [--publish-on-lockbox-fail]
//! ```
//!
//! All env vars consumed by the Python core (`MIN_OOS_AUC`,
//! `MAX_OOS_LOG_LOSS`, `REQUIRE_LOCKBOX_PASS`, etc.) propagate
//! through the spawned subprocess automatically.

use std::path::PathBuf;
use std::process::Stdio;

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

use observability::RunTracker;
use persistence::Db;

#[derive(Parser, Debug)]
#[command(name = "pipeline-orchestrator", about = "Rust entry for the retrain pipeline")]
struct Args {
    /// Instrument to retrain.
    #[arg(short, long)]
    instrument: String,
    /// Side-trainer Optuna trials. Forwarded to the Python core.
    #[arg(long, default_value_t = 12)]
    side_trials: u32,
    /// Trader optimiser Optuna trials. Forwarded to the Python core.
    #[arg(long, default_value_t = 20)]
    trader_trials: u32,
    /// Publish even if lockbox fails (development only).
    #[arg(long, default_value_t = false)]
    publish_on_lockbox_fail: bool,
    /// Override the database path. Defaults to `DATABASE_PATH` env or
    /// `./data/oanda.duckdb`.
    #[arg(long)]
    database_path: Option<PathBuf>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();

    let db_path = args
        .database_path
        .clone()
        .or_else(|| std::env::var("DATABASE_PATH").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("./data/oanda.duckdb"));
    let db = Db::open(&db_path).with_context(|| {
        format!(
            "pipeline-orchestrator: open db {}",
            db_path.display()
        )
    })?;

    // Start tracking.
    let args_json = serde_json::json!({
        "instrument": args.instrument,
        "side_trials": args.side_trials,
        "trader_trials": args.trader_trials,
        "publish_on_lockbox_fail": args.publish_on_lockbox_fail,
    });
    let tracker = RunTracker::start(&db, "pipeline.full", Some(&args.instrument), &args_json);
    tracing::info!(
        run_id = %tracker.run_id,
        instrument = %args.instrument,
        "pipeline-orchestrator: starting"
    );

    // Drop our DB handle BEFORE spawning Python so the file lock is
    // released. The Python core needs the lock to write
    // labels/oof/metrics/etc.
    drop(db);

    let py_outcome = run_python_core(&args);
    let py_ok = py_outcome.is_ok();
    let py_err = py_outcome.as_ref().err().map(|e| format!("{e:#}"));

    // Re-open DB for post-flight reads + JSONL writes.
    let db = Db::open(&db_path).with_context(|| {
        format!("pipeline-orchestrator: re-open db after python {}", db_path.display())
    })?;

    // Read back the rows Python just wrote and emit JSONL artifacts.
    let post = post_flight(&db, &args.instrument, &tracker.run_id, py_ok, py_err.as_deref());
    if let Err(e) = post {
        tracing::warn!(error = %e, "pipeline-orchestrator: post-flight JSONL write failed");
    }

    if py_ok {
        tracker.success(&db);
        Ok(())
    } else {
        tracker.fail(
            &db,
            py_err.as_deref().unwrap_or("unknown python failure"),
        );
        anyhow::bail!("python core exited non-zero");
    }
}

fn run_python_core(args: &Args) -> Result<()> {
    let venv_python = std::env::var("PIPELINE_PYTHON_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./.venv/bin/python"));
    let research_dir = std::env::var("PIPELINE_RESEARCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./research"));

    let mut cmd = std::process::Command::new(&venv_python);
    cmd.arg("-m").arg("research").arg("pipeline").arg("run")
        .arg("--instrument").arg(&args.instrument)
        .arg("--side-trials").arg(args.side_trials.to_string())
        .arg("--trader-trials").arg(args.trader_trials.to_string())
        .current_dir(&research_dir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .stdin(Stdio::null());
    if args.publish_on_lockbox_fail {
        cmd.arg("--publish-on-lockbox-fail");
    }

    tracing::info!(
        python = %venv_python.display(),
        cwd = %research_dir.display(),
        "pipeline-orchestrator: spawning python ML core"
    );
    let status = cmd
        .status()
        .with_context(|| format!("spawn {}", venv_python.display()))?;
    if !status.success() {
        anyhow::bail!("python pipeline exited with {status}");
    }
    Ok(())
}

#[derive(Serialize, Deserialize, Debug)]
struct ModelMetricsRow {
    model_id: String,
    instrument: String,
    ts_ms: i64,
    oos_auc: f64,
    oos_log_loss: f64,
    oos_brier: f64,
    oos_balanced_acc: f64,
    n_train: i64,
    n_oof: i64,
}

#[derive(Serialize, Deserialize, Debug)]
struct CandidateRow {
    spec: String,
    oos_log_loss: f64,
    oos_auc: f64,
    is_winner: bool,
}

fn post_flight(
    db: &Db,
    instrument: &str,
    parent_run_id: &str,
    py_ok: bool,
    py_err: Option<&str>,
) -> Result<()> {
    // Look up the latest model_metrics row for this instrument.
    let latest_metrics: Option<ModelMetricsRow> = db.with_conn(|conn| -> duckdb::Result<Option<ModelMetricsRow>> {
        let mut stmt = conn.prepare(
            "SELECT model_id, instrument, ts_ms, oos_auc, oos_log_loss, oos_brier,
                    oos_balanced_acc, n_train, n_oof
             FROM model_metrics
             WHERE instrument = ?
             ORDER BY ts_ms DESC LIMIT 1",
        )?;
        let mut iter = stmt.query_map(duckdb::params![instrument], |r| {
            Ok(ModelMetricsRow {
                model_id: r.get(0)?,
                instrument: r.get(1)?,
                ts_ms: r.get(2)?,
                oos_auc: r.get(3)?,
                oos_log_loss: r.get(4)?,
                oos_brier: r.get(5)?,
                oos_balanced_acc: r.get(6)?,
                n_train: r.get(7)?,
                n_oof: r.get(8)?,
            })
        })?;
        Ok(iter.next().transpose()?)
    })?;

    let Some(metrics) = latest_metrics else {
        if !py_ok {
            tracing::warn!(
                instrument,
                py_err = py_err.unwrap_or(""),
                "pipeline-orchestrator: python failed before metrics row landed"
            );
        }
        return Ok(()); // Nothing more to write.
    };

    // Read candidates for this run by joining on most-recent run_id
    // matching this instrument's ts_ms.
    let candidates: Vec<CandidateRow> = db.with_conn(|conn| -> duckdb::Result<Vec<CandidateRow>> {
        let mut stmt = conn.prepare(
            "SELECT spec_name, oos_log_loss, oos_auc, is_winner
             FROM model_candidates
             WHERE instrument = ?
             ORDER BY ts_ms DESC, oos_log_loss ASC
             LIMIT 7",
        )?;
        let rows = stmt
            .query_map(duckdb::params![instrument], |r| {
                Ok(CandidateRow {
                    spec: r.get(0)?,
                    oos_log_loss: r.get(1)?,
                    oos_auc: r.get(2)?,
                    is_winner: r.get(3)?,
                })
            })?
            .collect::<duckdb::Result<Vec<_>>>()?;
        Ok(rows)
    })?;

    // Read the latest gate decision for this model_id.
    let gate: Option<(bool, String, String, f64)> = db.with_conn(|conn| -> duckdb::Result<_> {
        let mut stmt = conn.prepare(
            "SELECT passed_gate, blocked_reasons, gate_thresholds_json, fine_tune_sortino
             FROM model_deployment_gate
             WHERE model_id = ?
             LIMIT 1",
        )?;
        let row: Option<(bool, String, String, f64)> = stmt
            .query_map(duckdb::params![&metrics.model_id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
            })?
            .next()
            .transpose()?;
        Ok(row)
    })?;

    let lockbox: Option<bool> = db.with_conn(|conn| -> duckdb::Result<_> {
        let mut stmt = conn.prepare(
            "SELECT summary_json
             FROM lockbox_results
             WHERE model_id = ?
             ORDER BY ts_ms DESC LIMIT 1",
        )?;
        let s: Option<String> = stmt
            .query_map(duckdb::params![&metrics.model_id], |r| r.get::<_, String>(0))?
            .next()
            .transpose()?;
        let parsed: Option<bool> = s.and_then(|json| {
            serde_json::from_str::<serde_json::Value>(&json)
                .ok()
                .and_then(|v| v.get("pass").and_then(|p| p.as_bool()))
        });
        Ok(parsed)
    })?;

    let trader_metrics: Option<(String, f64, f64)> = db.with_conn(|conn| -> duckdb::Result<_> {
        let mut stmt = conn.prepare(
            "SELECT params_id, fine_tune_sortino, max_dd_bp
             FROM trader_metrics
             WHERE model_id = ?
             ORDER BY ts_ms DESC LIMIT 1",
        )?;
        let row: Option<(String, f64, f64)> = stmt
            .query_map(duckdb::params![&metrics.model_id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })?
            .next()
            .transpose()?;
        Ok(row)
    })?;

    // training_log.jsonl
    let training_record = build_training_record(
        instrument,
        parent_run_id,
        &metrics,
        gate.as_ref(),
        lockbox,
        trader_metrics.as_ref(),
        &candidates,
        py_ok,
    );
    write_jsonl_to_repo("training_log.jsonl", &training_record)?;

    // deployment_gate.jsonl
    if let Some(gate_ref) = gate.as_ref() {
        let gate_record = build_gate_record(instrument, &metrics.model_id, gate_ref);
        write_jsonl_to_repo("deployment_gate.jsonl", &gate_record)?;
    }
    Ok(())
}

/// Build the `training_log.jsonl` record. Pure function — does not
/// touch the DB or filesystem; takes already-fetched rows in.
///
/// `gate` is the `(passed, blocked_reasons_joined, thresholds_json,
/// fine_tune_sortino)` tuple that `post_flight` reads from
/// `model_deployment_gate`. `trader_metrics` is `(params_id,
/// fine_tune_sortino, max_dd_bp)` from `trader_metrics`. Any of
/// these may be absent (`None`); the record fills in safe defaults.
///
/// Extracted from `post_flight` so the assembly logic is testable
/// without spawning Python or seeding the full DB.
fn build_training_record(
    instrument: &str,
    parent_run_id: &str,
    metrics: &ModelMetricsRow,
    gate: Option<&(bool, String, String, f64)>,
    lockbox_passed: Option<bool>,
    trader_metrics: Option<&(String, f64, f64)>,
    candidates: &[CandidateRow],
    py_ok: bool,
) -> serde_json::Value {
    let model_kind = metrics
        .model_id
        .split('_')
        .nth(1)
        .unwrap_or("unknown")
        .to_string();
    let blocked_reasons = gate.map(|g| g.1.clone()).unwrap_or_default();
    let passed_gate = gate.map(|g| g.0).unwrap_or(false);
    let fine_tune_sortino = gate.map(|g| g.3).unwrap_or(0.0);
    let (params_id, fine_tune_dd) = trader_metrics
        .map(|t| (t.0.clone(), t.2))
        .unwrap_or_default();

    serde_json::json!({
        "v": format!("v{}", env!("CARGO_PKG_VERSION")),
        "ts_ms": chrono::Utc::now().timestamp_millis(),
        "instrument": instrument,
        "run_id": parent_run_id,
        "model_id": metrics.model_id,
        "model_kind": model_kind,
        "n_features": 24,
        "n_train": metrics.n_train,
        "n_oof": metrics.n_oof,
        "oos_auc": metrics.oos_auc,
        "oos_log_loss": metrics.oos_log_loss,
        "oos_brier": metrics.oos_brier,
        "oos_balanced_acc": metrics.oos_balanced_acc,
        "fine_tune_sortino": fine_tune_sortino,
        "fine_tune_max_dd_bp": fine_tune_dd,
        "passed_gate": passed_gate,
        "blocked_reasons": blocked_reasons,
        "lockbox_passed": lockbox_passed,
        "candidates": candidates,
        "params_id": params_id,
        "py_ok": py_ok,
    })
}

/// Build the `deployment_gate.jsonl` record. Pure function. The
/// `blocked_reasons` string is split on `"; "` (the same separator
/// `deploy_gate::persist` joins with) and filtered to drop empty
/// fragments produced when there were no reasons.
fn build_gate_record(
    instrument: &str,
    model_id: &str,
    gate: &(bool, String, String, f64),
) -> serde_json::Value {
    let (passed, blocked, thresholds_json, _) = gate;
    let thresholds: serde_json::Value =
        serde_json::from_str(thresholds_json).unwrap_or(serde_json::Value::Null);
    serde_json::json!({
        "v": format!("v{}", env!("CARGO_PKG_VERSION")),
        "ts_ms": chrono::Utc::now().timestamp_millis(),
        "instrument": instrument,
        "model_id": model_id,
        "passed": passed,
        "blocked": blocked
            .split("; ")
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>(),
        "thresholds": thresholds,
    })
}

fn write_jsonl_to_repo(file: &str, record: &serde_json::Value) -> Result<()> {
    let root = std::env::var("TRADE_LOGS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Anchor at the workspace root via CARGO_MANIFEST_DIR.
            let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            crate_dir
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .map(|repo| repo.join("trade_logs"))
                .unwrap_or_else(|| PathBuf::from("./trade_logs"))
        });
    let version = env!("CARGO_PKG_VERSION");
    live_trader::jsonl_log::append(&root, version, file, record)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn sample_metrics(model_id: &str) -> ModelMetricsRow {
        ModelMetricsRow {
            model_id: model_id.to_string(),
            instrument: "EUR_USD".to_string(),
            ts_ms: 1_700_000_000_000,
            oos_auc: 0.61,
            oos_log_loss: 0.65,
            oos_brier: 0.21,
            oos_balanced_acc: 0.55,
            n_train: 800,
            n_oof: 200,
        }
    }

    fn sample_candidates() -> Vec<CandidateRow> {
        vec![
            CandidateRow {
                spec: "lgbm".to_string(),
                oos_log_loss: 0.65,
                oos_auc: 0.61,
                is_winner: true,
            },
            CandidateRow {
                spec: "xgb".to_string(),
                oos_log_loss: 0.66,
                oos_auc: 0.60,
                is_winner: false,
            },
        ]
    }

    // --- build_training_record -------------------------------------

    #[test]
    fn training_record_full_happy_path_carries_all_fields() {
        let metrics = sample_metrics("lgbm_main");
        let gate = (true, String::new(), "{\"min_oos_auc\":0.55}".to_string(), 0.42);
        let trader = ("p_xyz".to_string(), 0.42, 700.0);
        let candidates = sample_candidates();
        let rec = build_training_record(
            "EUR_USD",
            "run-1",
            &metrics,
            Some(&gate),
            Some(true),
            Some(&trader),
            &candidates,
            true,
        );
        assert_eq!(rec["instrument"], "EUR_USD");
        assert_eq!(rec["run_id"], "run-1");
        assert_eq!(rec["model_id"], "lgbm_main");
        assert_eq!(rec["model_kind"], "main"); // split on '_', nth(1)
        assert_eq!(rec["n_features"], 24);
        assert_eq!(rec["passed_gate"], true);
        assert_eq!(rec["blocked_reasons"], "");
        assert_eq!(rec["fine_tune_sortino"], 0.42);
        assert_eq!(rec["fine_tune_max_dd_bp"], 700.0);
        assert_eq!(rec["lockbox_passed"], true);
        assert_eq!(rec["params_id"], "p_xyz");
        assert_eq!(rec["py_ok"], true);
        assert_eq!(rec["candidates"].as_array().unwrap().len(), 2);
        assert_eq!(rec["candidates"][0]["is_winner"], true);
    }

    #[test]
    fn training_record_edge_missing_gate_and_trader_uses_safe_defaults() {
        // Edge: Python finished but gate / trader rows didn't land.
        // Defaults must NOT crash and must produce serializable JSON.
        let metrics = sample_metrics("xgb_zoo");
        let rec = build_training_record(
            "USD_JPY",
            "run-2",
            &metrics,
            None,
            None,
            None,
            &[],
            true,
        );
        assert_eq!(rec["passed_gate"], false);
        assert_eq!(rec["blocked_reasons"], "");
        assert_eq!(rec["fine_tune_sortino"], 0.0);
        assert_eq!(rec["fine_tune_max_dd_bp"], 0.0);
        assert_eq!(rec["params_id"], "");
        assert!(rec["lockbox_passed"].is_null());
        assert_eq!(rec["candidates"].as_array().unwrap().len(), 0);
        assert_eq!(rec["model_kind"], "zoo");
        // Sanity: must be serialisable JSONL line.
        let s = serde_json::to_string(&rec).expect("serialise");
        assert!(s.starts_with('{') && s.ends_with('}'));
    }

    #[test]
    fn training_record_error_path_python_failed_carries_blocked_reasons() {
        // Error path: Python core failed, gate row recorded the failures.
        let metrics = sample_metrics("lgbm_v2");
        let gate = (
            false,
            "oos_auc 0.40 < 0.55; lockbox did not pass".to_string(),
            "{}".to_string(),
            0.10,
        );
        let rec = build_training_record(
            "GBP_USD",
            "run-3",
            &metrics,
            Some(&gate),
            Some(false),
            None,
            &[],
            false,
        );
        assert_eq!(rec["passed_gate"], false);
        assert_eq!(rec["py_ok"], false);
        assert_eq!(rec["lockbox_passed"], false);
        let blocked = rec["blocked_reasons"].as_str().unwrap();
        assert!(blocked.contains("oos_auc"));
        assert!(blocked.contains("lockbox did not pass"));
    }

    #[test]
    fn training_record_handles_model_id_without_underscore() {
        // Edge: defensive fallback when model_id has no '_'.
        let mut m = sample_metrics("flatname");
        m.model_id = "flatname".to_string();
        let rec = build_training_record(
            "EUR_USD",
            "run-4",
            &m,
            None,
            None,
            None,
            &[],
            true,
        );
        assert_eq!(rec["model_kind"], "unknown");
    }

    // --- build_gate_record -----------------------------------------

    #[test]
    fn gate_record_splits_blocked_reasons_on_semicolon() {
        let gate = (
            false,
            "oos_auc 0.40 < 0.55; max_dd 1800 > 1500".to_string(),
            "{\"min_oos_auc\":0.55,\"max_oos_log_loss\":0.7,\"min_oos_balanced_acc\":0.52,\"min_fine_tune_sortino\":0.3,\"max_fine_tune_dd_bp\":1500.0,\"require_lockbox_pass\":true}".to_string(),
            0.0,
        );
        let rec = build_gate_record("EUR_USD", "lgbm_main", &gate);
        let blocked = rec["blocked"].as_array().unwrap();
        assert_eq!(blocked.len(), 2);
        assert_eq!(blocked[0], "oos_auc 0.40 < 0.55");
        assert_eq!(blocked[1], "max_dd 1800 > 1500");
        assert_eq!(rec["thresholds"]["min_oos_auc"], 0.55);
        assert_eq!(rec["passed"], false);
    }

    #[test]
    fn gate_record_empty_blocked_string_yields_empty_array() {
        let gate = (true, String::new(), "{}".to_string(), 0.42);
        let rec = build_gate_record("USD_JPY", "lgbm_main", &gate);
        let blocked = rec["blocked"].as_array().unwrap();
        assert!(blocked.is_empty(), "got {blocked:?}");
        assert_eq!(rec["passed"], true);
    }

    #[test]
    fn gate_record_malformed_thresholds_json_falls_back_to_null() {
        let gate = (
            false,
            "x".to_string(),
            "not json".to_string(),
            0.0,
        );
        let rec = build_gate_record("EUR_USD", "m1", &gate);
        assert!(rec["thresholds"].is_null());
        assert_eq!(rec["model_id"], "m1");
    }

    // --- CLI smoke -------------------------------------------------

    #[test]
    fn cli_help_renders() {
        // Smoke: clap can build the command; --help text mentions
        // the documented flags. Any future regression in the Args
        // struct surfaces here without spawning the binary.
        let mut cmd = Args::command();
        let help = cmd.render_long_help().to_string();
        assert!(help.contains("--instrument"));
        assert!(help.contains("--side-trials"));
        assert!(help.contains("--trader-trials"));
        assert!(help.contains("--publish-on-lockbox-fail"));
    }

    #[test]
    fn cli_parses_required_instrument_and_defaults() {
        let args = Args::try_parse_from([
            "pipeline-orchestrator",
            "--instrument",
            "EUR_USD",
        ])
        .expect("parse");
        assert_eq!(args.instrument, "EUR_USD");
        assert_eq!(args.side_trials, 12);
        assert_eq!(args.trader_trials, 20);
        assert!(!args.publish_on_lockbox_fail);
    }

    #[test]
    fn cli_rejects_missing_instrument() {
        let err = Args::try_parse_from(["pipeline-orchestrator"]).unwrap_err();
        // clap returns an error kind, not a panic.
        assert!(
            err.to_string().contains("--instrument")
                || err.kind() == clap::error::ErrorKind::MissingRequiredArgument,
            "unexpected error: {err}",
        );
    }
}

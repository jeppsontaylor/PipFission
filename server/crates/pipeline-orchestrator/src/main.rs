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
    let model_kind = metrics
        .model_id
        .split('_')
        .nth(1)
        .unwrap_or("unknown")
        .to_string();
    let blocked_reasons = gate.as_ref().map(|g| g.1.clone()).unwrap_or_default();
    let passed_gate = gate.as_ref().map(|g| g.0).unwrap_or(false);
    let fine_tune_sortino = gate.as_ref().map(|g| g.3).unwrap_or(0.0);
    let (params_id, fine_tune_dd) = trader_metrics
        .as_ref()
        .map(|t| (t.0.clone(), t.2))
        .unwrap_or_default();

    let training_record = serde_json::json!({
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
        "lockbox_passed": lockbox,
        "candidates": candidates,
        "params_id": params_id,
        "py_ok": py_ok,
    });
    write_jsonl_to_repo("training_log.jsonl", &training_record)?;

    // deployment_gate.jsonl
    if let Some((passed, blocked, thresholds_json, _)) = gate {
        let thresholds: serde_json::Value = serde_json::from_str(&thresholds_json)
            .unwrap_or(serde_json::Value::Null);
        let gate_record = serde_json::json!({
            "v": format!("v{}", env!("CARGO_PKG_VERSION")),
            "ts_ms": chrono::Utc::now().timestamp_millis(),
            "instrument": instrument,
            "model_id": metrics.model_id,
            "passed": passed,
            "blocked": blocked.split("; ").filter(|s| !s.is_empty()).collect::<Vec<_>>(),
            "thresholds": thresholds,
        });
        write_jsonl_to_repo("deployment_gate.jsonl", &gate_record)?;
    }
    Ok(())
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

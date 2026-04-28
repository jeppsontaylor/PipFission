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

use std::path::{Path, PathBuf};
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
    // released. The Python steps need the lock to write
    // labels/oof/metrics/etc.
    drop(db);

    // Run the staged pipeline. Each step is a separate Python
    // subprocess; the orchestrator threads model_id / params_id
    // between them. If any step fails, the rest are skipped and the
    // failure is recorded in the post-flight tracker.
    let py_outcome = run_staged_pipeline(&args, &db_path);
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

/// Cached paths to the venv python interpreter + research dir. Both
/// are env-overridable; defaults match the supervisor convention.
struct PyEnv {
    python: PathBuf,
    research_dir: PathBuf,
}

impl PyEnv {
    fn from_env() -> Self {
        // Resolve both paths to absolute so they work after the
        // subprocess sets `current_dir(research_dir)`. Relative paths
        // would otherwise be interpreted against the new cwd and the
        // python interpreter wouldn't be found.
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let python = std::env::var("PIPELINE_PYTHON_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./.venv/bin/python"));
        let python = if python.is_absolute() { python } else { cwd.join(&python) };
        let research_dir = std::env::var("PIPELINE_RESEARCH_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./research"));
        let research_dir = if research_dir.is_absolute() {
            research_dir
        } else {
            cwd.join(&research_dir)
        };
        Self { python, research_dir }
    }

    /// Build a `python -m research <subcmd...>` command rooted at
    /// `research_dir`. Stdout/stderr inherit so the supervisor log
    /// captures everything.
    fn cmd(&self, subargs: &[&str]) -> std::process::Command {
        let mut cmd = std::process::Command::new(&self.python);
        cmd.arg("-m").arg("research");
        for a in subargs {
            cmd.arg(a);
        }
        cmd.current_dir(&self.research_dir)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .stdin(Stdio::null());
        cmd
    }
}

/// Run the full pipeline as a sequence of focused Python subprocesses,
/// threading IDs (label_run_id → model_id → params_id) between them.
///
/// Steps:
/// 1. `label run` → emits `{"label_run_id": "...", "n_chosen": N}`
///    on stdout; the orchestrator reads the most recent
///    `labels.label_run_id` from DB instead of parsing stdout.
/// 2. `train side --json-out` → produces a `model_id` + per-candidate
///    metrics. Persists to model_metrics + model_candidates +
///    model_artifacts.
/// 3. `finetune run --json-out` → produces a `params_id`. Persists
///    to trader_metrics + optimizer_trials.
/// 4. `lockbox seal --json-out --fail-silently` → produces a sealed
///    `lockbox_results` row. Pass/fail is read from the JSON.
/// 5. `export champion --json-out` → produces ONNX + manifest.
///    Skipped when the lockbox failed AND `--publish-on-lockbox-fail`
///    is not set. The deployment-gate decision is computed in Rust
///    via the `deploy-gate` crate before deciding whether to export.
fn run_staged_pipeline(args: &Args, db_path: &Path) -> Result<()> {
    let py = PyEnv::from_env();
    let scratch = std::env::temp_dir().join(format!(
        "pipeline-orchestrator-{}-{}",
        args.instrument.replace('/', "_"),
        std::process::id(),
    ));
    std::fs::create_dir_all(&scratch).context("create scratch dir")?;

    // 1. Label — fully Rust (no python subprocess). Reads bars from
    //    persistence, runs `labeling::run_label_pipeline`, persists
    //    the chosen labels to DuckDB via `Db::insert_labels`. Closes
    //    the DB handle before the next step so the lock is free for
    //    the python ML steps.
    tracing::info!("step 1/5: label (Rust in-process)");
    let label_run_id = run_label_step(db_path, &args.instrument)
        .context("label step failed")?;
    tracing::info!(label_run_id = %label_run_id, "step 1/5: label done");

    // 2. Train side classifier.
    tracing::info!("step 2/5: train side classifier (zoo)");
    let train_json = scratch.join("train.json");
    let status = py
        .cmd(&[
            "train", "side",
            "--instrument", &args.instrument,
            "--label-run-id", &label_run_id,
            "--trials", &args.side_trials.to_string(),
            "--json-out", &train_json.to_string_lossy(),
        ])
        .status()
        .context("spawn train side")?;
    if !status.success() {
        anyhow::bail!("train.side step failed: {status}");
    }
    let train_json_text = std::fs::read_to_string(&train_json)
        .with_context(|| format!("read train.json at {}", train_json.display()))?;
    let train_payload: serde_json::Value = serde_json::from_str(&train_json_text)
        .context("parse train.json")?;
    let model_id = train_payload
        .get("model_id")
        .and_then(|v| v.as_str())
        .context("train.json missing model_id")?
        .to_string();
    tracing::info!(model_id = %model_id, "step 2/5: train done");

    // 3. Fine-tune trader.
    tracing::info!("step 3/5: trader fine-tune");
    let ft_json = scratch.join("finetune.json");
    let status = py
        .cmd(&[
            "finetune", "run",
            "--instrument", &args.instrument,
            "--model-id", &model_id,
            "--trials", &args.trader_trials.to_string(),
            "--json-out", &ft_json.to_string_lossy(),
        ])
        .status()
        .context("spawn finetune")?;
    if !status.success() {
        anyhow::bail!("finetune step failed: {status}");
    }
    let ft_json_text = std::fs::read_to_string(&ft_json)
        .with_context(|| format!("read finetune.json at {}", ft_json.display()))?;
    let ft_payload: serde_json::Value = serde_json::from_str(&ft_json_text)
        .context("parse finetune.json")?;
    let params_id = ft_payload
        .get("params_id")
        .and_then(|v| v.as_str())
        .context("finetune.json missing params_id")?
        .to_string();
    tracing::info!(params_id = %params_id, "step 3/5: finetune done");

    // 4. Export ONNX (no live publish yet). The Rust lockbox needs an
    //    on-disk ONNX to score the held-out slice via
    //    `inference::PredictorRegistry`. We write the ONNX + manifest
    //    to the model's per-id directory but DELIBERATELY do NOT
    //    promote to the live dir until lockbox + deploy-gate both
    //    pass. The lockbox + gate run in Rust below.
    tracing::info!("step 4/6: ONNX export (no-publish-live)");
    let exp_json = scratch.join("export.json");
    let status = py
        .cmd(&[
            "export", "champion",
            "--model-id", &model_id,
            "--json-out", &exp_json.to_string_lossy(),
            "--no-publish-live",
        ])
        .status()
        .context("spawn export")?;
    if !status.success() {
        anyhow::bail!("export step failed: {status}");
    }
    let exp_json_text = std::fs::read_to_string(&exp_json)
        .with_context(|| format!("read export.json at {}", exp_json.display()))?;
    let exp_payload: serde_json::Value = serde_json::from_str(&exp_json_text)
        .context("parse export.json")?;
    let onnx_path = exp_payload
        .get("onnx_path")
        .and_then(|v| v.as_str())
        .context("export.json missing onnx_path")?
        .to_string();
    tracing::info!(onnx_path = %onnx_path, "step 4/6: export done (per-id only)");

    // 5. Lockbox — fully Rust (no python subprocess). Loads the just-
    //    written ONNX via `inference::PredictorRegistry`, runs the
    //    deterministic Rust backtest on the sealed slice, persists
    //    `lockbox_results`. Replaces `python -m research lockbox seal`.
    tracing::info!("step 5/6: lockbox seal (Rust in-process)");
    let lb_run_id = format!("lockbox_{model_id}_{params_id}");
    let lockbox_passed = run_lockbox_step(
        db_path,
        &args.instrument,
        &model_id,
        &params_id,
        &onnx_path,
        &lb_run_id,
    )
    .context("lockbox step failed")?;
    tracing::info!(
        lockbox_passed,
        run_id = %lb_run_id,
        "step 5/6: lockbox done"
    );

    // 6a. Deployment gate (Rust). Re-opens the DB briefly to read
    //     model_metrics + the lockbox decision, runs the pure-logic
    //     gate, persists the decision to model_deployment_gate.
    let gate_passed = evaluate_and_persist_gate(
        db_path,
        &args.instrument,
        &model_id,
        lockbox_passed,
    )
    .context("deployment gate evaluation")?;

    let should_publish = (lockbox_passed || args.publish_on_lockbox_fail) && gate_passed;
    if !should_publish {
        tracing::info!(
            lockbox_passed,
            gate_passed,
            "step 6/6: publish-to-live skipped (lockbox+gate)"
        );
        let _ = std::fs::remove_dir_all(&scratch);
        return Ok(());
    }

    // 6b. Promote ONNX from per-id dir to the live serving dir. Pure
    //     file-copy in Rust (no python). The api-server's hot-swap
    //     watcher picks it up on its next inotify tick.
    tracing::info!("step 6/6: publish ONNX to live dir");
    publish_onnx_to_live(&onnx_path).context("publish to live dir")?;
    tracing::info!(model_id = %model_id, "step 6/6: live-publish done");

    let _ = std::fs::remove_dir_all(&scratch);
    Ok(())
}

/// Score the lockbox slice via `inference::PredictorRegistry`, run
/// the Rust lockbox crate, persist + return pass/fail. Replaces the
/// Python `lockbox seal` subprocess entirely.
fn run_lockbox_step(
    db_path: &Path,
    instrument: &str,
    model_id: &str,
    params_id: &str,
    onnx_path: &str,
    run_id: &str,
) -> Result<bool> {
    use std::sync::Arc;
    let db = Db::open(db_path).context("open db for lockbox")?;
    // Load TraderParams for this params_id from the trader_metrics row.
    let params_json: String = db.with_conn(|conn| -> duckdb::Result<String> {
        conn.query_row(
            "SELECT params_json FROM trader_metrics WHERE params_id = ? LIMIT 1",
            duckdb::params![params_id],
            |r| r.get::<_, String>(0),
        )
    })?;
    let params: trader::TraderParams =
        serde_json::from_str(&params_json).context("parse trader params_json")?;

    // Load the ONNX manifest the python export step just wrote and
    // construct a Predictor from it. `try_load_onnx` validates the
    // input/output shape against `expected_n_features` (24).
    let manifest_path = Path::new(onnx_path)
        .parent()
        .map(|p| p.join("manifest.json"))
        .context("derive manifest path from onnx_path")?;
    let registry =
        Arc::new(inference::PredictorRegistry::new(market_domain::FEATURE_DIM));
    registry
        .try_load_onnx(&manifest_path)
        .with_context(|| format!("try_load_onnx {}", manifest_path.display()))?;
    let predictor = registry.current();

    // Default lockbox config; thresholds are env-overridable via
    // LOCKBOX_* env vars at the lockbox crate boundary in the future.
    let cfg = lockbox::LockboxConfig {
        instrument: instrument.to_string(),
        model_id: model_id.to_string(),
        params_id: params_id.to_string(),
        ..Default::default()
    };
    let res = lockbox::seal_lockbox(&db, &cfg, &params, predictor.0.as_ref(), run_id)?;
    Ok(res.passed)
}

/// Copy `<model_dir>/champion.onnx` + `<model_dir>/manifest.json` to
/// the live serving dir. Atomic via tmp + rename so the api-server's
/// inotify watcher never sees a partial file.
fn publish_onnx_to_live(onnx_path: &str) -> Result<()> {
    let onnx = Path::new(onnx_path);
    let model_dir = onnx.parent().context("onnx_path has no parent")?;
    let manifest = model_dir.join("manifest.json");
    if !manifest.exists() {
        anyhow::bail!("manifest.json missing next to {onnx_path}");
    }

    // Anchor live dir at the workspace root via CARGO_MANIFEST_DIR.
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = crate_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .context("derive workspace root")?;
    let live_dir = workspace_root.join("research/artifacts/models/live");
    std::fs::create_dir_all(&live_dir).context("mkdir live dir")?;

    for (src, name) in [(onnx, "champion.onnx"), (manifest.as_path(), "manifest.json")] {
        let dst = live_dir.join(name);
        let tmp = live_dir.join(format!("{name}.tmp"));
        std::fs::copy(src, &tmp)
            .with_context(|| format!("copy {} → {}", src.display(), tmp.display()))?;
        std::fs::rename(&tmp, &dst)
            .with_context(|| format!("rename {} → {}", tmp.display(), dst.display()))?;
    }
    Ok(())
}

/// Run the label step entirely in-process: load bars, run the
/// labelling pipeline, persist the chosen labels. Returns the new
/// `label_run_id` for downstream steps.
fn run_label_step(db_path: &Path, instrument: &str) -> Result<String> {
    let db = Db::open(db_path).context("open db for label step")?;
    let bars = db
        .recent_bars_10s(instrument, 1000)
        .with_context(|| format!("load bars for {instrument}"))?;
    if bars.is_empty() {
        anyhow::bail!("no bars in DB for {instrument}");
    }
    let bars_typed: Vec<market_domain::Bar10s> = bars
        .into_iter()
        .map(|b| market_domain::Bar10s {
            instrument_id: 0,
            ts_ms: b.ts_ms,
            open: b.open,
            high: b.high,
            low: b.low,
            close: b.close,
            n_ticks: b.n_ticks,
            spread_bp_avg: b.spread_bp_avg,
        })
        .collect();
    let cfg = labeling::LabelPipelineConfig::default();
    let out = labeling::run_label_pipeline(&bars_typed, &cfg);
    let run_id = format!(
        "run_{}_{}",
        chrono::Utc::now().timestamp(),
        uuid::Uuid::new_v4().simple().to_string()[..6].to_string()
    );
    let n_persisted = db
        .insert_labels(instrument, &run_id, &out.labels)
        .context("persist labels")?;
    tracing::info!(
        n_bars = out.n_bars,
        n_events = out.n_events,
        n_chosen = out.labels.len(),
        n_persisted,
        run_id = %run_id,
        "labelling complete"
    );
    Ok(run_id)
}

/// Read the most recent `label_run_id` for `instrument` from the
/// `labels` table. Used to thread the id from the label step to the
/// train step without parsing python stdout.
#[allow(dead_code)]
fn read_latest_label_run_id(db_path: &Path, instrument: &str) -> Result<Option<String>> {
    let db = Db::open(db_path).context("open db for label_run_id lookup")?;
    let id: Option<String> = db.with_conn(|conn| -> duckdb::Result<Option<String>> {
        let mut stmt = conn.prepare(
            "SELECT label_run_id FROM labels WHERE instrument = ?
             ORDER BY ts_ms DESC LIMIT 1",
        )?;
        let r: Option<String> = stmt
            .query_map(duckdb::params![instrument], |row| row.get::<_, String>(0))?
            .next()
            .transpose()?;
        Ok(r)
    })?;
    Ok(id)
}

/// Read model_metrics + lockbox_results for `model_id`, run the
/// `deploy-gate` evaluation, persist the decision. Returns `passed_gate`.
fn evaluate_and_persist_gate(
    db_path: &Path,
    instrument: &str,
    model_id: &str,
    lockbox_passed: bool,
) -> Result<bool> {
    use deploy_gate::{evaluate, persist, DeploymentGateThresholds, GateInputs};

    let db = Db::open(db_path).context("open db for gate evaluation")?;
    let metrics: (f64, f64, f64, f64) = db
        .with_conn(|conn| -> duckdb::Result<(f64, f64, f64, f64)> {
            conn.query_row(
                "SELECT oos_auc, oos_log_loss, oos_brier, oos_balanced_acc
                 FROM model_metrics WHERE model_id = ? LIMIT 1",
                duckdb::params![model_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
        })
        .context("read model_metrics for gate")?;
    let trader: (f64, f64) = db
        .with_conn(|conn| -> duckdb::Result<(f64, f64)> {
            conn.query_row(
                "SELECT fine_tune_sortino, max_dd_bp
                 FROM trader_metrics WHERE model_id = ?
                 ORDER BY ts_ms DESC LIMIT 1",
                duckdb::params![model_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
        })
        .context("read trader_metrics for gate")?;
    let inputs = GateInputs {
        model_id: model_id.to_string(),
        instrument: instrument.to_string(),
        oos_auc: metrics.0,
        oos_log_loss: metrics.1,
        oos_brier: metrics.2,
        oos_balanced_acc: metrics.3,
        fine_tune_sortino: trader.0,
        fine_tune_max_dd_bp: trader.1,
        lockbox_passed,
        ts_ms: None,
    };
    let thresholds = DeploymentGateThresholds::from_env();
    let result = evaluate(&inputs, &thresholds);
    persist(&db, &result).context("persist gate result")?;
    tracing::info!(
        passed_gate = result.passed_gate,
        n_blocked = result.blocked_reasons.len(),
        "deploy-gate evaluated"
    );
    Ok(result.passed_gate)
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

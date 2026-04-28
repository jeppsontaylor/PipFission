//! Deployment quality gate.
//!
//! Sits between the model-zoo training step (`train.side`, picks the
//! lowest-OOS-log-loss winner) and the ONNX export step. Evaluates
//! the would-be champion against a set of performance floors. If ANY
//! floor fails, `passed_gate = false` and the orchestrator skips the
//! export — the prior champion stays live. The decision is persisted
//! to the `model_deployment_gate` table so the dashboard + post-mortem
//! analysts know exactly which bar a given model was held to.
//!
//! This is a pure-logic + DB-write Rust port of
//! `research/deployment/gate.py`. Same defaults, same env-var names
//! (`MIN_OOS_AUC`, `MAX_OOS_LOG_LOSS`, `MIN_OOS_BALANCED_ACC`,
//! `MIN_FINE_TUNE_SORTINO`, `MAX_FINE_TUNE_DD_BP`,
//! `REQUIRE_LOCKBOX_PASS`).

#![deny(unsafe_code)]

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use persistence::Db;

/// Default thresholds — conservative "no-skill defence":
///
/// * `min_oos_auc` 0.55      — beat coin-flip + tiny edge
/// * `max_oos_log_loss` 0.70 — calibrated probabilities
/// * `min_oos_balanced_acc` 0.52
/// * `min_fine_tune_sortino` 0.30 — return per unit downside on next 100 bars
/// * `max_fine_tune_dd_bp` 1500   — drawdown ceiling on fine-tune
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DeploymentGateThresholds {
    pub min_oos_auc: f64,
    pub max_oos_log_loss: f64,
    pub min_oos_balanced_acc: f64,
    pub min_fine_tune_sortino: f64,
    pub max_fine_tune_dd_bp: f64,
    pub require_lockbox_pass: bool,
}

impl Default for DeploymentGateThresholds {
    fn default() -> Self {
        Self {
            min_oos_auc: 0.55,
            max_oos_log_loss: 0.70,
            min_oos_balanced_acc: 0.52,
            min_fine_tune_sortino: 0.30,
            max_fine_tune_dd_bp: 1500.0,
            require_lockbox_pass: true,
        }
    }
}

impl DeploymentGateThresholds {
    /// Override any threshold via env. Same env-var names as the
    /// Python `from_env()` so the supervisor can set them once and
    /// both sides read the same values.
    pub fn from_env() -> Self {
        fn f64_env(name: &str, default: f64) -> f64 {
            std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        fn bool_env(name: &str, default: bool) -> bool {
            match std::env::var(name) {
                Err(_) => default,
                Ok(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
            }
        }
        let d = Self::default();
        Self {
            min_oos_auc: f64_env("MIN_OOS_AUC", d.min_oos_auc),
            max_oos_log_loss: f64_env("MAX_OOS_LOG_LOSS", d.max_oos_log_loss),
            min_oos_balanced_acc: f64_env("MIN_OOS_BALANCED_ACC", d.min_oos_balanced_acc),
            min_fine_tune_sortino: f64_env("MIN_FINE_TUNE_SORTINO", d.min_fine_tune_sortino),
            max_fine_tune_dd_bp: f64_env("MAX_FINE_TUNE_DD_BP", d.max_fine_tune_dd_bp),
            require_lockbox_pass: bool_env("REQUIRE_LOCKBOX_PASS", d.require_lockbox_pass),
        }
    }
}

/// Inputs to `evaluate`. Numeric fields are the OOS metrics the
/// trainer + fine-tuner reported for this model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateInputs {
    pub model_id: String,
    pub instrument: String,
    pub oos_auc: f64,
    pub oos_log_loss: f64,
    pub oos_brier: f64,
    pub oos_balanced_acc: f64,
    pub fine_tune_sortino: f64,
    pub fine_tune_max_dd_bp: f64,
    pub lockbox_passed: bool,
    /// Override the timestamp written to the DB row. None = use now.
    pub ts_ms: Option<i64>,
}

/// Outcome of the deployment gate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateResult {
    pub model_id: String,
    pub instrument: String,
    pub ts_ms: i64,
    pub passed_gate: bool,
    pub blocked_reasons: Vec<String>,
    pub oos_auc: f64,
    pub oos_log_loss: f64,
    pub oos_brier: f64,
    pub oos_balanced_acc: f64,
    pub fine_tune_sortino: f64,
    pub fine_tune_max_dd_bp: f64,
    pub thresholds: DeploymentGateThresholds,
}

/// Pure evaluation. Doesn't touch the DB. Callers should `persist`
/// the result.
pub fn evaluate(inputs: &GateInputs, thresholds: &DeploymentGateThresholds) -> GateResult {
    let mut reasons: Vec<String> = Vec::new();
    if inputs.oos_auc < thresholds.min_oos_auc {
        reasons.push(format!(
            "oos_auc {:.3} < {:.3}",
            inputs.oos_auc, thresholds.min_oos_auc
        ));
    }
    if inputs.oos_log_loss > thresholds.max_oos_log_loss {
        reasons.push(format!(
            "oos_log_loss {:.3} > {:.3}",
            inputs.oos_log_loss, thresholds.max_oos_log_loss
        ));
    }
    if inputs.oos_balanced_acc < thresholds.min_oos_balanced_acc {
        reasons.push(format!(
            "oos_balanced_acc {:.3} < {:.3}",
            inputs.oos_balanced_acc, thresholds.min_oos_balanced_acc
        ));
    }
    if inputs.fine_tune_sortino < thresholds.min_fine_tune_sortino {
        reasons.push(format!(
            "fine_tune_sortino {:.3} < {:.3}",
            inputs.fine_tune_sortino, thresholds.min_fine_tune_sortino
        ));
    }
    if inputs.fine_tune_max_dd_bp > thresholds.max_fine_tune_dd_bp {
        reasons.push(format!(
            "fine_tune_max_dd_bp {:.1} > {:.1}",
            inputs.fine_tune_max_dd_bp, thresholds.max_fine_tune_dd_bp
        ));
    }
    if thresholds.require_lockbox_pass && !inputs.lockbox_passed {
        reasons.push("lockbox did not pass".to_string());
    }
    let ts_ms = inputs
        .ts_ms
        .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
    GateResult {
        model_id: inputs.model_id.clone(),
        instrument: inputs.instrument.clone(),
        ts_ms,
        passed_gate: reasons.is_empty(),
        blocked_reasons: reasons,
        oos_auc: inputs.oos_auc,
        oos_log_loss: inputs.oos_log_loss,
        oos_brier: inputs.oos_brier,
        oos_balanced_acc: inputs.oos_balanced_acc,
        fine_tune_sortino: inputs.fine_tune_sortino,
        fine_tune_max_dd_bp: inputs.fine_tune_max_dd_bp,
        thresholds: *thresholds,
    }
}

/// Persist a `GateResult` to `model_deployment_gate`. Idempotent —
/// DELETEs any prior row for the same `model_id` first.
pub fn persist(db: &Db, result: &GateResult) -> Result<()> {
    let thresholds_json = serde_json::to_string(&result.thresholds)
        .context("serialize gate thresholds")?;
    let blocked = result.blocked_reasons.join("; ");
    db.with_conn(|conn| -> duckdb::Result<()> {
        conn.execute(
            "DELETE FROM model_deployment_gate WHERE model_id = ?",
            duckdb::params![result.model_id],
        )?;
        conn.execute(
            "INSERT INTO model_deployment_gate
                 (model_id, instrument, ts_ms, oos_auc, oos_log_loss,
                  oos_brier, oos_balanced_acc, fine_tune_sortino,
                  fine_tune_max_dd_bp, passed_gate, blocked_reasons,
                  gate_thresholds_json)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            duckdb::params![
                result.model_id,
                result.instrument,
                result.ts_ms,
                result.oos_auc,
                result.oos_log_loss,
                result.oos_brier,
                result.oos_balanced_acc,
                result.fine_tune_sortino,
                result.fine_tune_max_dd_bp,
                result.passed_gate,
                blocked,
                thresholds_json,
            ],
        )?;
        Ok(())
    })
    .context("persist gate result")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass_inputs() -> GateInputs {
        GateInputs {
            model_id: "m_passing".to_string(),
            instrument: "EUR_USD".to_string(),
            oos_auc: 0.60,
            oos_log_loss: 0.65,
            oos_brier: 0.20,
            oos_balanced_acc: 0.55,
            fine_tune_sortino: 0.50,
            fine_tune_max_dd_bp: 800.0,
            lockbox_passed: true,
            ts_ms: Some(1_700_000_000_000),
        }
    }

    #[test]
    fn passes_with_default_thresholds_when_all_floors_met() {
        let r = evaluate(&pass_inputs(), &DeploymentGateThresholds::default());
        assert!(r.passed_gate, "expected gate pass, blocked: {:?}", r.blocked_reasons);
        assert!(r.blocked_reasons.is_empty());
    }

    #[test]
    fn blocks_each_floor_independently() {
        let mut bad = pass_inputs();
        bad.oos_auc = 0.50;
        let r = evaluate(&bad, &DeploymentGateThresholds::default());
        assert!(!r.passed_gate);
        assert!(r.blocked_reasons.iter().any(|s| s.starts_with("oos_auc")));

        let mut bad = pass_inputs();
        bad.oos_log_loss = 0.85;
        let r = evaluate(&bad, &DeploymentGateThresholds::default());
        assert!(r.blocked_reasons.iter().any(|s| s.starts_with("oos_log_loss")));

        let mut bad = pass_inputs();
        bad.oos_balanced_acc = 0.48;
        let r = evaluate(&bad, &DeploymentGateThresholds::default());
        assert!(r.blocked_reasons.iter().any(|s| s.starts_with("oos_balanced_acc")));

        let mut bad = pass_inputs();
        bad.fine_tune_sortino = 0.10;
        let r = evaluate(&bad, &DeploymentGateThresholds::default());
        assert!(r.blocked_reasons.iter().any(|s| s.starts_with("fine_tune_sortino")));

        let mut bad = pass_inputs();
        bad.fine_tune_max_dd_bp = 2000.0;
        let r = evaluate(&bad, &DeploymentGateThresholds::default());
        assert!(r.blocked_reasons.iter().any(|s| s.starts_with("fine_tune_max_dd_bp")));

        let mut bad = pass_inputs();
        bad.lockbox_passed = false;
        let r = evaluate(&bad, &DeploymentGateThresholds::default());
        assert!(r.blocked_reasons.iter().any(|s| s == "lockbox did not pass"));
    }

    #[test]
    fn lockbox_failure_ignored_when_require_lockbox_pass_off() {
        let mut bad = pass_inputs();
        bad.lockbox_passed = false;
        let mut th = DeploymentGateThresholds::default();
        th.require_lockbox_pass = false;
        let r = evaluate(&bad, &th);
        assert!(r.passed_gate, "blocked: {:?}", r.blocked_reasons);
    }
}

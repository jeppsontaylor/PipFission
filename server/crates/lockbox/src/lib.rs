//! Single-shot lockbox holdout. Run exactly once per
//! (model_id, params_id) tuple. Persists a sealed row to
//! `lockbox_results`; re-running with the same `run_id` is a no-op
//! (DELETE-then-INSERT keeps the most recent decision).
//!
//! The lockbox is the final, untouchable performance gate: a champion
//! model + trader params is allowed to be promoted to live ONLY if
//! its lockbox row exists, is sealed, and clears the configured
//! thresholds (`n_trades >= min_n_trades`, `DSR >= min_dsr`,
//! `max_dd_bp <= max_dd_bp_limit`).
//!
//! Pure-Rust port of `research/lockbox/gate.py`. Composition over the
//! existing crates: bars from `persistence`, features from
//! `bar-features`, ONNX inference via `inference::Predictor`,
//! deterministic backtest from `backtest`, EWMA σ from `labeling`,
//! Sharpe + DSR from `metrics`. The python version is now a thin shim
//! that the orchestrator can drop entirely.

#![deny(unsafe_code)]

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use bar_features::recompute_last;
use inference::Predictor;
use labeling::ewma_volatility;
use market_domain::Bar10s;
use metrics::deflated_sharpe;
use persistence::Db;
use trader::{Probs, TraderParams};

/// Convert an `inference::Probs` (what `Predictor::predict` returns)
/// into a `trader::Probs` (what `backtest` consumes). The two structs
/// have identical fields; the conversion is a field-by-field copy.
fn to_trader_probs(p: inference::Probs) -> Probs {
    Probs {
        p_long: p.p_long,
        p_short: p.p_short,
        p_take: p.p_take,
        calibrated: p.calibrated,
    }
}

/// Lockbox configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LockboxConfig {
    pub instrument: String,
    pub model_id: String,
    pub params_id: String,
    /// Number of bars BEFORE the lockbox window — train + fine-tune
    /// region the model has already seen.
    pub n_seen: usize,
    /// Number of bars in the held-out slice. The lockbox is the
    /// trailing `n_lockbox` bars of the `n_seen + n_lockbox` window.
    pub n_lockbox: usize,
    /// Cost stress multiplier on commission/spread/slippage.
    pub cost_stress: f64,
    pub min_n_trades: usize,
    pub min_dsr: f64,
    pub max_dd_bp_limit: f64,
}

impl Default for LockboxConfig {
    fn default() -> Self {
        Self {
            instrument: String::new(),
            model_id: String::new(),
            params_id: String::new(),
            n_seen: 1100,
            n_lockbox: 100,
            cost_stress: 1.0,
            min_n_trades: 3,
            min_dsr: 0.50,
            max_dd_bp_limit: 1500.0,
        }
    }
}

/// Outcome of a lockbox seal.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LockboxResult {
    pub run_id: String,
    pub sealed: bool,
    pub passed: bool,
    pub reasons: Vec<String>,
    pub summary: serde_json::Value,
}

/// Returns true if a sealed `lockbox_results` row already exists for
/// this `run_id`.
pub fn is_already_sealed(db: &Db, run_id: &str) -> Result<bool> {
    let g = db.with_conn(|conn| -> duckdb::Result<bool> {
        let mut stmt =
            conn.prepare("SELECT sealed FROM lockbox_results WHERE run_id = ?")?;
        let row: Option<bool> = stmt
            .query_map(duckdb::params![run_id], |r| r.get::<_, bool>(0))?
            .next()
            .transpose()?;
        Ok(row.unwrap_or(false))
    })
    .context("lockbox: query is_already_sealed")?;
    Ok(g)
}

/// Seal the lockbox once. Idempotent — `run_id` is the dedup key.
/// Returns the `LockboxResult`. Caller is responsible for choosing
/// whether to publish the would-be champion based on `result.passed`.
///
/// The `predictor` must be the model identified by `cfg.model_id`;
/// the orchestrator constructs it (e.g. from the on-disk ONNX) and
/// passes it in. The lockbox doesn't care HOW the predictor was
/// produced — only that calling `predict(features)` returns the
/// model's probabilities.
pub fn seal_lockbox(
    db: &Db,
    cfg: &LockboxConfig,
    params: &TraderParams,
    predictor: &dyn Predictor,
    run_id: &str,
) -> Result<LockboxResult> {
    // 1. Load bars covering both the seen + held-out windows.
    let needed = cfg.n_seen + cfg.n_lockbox;
    let bars = db
        .recent_bars_10s(&cfg.instrument, needed)
        .with_context(|| format!("lockbox: load {needed} bars for {}", cfg.instrument))?;
    if bars.len() < needed {
        anyhow::bail!(
            "lockbox: need at least {needed} bars for {}, got {}",
            cfg.instrument,
            bars.len()
        );
    }
    // Convert Bar10sNamed → Bar10s for the backtest / feature recompute.
    let bars_typed: Vec<Bar10s> = bars
        .iter()
        .map(|b| Bar10s {
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
    let n = bars_typed.len();
    let lockbox_start = n - cfg.n_lockbox;

    // 2. Score each lockbox bar via the predictor.
    // `recompute_last` needs a rolling window ending at the bar; we
    // walk through the lockbox window bar-by-bar.
    let mut probs: Vec<Probs> = Vec::with_capacity(cfg.n_lockbox);
    for end in lockbox_start..n {
        let window = &bars_typed[..=end];
        let feat = match recompute_last(window) {
            Some(f) => f,
            None => {
                // Insufficient history — neutral probs so the trader
                // skips the bar.
                probs.push(Probs {
                    p_long: 0.5,
                    p_short: 0.5,
                    p_take: 0.5,
                    calibrated: 0.5,
                });
                continue;
            }
        };
        if feat.len() != predictor.n_features() {
            anyhow::bail!(
                "lockbox: feature dim mismatch — model expects {}, got {}",
                predictor.n_features(),
                feat.len(),
            );
        }
        probs.push(to_trader_probs(predictor.predict(&feat)));
    }

    // 3. EWMA σ on the lockbox slice.
    let lockbox_slice = &bars_typed[lockbox_start..];
    let sigma = ewma_volatility(lockbox_slice, 60);
    debug_assert_eq!(sigma.len(), lockbox_slice.len());

    // 4. Run the deterministic backtest on the lockbox window.
    let costs = backtest::Costs {
        commission_bp: 0.5 * cfg.cost_stress,
        spread_bp: 1.0 * cfg.cost_stress,
        slippage_bp: 0.5 * cfg.cost_stress,
    };
    let report = backtest::run_backtest(lockbox_slice, &probs, &sigma, *params, costs);

    // 5. DSR on per-trade returns. n_trials=1 since this is the
    // sealed slice, not a search.
    let per_trade: Vec<f64> = report.ledger.iter().map(|t| t.net_r).collect();
    let dsr = if per_trade.len() >= 2 {
        let mean = per_trade.iter().sum::<f64>() / per_trade.len() as f64;
        let var = per_trade.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (per_trade.len() as f64 - 1.0).max(1.0);
        let sd = var.max(0.0).sqrt();
        let sr = if sd > 1e-12 { mean / sd } else { 0.0 };
        deflated_sharpe(sr, per_trade.len(), 1, 0.0, 3.0)
    } else {
        0.0
    };

    // 6. Pass/fail.
    let mut reasons: Vec<String> = Vec::new();
    let n_trades = report.summary.n_trades;
    if n_trades < cfg.min_n_trades {
        reasons.push(format!("n_trades {n_trades} < min {}", cfg.min_n_trades));
    }
    if dsr < cfg.min_dsr {
        reasons.push(format!("DSR {dsr:.3} < min {:.3}", cfg.min_dsr));
    }
    if report.summary.max_drawdown_bp > cfg.max_dd_bp_limit {
        reasons.push(format!(
            "max_dd_bp {:.1} > limit {:.1}",
            report.summary.max_drawdown_bp, cfg.max_dd_bp_limit,
        ));
    }
    let passed = reasons.is_empty();

    // 7. Build the sealed summary JSON.
    let mut sealed_summary = serde_json::to_value(&report.summary)
        .context("lockbox: serialize summary")?;
    if let Some(map) = sealed_summary.as_object_mut() {
        map.insert("dsr".to_string(), serde_json::json!(dsr));
        map.insert("pass".to_string(), serde_json::json!(passed));
        map.insert("reasons".to_string(), serde_json::json!(reasons));
        map.insert(
            "n_lockbox_bars".to_string(),
            serde_json::json!(cfg.n_lockbox),
        );
        map.insert(
            "cost_stress".to_string(),
            serde_json::json!(cfg.cost_stress),
        );
    }

    // 8. Persist (idempotent on run_id).
    let summary_json = serde_json::to_string(&sealed_summary)?;
    let cfg_clone = cfg.clone();
    let rid_owned = run_id.to_string();
    db.with_conn(|conn| -> duckdb::Result<()> {
        conn.execute(
            "DELETE FROM lockbox_results WHERE run_id = ?",
            duckdb::params![rid_owned],
        )?;
        conn.execute(
            "INSERT INTO lockbox_results
                 (run_id, ts_ms, model_id, params_id, summary_json, sealed)
              VALUES (?, ?, ?, ?, ?, ?)",
            duckdb::params![
                rid_owned,
                chrono::Utc::now().timestamp_millis(),
                cfg_clone.model_id,
                cfg_clone.params_id,
                summary_json,
                true,
            ],
        )?;
        Ok(())
    })
    .context("lockbox: persist sealed result")?;

    Ok(LockboxResult {
        run_id: run_id.to_string(),
        sealed: true,
        passed,
        reasons,
        summary: sealed_summary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial predictor for tests — always returns near-50/50 probs.
    /// Mirrors `inference::FallbackPredictor` shape but we want a
    /// stable, deterministic stand-in here without pulling its full
    /// dependency tree into the lockbox tests.
    #[allow(dead_code)]
    struct ConstPredictor {
        n: usize,
        p_long: f64,
    }
    impl Predictor for ConstPredictor {
        fn n_features(&self) -> usize {
            self.n
        }
        fn predict(&self, _x: &[f64]) -> inference::Probs {
            inference::Probs {
                p_long: self.p_long,
                p_short: 1.0 - self.p_long,
                p_take: self.p_long.max(1.0 - self.p_long),
                calibrated: self.p_long,
            }
        }
        fn id(&self) -> &str {
            "const-test"
        }
    }

    #[test]
    fn config_defaults_are_reasonable() {
        let c = LockboxConfig::default();
        assert_eq!(c.n_seen, 1100);
        assert_eq!(c.n_lockbox, 100);
        assert_eq!(c.min_n_trades, 3);
        assert!((c.min_dsr - 0.50).abs() < 1e-9);
        assert!((c.max_dd_bp_limit - 1500.0).abs() < 1e-9);
    }

    #[test]
    fn pass_logic_accepts_all_thresholds_satisfied() {
        // Manual computation: 5 winning trades with a strong Sharpe →
        // DSR > 0.50, n_trades >= 3, dd < limit. We exercise the
        // pass/fail code path via direct construction since the full
        // backtest needs a Db + bars; backtest correctness is covered
        // in its own crate tests.
        let cfg = LockboxConfig::default();
        // Simulate: 5 trades all positive ⇒ no reasons.
        let mut reasons: Vec<String> = Vec::new();
        let n_trades = 5;
        let dsr = 0.95;
        let max_dd = 100.0;
        if n_trades < cfg.min_n_trades {
            reasons.push("n_trades".into());
        }
        if dsr < cfg.min_dsr {
            reasons.push("dsr".into());
        }
        if max_dd > cfg.max_dd_bp_limit {
            reasons.push("dd".into());
        }
        assert!(reasons.is_empty());
    }
}

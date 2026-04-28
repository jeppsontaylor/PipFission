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
/// this `run_id`. Use this before [`seal_lockbox`] when you want to
/// short-circuit re-seal attempts; `seal_lockbox` itself is also
/// idempotent on `run_id`, so the check is purely an optimisation.
///
/// # Errors
///
/// Propagates DuckDB query errors. Returns `Ok(false)` when the row
/// does not exist.
///
/// # Example
///
/// ```no_run
/// use lockbox::is_already_sealed;
/// use persistence::Db;
///
/// # fn run(db: &Db) -> anyhow::Result<()> {
/// if is_already_sealed(db, "run-2026-04-28-EUR_USD")? {
///     println!("already sealed; skipping");
/// }
/// # Ok(()) }
/// ```
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
///
/// Pipeline:
///
/// 1. Load `cfg.n_seen + cfg.n_lockbox` bars from `bars_10s`.
/// 2. Score each bar in the lockbox tail via `predictor.predict`.
/// 3. Run the deterministic backtest with stress-multiplied costs.
/// 4. Compute Deflated Sharpe from per-trade returns.
/// 5. Compare against `min_n_trades`, `min_dsr`, `max_dd_bp_limit`.
/// 6. Persist (`DELETE WHERE run_id = ? ; INSERT`) — sealed=true.
///
/// # Errors
///
/// * Insufficient bars in the DB for `cfg.instrument`.
/// * Feature dim returned by `bar_features::recompute_last` does not
///   match `predictor.n_features()`.
/// * DuckDB read or write failure.
///
/// # Example
///
/// ```no_run
/// use lockbox::{seal_lockbox, LockboxConfig};
/// use persistence::Db;
/// use trader::TraderParams;
/// use inference::Predictor;
///
/// # fn run(db: &Db, predictor: &dyn Predictor) -> anyhow::Result<()> {
/// let cfg = LockboxConfig {
///     instrument: "EUR_USD".to_string(),
///     model_id: "lgbm_v3".to_string(),
///     params_id: "p_42".to_string(),
///     ..Default::default()
/// };
/// let result = seal_lockbox(
///     db, &cfg, &TraderParams::default(), predictor,
///     "run-2026-04-28-EUR_USD",
/// )?;
/// if !result.passed {
///     eprintln!("blocked: {:?}", result.reasons);
/// }
/// # Ok(()) }
/// ```
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
    use bar_features::N_FEATURES;
    use tempfile::tempdir;

    /// Trivial predictor for tests — always returns near-50/50 probs.
    /// Mirrors `inference::FallbackPredictor` shape but we want a
    /// stable, deterministic stand-in here without pulling its full
    /// dependency tree into the lockbox tests.
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

    /// Seed `n` synthetic 10s bars for `instrument` directly into
    /// `bars_10s`. Prices follow a deterministic mild random walk so
    /// EWMA σ + features come out finite. Timestamps are 10s apart
    /// ending at `base_ts_ms`.
    fn seed_bars(db: &Db, instrument: &str, n: usize, base_ts_ms: i64) {
        db.with_conn(|conn| {
            let tx = conn.unchecked_transaction().unwrap();
            for i in 0..n {
                let ts = base_ts_ms - ((n - 1 - i) as i64) * 10_000;
                // Deterministic wiggle around 1.10 so log-returns are nonzero.
                let phase = (i as f64) * 0.07;
                let close = 1.10 + 0.001 * phase.sin();
                let open = close - 0.0001;
                let high = close + 0.0002;
                let low = open - 0.0002;
                tx.execute(
                    "INSERT INTO bars_10s
                     (instrument, ts_ms, open, high, low, close, n_ticks, spread_bp_avg)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    duckdb::params![
                        instrument,
                        ts,
                        open,
                        high,
                        low,
                        close,
                        12_i64,
                        1.0_f64,
                    ],
                )
                .unwrap();
            }
            tx.commit().unwrap();
        });
    }

    fn tmp_db(label: &str) -> (Db, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join(format!("{label}.duckdb"));
        let db = Db::open(&path).expect("open db");
        (db, dir)
    }

    fn small_cfg(model_id: &str) -> LockboxConfig {
        LockboxConfig {
            instrument: "EUR_USD".to_string(),
            model_id: model_id.to_string(),
            params_id: "p_test".to_string(),
            n_seen: 30,
            n_lockbox: 20,
            cost_stress: 1.0,
            min_n_trades: 3,
            min_dsr: 0.50,
            max_dd_bp_limit: 1500.0,
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
    fn errors_when_db_has_insufficient_bars() {
        // Error path: ask for 50 bars but only seed 10.
        let (db, _t) = tmp_db("lb_short");
        seed_bars(&db, "EUR_USD", 10, 1_700_000_000_000);
        let cfg = small_cfg("m_short"); // needs 50 bars
        let pred = ConstPredictor { n: N_FEATURES, p_long: 0.5 };
        let err = seal_lockbox(
            &db,
            &cfg,
            &TraderParams::default(),
            &pred,
            "run-short",
        )
        .expect_err("should fail when too few bars");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("need at least"),
            "expected 'need at least' in error, got: {msg}"
        );
    }

    #[test]
    fn fails_lockbox_with_neutral_predictor_and_writes_sealed_row() {
        // Neutral 50/50 predictor produces no entries → n_trades=0 →
        // blocked on min_n_trades. Result must still be sealed=true
        // and the row must land in `lockbox_results`.
        let (db, _t) = tmp_db("lb_neutral");
        let cfg = small_cfg("m_neutral");
        seed_bars(
            &db,
            "EUR_USD",
            cfg.n_seen + cfg.n_lockbox + 5,
            1_700_000_000_000,
        );
        let pred = ConstPredictor { n: N_FEATURES, p_long: 0.5 };
        let res = seal_lockbox(
            &db,
            &cfg,
            &TraderParams::default(),
            &pred,
            "run-neutral",
        )
        .expect("seal");
        assert!(res.sealed);
        assert!(!res.passed, "neutral predictor should fail min_n_trades");
        assert!(
            res.reasons.iter().any(|r| r.starts_with("n_trades ")),
            "expected n_trades reason, got {:?}",
            res.reasons,
        );
        // Row landed.
        let (n, sealed): (i64, bool) = db.with_conn(|c| {
            c.query_row(
                "SELECT COUNT(*), MIN(sealed::INT)::BOOL
                 FROM lockbox_results WHERE run_id = ?",
                duckdb::params!["run-neutral"],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap()
        });
        assert_eq!(n, 1);
        assert!(sealed);
    }

    #[test]
    fn idempotent_on_run_id_overwrites_prior_row() {
        // Happy-ish path: re-seal with the same run_id. Second call
        // must DELETE-then-INSERT, leaving exactly one row.
        let (db, _t) = tmp_db("lb_idem");
        let cfg = small_cfg("m_idem");
        seed_bars(
            &db,
            "EUR_USD",
            cfg.n_seen + cfg.n_lockbox + 5,
            1_700_000_000_000,
        );
        let pred = ConstPredictor { n: N_FEATURES, p_long: 0.5 };
        let r1 = seal_lockbox(
            &db,
            &cfg,
            &TraderParams::default(),
            &pred,
            "run-idem",
        )
        .expect("seal-1");
        assert!(r1.sealed);

        // is_already_sealed reflects the row.
        assert!(is_already_sealed(&db, "run-idem").expect("already?"));
        assert!(!is_already_sealed(&db, "run-missing").expect("missing?"));

        // Second seal with same run_id should overwrite, not duplicate.
        let r2 = seal_lockbox(
            &db,
            &cfg,
            &TraderParams::default(),
            &pred,
            "run-idem",
        )
        .expect("seal-2");
        assert!(r2.sealed);
        let n: i64 = db.with_conn(|c| {
            c.query_row(
                "SELECT COUNT(*) FROM lockbox_results WHERE run_id = ?",
                duckdb::params!["run-idem"],
                |r| r.get(0),
            )
            .unwrap()
        });
        assert_eq!(n, 1, "DELETE-then-INSERT must keep exactly one row");
    }

    #[test]
    fn errors_on_predictor_feature_dim_mismatch() {
        // Error path: predictor declares fewer features than recompute_last
        // returns → the dim-check should bail immediately.
        let (db, _t) = tmp_db("lb_dim");
        let cfg = small_cfg("m_dim");
        seed_bars(
            &db,
            "EUR_USD",
            cfg.n_seen + cfg.n_lockbox + 5,
            1_700_000_000_000,
        );
        let bad_pred = ConstPredictor { n: 7, p_long: 0.5 }; // not 24
        let err = seal_lockbox(
            &db,
            &cfg,
            &TraderParams::default(),
            &bad_pred,
            "run-dim",
        )
        .expect_err("dim mismatch must fail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("feature dim mismatch"),
            "unexpected error: {msg}"
        );
    }
}

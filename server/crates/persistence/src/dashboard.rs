//! Read-side helpers powering the new dashboard REST endpoints. Each
//! function maps directly to one route in `api-server/src/http.rs`:
//!
//!   GET /api/optimizer/trials      → `recent_optimizer_trials`
//!   GET /api/lockbox/result        → `latest_lockbox_result`
//!   GET /api/model/metrics         → `latest_model_metrics`
//!   GET /api/trader/metrics        → `latest_trader_metrics`
//!   GET /api/labels/recent         → `recent_labels`
//!   GET /api/champion/signals      → `recent_champion_signals`
//!
//! All queries are read-only (`open_ro` would also work but the live
//! engine uses the writer mutex for everything; sharing is fine).

use anyhow::Result;
use duckdb::params;
use serde::Serialize;

use crate::connection::Db;
use crate::MAX_HISTORY_LIMIT;

#[derive(Clone, Debug, Serialize)]
pub struct OptimizerTrialRow {
    pub study_id: String,
    pub trial_id: i64,
    pub ts_ms: i64,
    pub params_json: String,
    pub score: f64,
    pub sortino: f64,
    pub max_dd_bp: f64,
    pub turnover: f64,
    pub pareto_rank: i32,
}

#[derive(Clone, Debug, Serialize)]
pub struct LockboxRow {
    pub run_id: String,
    pub ts_ms: i64,
    pub model_id: String,
    pub params_id: String,
    pub summary_json: String,
    pub sealed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelMetricsRow {
    pub model_id: String,
    pub instrument: String,
    pub ts_ms: i64,
    pub oos_auc: f64,
    pub oos_log_loss: f64,
    pub oos_brier: f64,
    pub oos_balanced_acc: f64,
    pub train_sharpe: f64,
    pub train_sortino: f64,
    pub max_train_sortino: f64,
    pub max_train_sharpe: f64,
    pub n_train: i64,
    pub n_oof: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TraderMetricsRow {
    pub params_id: String,
    pub model_id: String,
    pub ts_ms: i64,
    pub in_sample_sharpe: f64,
    pub in_sample_sortino: f64,
    pub fine_tune_sharpe: f64,
    pub fine_tune_sortino: f64,
    pub max_dd_bp: f64,
    pub turnover_per_day: f64,
    pub hit_rate: f64,
    pub profit_factor: f64,
    pub n_trades: i64,
    pub params_json: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct LabelRow {
    pub instrument: String,
    pub ts_ms: i64,
    pub t1_ms: i64,
    pub side: i8,
    pub meta_y: i8,
    pub realized_r: f64,
    pub barrier_hit: String,
    pub oracle_score: f64,
    pub label_run_id: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelDeploymentGateRow {
    pub model_id: String,
    pub instrument: String,
    pub ts_ms: i64,
    pub oos_auc: f64,
    pub oos_log_loss: f64,
    pub oos_brier: f64,
    pub oos_balanced_acc: f64,
    pub fine_tune_sortino: f64,
    pub fine_tune_max_dd_bp: f64,
    pub passed_gate: bool,
    pub blocked_reasons: String,
    pub gate_thresholds_json: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelCandidateRow {
    pub run_id: String,
    pub spec_name: String,
    pub model_id: String,
    pub instrument: String,
    pub ts_ms: i64,
    pub oos_auc: f64,
    pub oos_log_loss: f64,
    pub oos_brier: f64,
    pub oos_balanced_acc: f64,
    pub n_train: i64,
    pub n_oof: i64,
    pub is_winner: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct PipelineRunRow {
    pub run_id: String,
    pub command: String,
    pub instrument: Option<String>,
    pub args_json: String,
    pub ts_started_ms: i64,
    pub ts_finished_ms: Option<i64>,
    /// `running` | `success` | `failed`.
    pub status: String,
    pub elapsed_ms: Option<i64>,
    pub error_msg: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TradeLedgerRow {
    pub run_id: String,
    pub instrument: String,
    pub ts_in_ms: i64,
    pub ts_out_ms: i64,
    pub side: i8,
    pub qty: f64,
    pub entry_px: f64,
    pub exit_px: f64,
    pub fee: f64,
    pub slip: f64,
    pub realized_r: f64,
    pub exit_reason: String,
    // Phase D enrichment (nullable: pre-Phase-D rows return None).
    pub model_id: Option<String>,
    pub params_id: Option<String>,
    pub entry_p_long: Option<f64>,
    pub entry_p_short: Option<f64>,
    pub entry_calibrated: Option<f64>,
    pub entry_spread_bp: Option<f64>,
    pub entry_atr_14: Option<f64>,
    pub exit_p_long: Option<f64>,
    pub exit_p_short: Option<f64>,
    pub decision_chain: Option<String>,
    pub snapshot_path: Option<String>,
}

/// Input row for `Db::insert_trade_ledger`. Phase D enriches the
/// trade ledger with the entry/exit signal context the trader saw at
/// decision time so post-mortem analysis doesn't need to re-join
/// `champion_signals` (which is shed every ~28h).
#[derive(Clone, Debug, Default)]
pub struct TradeLedgerInsert<'a> {
    pub run_id: &'a str,
    pub instrument: &'a str,
    pub ts_in_ms: i64,
    pub ts_out_ms: i64,
    pub side: i8,
    pub qty: f64,
    pub entry_px: f64,
    pub exit_px: f64,
    pub fee: f64,
    pub slip: f64,
    pub realized_r: f64,
    pub exit_reason: &'a str,
    pub model_id: Option<&'a str>,
    pub params_id: Option<&'a str>,
    pub entry_p_long: Option<f64>,
    pub entry_p_short: Option<f64>,
    pub entry_calibrated: Option<f64>,
    pub entry_spread_bp: Option<f64>,
    pub entry_atr_14: Option<f64>,
    pub exit_p_long: Option<f64>,
    pub exit_p_short: Option<f64>,
    /// JSON-encoded list of decision rules that fired (entry + exit).
    pub decision_chain: Option<&'a str>,
    /// Absolute path to the per-trade JSON snapshot file (D3).
    pub snapshot_path: Option<&'a str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChampionSignalRow {
    pub instrument: String,
    pub ts_ms: i64,
    pub p_long: f64,
    pub p_short: f64,
    pub p_take: f64,
    pub calibrated: f64,
    pub model_id: String,
    pub kind: String,
}

impl Db {
    /// Most-recent N optimizer trials, optionally filtered by study_id.
    pub fn recent_optimizer_trials(
        &self,
        study_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<OptimizerTrialRow>> {
        let limit = limit.min(MAX_HISTORY_LIMIT) as i64;
        let conn = self.inner.lock();
        let map = |row: &duckdb::Row<'_>| -> duckdb::Result<OptimizerTrialRow> {
            Ok(OptimizerTrialRow {
                study_id: row.get(0)?,
                trial_id: row.get(1)?,
                ts_ms: row.get(2)?,
                params_json: row.get(3)?,
                score: row.get(4)?,
                sortino: row.get(5)?,
                max_dd_bp: row.get(6)?,
                turnover: row.get(7)?,
                pareto_rank: row.get(8)?,
            })
        };
        let mut out = Vec::new();
        if let Some(sid) = study_id {
            let mut stmt = conn.prepare(
                "SELECT study_id, trial_id, ts_ms, params_json, score, sortino,
                        max_dd_bp, turnover, pareto_rank
                 FROM optimizer_trials
                 WHERE study_id = ?
                 ORDER BY trial_id DESC
                 LIMIT ?",
            )?;
            for r in stmt.query_map(params![sid, limit], map)? {
                out.push(r?);
            }
        } else {
            let mut stmt = conn.prepare(
                "SELECT study_id, trial_id, ts_ms, params_json, score, sortino,
                        max_dd_bp, turnover, pareto_rank
                 FROM optimizer_trials
                 ORDER BY ts_ms DESC
                 LIMIT ?",
            )?;
            for r in stmt.query_map(params![limit], map)? {
                out.push(r?);
            }
        }
        Ok(out)
    }

    /// Most recent lockbox result. The dashboard's "lockbox status"
    /// chip reads this.
    pub fn latest_lockbox_result(&self) -> Result<Option<LockboxRow>> {
        let conn = self.inner.lock();
        let mut stmt = conn.prepare(
            "SELECT run_id, ts_ms, model_id, params_id, summary_json, sealed
             FROM lockbox_results
             ORDER BY ts_ms DESC
             LIMIT 1",
        )?;
        let mut iter = stmt.query_map([], |row| {
            Ok(LockboxRow {
                run_id: row.get(0)?,
                ts_ms: row.get(1)?,
                model_id: row.get(2)?,
                params_id: row.get(3)?,
                summary_json: row.get(4)?,
                sealed: row.get(5)?,
            })
        })?;
        match iter.next() {
            Some(Ok(row)) => Ok(Some(row)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// Most recent model_metrics row for `instrument`. The dashboard's
    /// "champion model" chip reads this.
    pub fn latest_model_metrics(&self, instrument: &str) -> Result<Option<ModelMetricsRow>> {
        let conn = self.inner.lock();
        let mut stmt = conn.prepare(
            "SELECT model_id, instrument, ts_ms, oos_auc, oos_log_loss, oos_brier,
                    oos_balanced_acc, train_sharpe, train_sortino, max_train_sortino,
                    max_train_sharpe, n_train, n_oof
             FROM model_metrics
             WHERE instrument = ?
             ORDER BY ts_ms DESC
             LIMIT 1",
        )?;
        let mut iter = stmt.query_map(params![instrument], |row| {
            Ok(ModelMetricsRow {
                model_id: row.get(0)?,
                instrument: row.get(1)?,
                ts_ms: row.get(2)?,
                oos_auc: row.get(3)?,
                oos_log_loss: row.get(4)?,
                oos_brier: row.get(5)?,
                oos_balanced_acc: row.get(6)?,
                train_sharpe: row.get(7)?,
                train_sortino: row.get(8)?,
                max_train_sortino: row.get(9)?,
                max_train_sharpe: row.get(10)?,
                n_train: row.get(11)?,
                n_oof: row.get(12)?,
            })
        })?;
        match iter.next() {
            Some(Ok(row)) => Ok(Some(row)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// Look up a `model_metrics` row by its `model_id`. Used by the
    /// live-trader's JSONL preview writer so each closed trade can carry
    /// the OOS scores of the model that produced its entry signal,
    /// regardless of instrument-row uniqueness.
    pub fn model_metrics_by_id(&self, model_id: &str) -> Result<Option<ModelMetricsRow>> {
        let conn = self.inner.lock();
        let mut stmt = conn.prepare(
            "SELECT model_id, instrument, ts_ms, oos_auc, oos_log_loss, oos_brier,
                    oos_balanced_acc, train_sharpe, train_sortino, max_train_sortino,
                    max_train_sharpe, n_train, n_oof
             FROM model_metrics
             WHERE model_id = ?
             LIMIT 1",
        )?;
        let mut iter = stmt.query_map(params![model_id], |row| {
            Ok(ModelMetricsRow {
                model_id: row.get(0)?,
                instrument: row.get(1)?,
                ts_ms: row.get(2)?,
                oos_auc: row.get(3)?,
                oos_log_loss: row.get(4)?,
                oos_brier: row.get(5)?,
                oos_balanced_acc: row.get(6)?,
                train_sharpe: row.get(7)?,
                train_sortino: row.get(8)?,
                max_train_sortino: row.get(9)?,
                max_train_sharpe: row.get(10)?,
                n_train: row.get(11)?,
                n_oof: row.get(12)?,
            })
        })?;
        match iter.next() {
            Some(Ok(row)) => Ok(Some(row)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// Most recent trader_metrics row. Used by the dashboard's
    /// "current trader" chip.
    pub fn latest_trader_metrics(&self) -> Result<Option<TraderMetricsRow>> {
        let conn = self.inner.lock();
        let mut stmt = conn.prepare(
            "SELECT params_id, model_id, ts_ms, in_sample_sharpe, in_sample_sortino,
                    fine_tune_sharpe, fine_tune_sortino, max_dd_bp, turnover_per_day,
                    hit_rate, profit_factor, n_trades, params_json
             FROM trader_metrics
             ORDER BY ts_ms DESC
             LIMIT 1",
        )?;
        let mut iter = stmt.query_map([], |row| {
            Ok(TraderMetricsRow {
                params_id: row.get(0)?,
                model_id: row.get(1)?,
                ts_ms: row.get(2)?,
                in_sample_sharpe: row.get(3)?,
                in_sample_sortino: row.get(4)?,
                fine_tune_sharpe: row.get(5)?,
                fine_tune_sortino: row.get(6)?,
                max_dd_bp: row.get(7)?,
                turnover_per_day: row.get(8)?,
                hit_rate: row.get(9)?,
                profit_factor: row.get(10)?,
                n_trades: row.get(11)?,
                params_json: row.get(12)?,
            })
        })?;
        match iter.next() {
            Some(Ok(row)) => Ok(Some(row)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// Recent labels for an instrument; ordered ascending so the
    /// dashboard can overlay them on the price chart.
    pub fn recent_labels(&self, instrument: &str, limit: usize) -> Result<Vec<LabelRow>> {
        let limit = limit.min(MAX_HISTORY_LIMIT) as i64;
        let conn = self.inner.lock();
        let mut stmt = conn.prepare(
            "SELECT instrument, ts_ms, t1_ms, side, meta_y, realized_r,
                    barrier_hit, oracle_score, label_run_id
             FROM labels
             WHERE instrument = ?
             ORDER BY ts_ms DESC
             LIMIT ?",
        )?;
        let rows = stmt.query_map(params![instrument, limit], |row| {
            Ok(LabelRow {
                instrument: row.get(0)?,
                ts_ms: row.get(1)?,
                t1_ms: row.get(2)?,
                side: row.get(3)?,
                meta_y: row.get(4)?,
                realized_r: row.get(5)?,
                barrier_hit: row.get(6)?,
                oracle_score: row.get(7)?,
                label_run_id: row.get(8)?,
            })
        })?;
        let mut out: Vec<_> = rows.collect::<duckdb::Result<Vec<_>>>()?;
        out.reverse();
        Ok(out)
    }

    /// Stream the trade-ledger join (× champion_signals × model_metrics)
    /// to a file in JSONL or PARQUET format using DuckDB's native
    /// `COPY ... TO`. Returns the absolute path written.
    ///
    /// `format` is one of `"jsonl"` | `"parquet"`. Time range is
    /// inclusive on both ends; `instrument` is None to scan every
    /// instrument.
    ///
    /// The query joins:
    ///   * trade_ledger (one row per round-trip)
    ///   * model_metrics (champion's training scores at trade time —
    ///     joined by model_id)
    /// `champion_signals` is intentionally NOT joined here because
    /// rows for old trades may be shed; the trade row already carries
    /// entry_p_long/short/calibrated as Phase D enrichment.
    pub fn export_research_trades_to_file(
        &self,
        out_path: &std::path::Path,
        format: &str,
        instrument: Option<&str>,
        start_ms: i64,
        end_ms: i64,
        limit: usize,
    ) -> Result<()> {
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let limit = limit.min(MAX_HISTORY_LIMIT) as i64;
        let format_clause = match format {
            "parquet" | "PARQUET" => "(FORMAT PARQUET)",
            _ => "(FORMAT JSON, ARRAY false)",
        };
        let where_inst = if instrument.is_some() {
            "AND tl.instrument = ?"
        } else {
            ""
        };
        let path_str = out_path.to_string_lossy().replace('\'', "''");
        let sql = format!(
            "COPY (
                SELECT
                    tl.run_id, tl.instrument, tl.ts_in_ms, tl.ts_out_ms,
                    tl.side, tl.qty, tl.entry_px, tl.exit_px,
                    tl.fee, tl.slip, tl.realized_r, tl.exit_reason,
                    tl.model_id, tl.params_id,
                    tl.entry_p_long, tl.entry_p_short, tl.entry_calibrated,
                    tl.entry_spread_bp, tl.entry_atr_14,
                    tl.exit_p_long, tl.exit_p_short,
                    tl.decision_chain, tl.snapshot_path,
                    mm.oos_auc AS model_oos_auc,
                    mm.oos_log_loss AS model_oos_log_loss,
                    mm.oos_brier AS model_oos_brier,
                    mm.train_sortino AS model_train_sortino
                FROM trade_ledger tl
                LEFT JOIN model_metrics mm ON mm.model_id = tl.model_id
                WHERE tl.ts_out_ms BETWEEN ? AND ?
                {where_inst}
                ORDER BY tl.ts_out_ms DESC
                LIMIT ?
            ) TO '{path_str}' {format_clause}",
        );
        let conn = self.inner.lock();
        if let Some(inst) = instrument {
            conn.execute(&sql, params![start_ms, end_ms, inst, limit])?;
        } else {
            conn.execute(&sql, params![start_ms, end_ms, limit])?;
        }
        Ok(())
    }

    /// Append one closed round-trip to `trade_ledger`. Used by the
    /// `live-trader` runner; idempotent in the sense that ledger rows
    /// are append-only and never deduplicated.
    pub fn insert_trade_ledger(&self, row: &TradeLedgerInsert) -> Result<()> {
        let conn = self.inner.lock();
        conn.execute(
            "INSERT INTO trade_ledger
             (run_id, instrument, ts_in_ms, ts_out_ms, side, qty, entry_px, exit_px,
              fee, slip, realized_r, exit_reason,
              model_id, params_id, entry_p_long, entry_p_short, entry_calibrated,
              entry_spread_bp, entry_atr_14, exit_p_long, exit_p_short,
              decision_chain, snapshot_path)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                     ?, ?, ?, ?, ?,
                     ?, ?, ?, ?,
                     ?, ?)",
            params![
                row.run_id,
                row.instrument,
                row.ts_in_ms,
                row.ts_out_ms,
                row.side,
                row.qty,
                row.entry_px,
                row.exit_px,
                row.fee,
                row.slip,
                row.realized_r,
                row.exit_reason,
                row.model_id,
                row.params_id,
                row.entry_p_long,
                row.entry_p_short,
                row.entry_calibrated,
                row.entry_spread_bp,
                row.entry_atr_14,
                row.exit_p_long,
                row.exit_p_short,
                row.decision_chain,
                row.snapshot_path,
            ],
        )?;
        Ok(())
    }

    /// Look up a single `trade_ledger` row by its `(run_id, ts_in_ms)`
    /// composite key. Used by `/api/trade/context` to enrich the
    /// per-trade context payload (e.g. snapshot file path).
    pub fn trade_ledger_by_key(
        &self,
        run_id: &str,
        ts_in_ms: i64,
    ) -> Result<Option<TradeLedgerRow>> {
        let conn = self.inner.lock();
        let select_cols = "run_id, instrument, ts_in_ms, ts_out_ms, side, qty, \
                           entry_px, exit_px, fee, slip, realized_r, exit_reason, \
                           model_id, params_id, entry_p_long, entry_p_short, \
                           entry_calibrated, entry_spread_bp, entry_atr_14, \
                           exit_p_long, exit_p_short, decision_chain, snapshot_path";
        let sql = format!(
            "SELECT {select_cols} FROM trade_ledger \
             WHERE run_id = ? AND ts_in_ms = ? \
             ORDER BY ts_out_ms DESC LIMIT 1",
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut iter = stmt.query_map(params![run_id, ts_in_ms], |row| {
            Ok(TradeLedgerRow {
                run_id: row.get(0)?,
                instrument: row.get(1)?,
                ts_in_ms: row.get(2)?,
                ts_out_ms: row.get(3)?,
                side: row.get(4)?,
                qty: row.get(5)?,
                entry_px: row.get(6)?,
                exit_px: row.get(7)?,
                fee: row.get(8)?,
                slip: row.get(9)?,
                realized_r: row.get(10)?,
                exit_reason: row.get(11)?,
                model_id: row.get(12)?,
                params_id: row.get(13)?,
                entry_p_long: row.get(14)?,
                entry_p_short: row.get(15)?,
                entry_calibrated: row.get(16)?,
                entry_spread_bp: row.get(17)?,
                entry_atr_14: row.get(18)?,
                exit_p_long: row.get(19)?,
                exit_p_short: row.get(20)?,
                decision_chain: row.get(21)?,
                snapshot_path: row.get(22)?,
            })
        })?;
        match iter.next() {
            Some(Ok(r)) => Ok(Some(r)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// Most recent deployment-gate row for `instrument`. Returns
    /// `None` when no training run has been gated for that instrument
    /// yet. The dashboard's `DeploymentGateCard` reads this to show
    /// the current champion's score vs each floor and the blocked
    /// reasons (if any) of the most-recent attempt.
    pub fn latest_deployment_gate(
        &self,
        instrument: &str,
    ) -> Result<Option<ModelDeploymentGateRow>> {
        let conn = self.inner.lock();
        let mut stmt = conn.prepare(
            "SELECT model_id, instrument, ts_ms, oos_auc, oos_log_loss,
                    oos_brier, oos_balanced_acc, fine_tune_sortino,
                    fine_tune_max_dd_bp, passed_gate, blocked_reasons,
                    gate_thresholds_json
             FROM model_deployment_gate
             WHERE instrument = ?
             ORDER BY ts_ms DESC LIMIT 1",
        )?;
        let mut iter = stmt.query_map(params![instrument], |row| {
            Ok(ModelDeploymentGateRow {
                model_id: row.get(0)?,
                instrument: row.get(1)?,
                ts_ms: row.get(2)?,
                oos_auc: row.get(3)?,
                oos_log_loss: row.get(4)?,
                oos_brier: row.get(5)?,
                oos_balanced_acc: row.get(6)?,
                fine_tune_sortino: row.get(7)?,
                fine_tune_max_dd_bp: row.get(8)?,
                passed_gate: row.get(9)?,
                blocked_reasons: row.get(10)?,
                gate_thresholds_json: row.get(11)?,
            })
        })?;
        match iter.next() {
            Some(Ok(row)) => Ok(Some(row)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// All candidates from the most recent training run for `instrument`.
    /// Ordered by OOS log loss ascending (winner first). Returns empty
    /// when the instrument has no `model_candidates` rows yet.
    pub fn latest_model_candidates(
        &self,
        instrument: &str,
    ) -> Result<Vec<ModelCandidateRow>> {
        let conn = self.inner.lock();
        // The most recent run_id for this instrument:
        let run_id: Option<String> = conn
            .query_row(
                "SELECT run_id FROM model_candidates
                 WHERE instrument = ?
                 ORDER BY ts_ms DESC LIMIT 1",
                params![instrument],
                |row| row.get(0),
            )
            .ok();
        let run_id = match run_id {
            Some(r) => r,
            None => return Ok(vec![]),
        };
        let mut stmt = conn.prepare(
            "SELECT run_id, spec_name, model_id, instrument, ts_ms,
                    oos_auc, oos_log_loss, oos_brier, oos_balanced_acc,
                    n_train, n_oof, is_winner
             FROM model_candidates
             WHERE run_id = ?
             ORDER BY oos_log_loss ASC",
        )?;
        let rows = stmt.query_map(params![run_id], |row| {
            Ok(ModelCandidateRow {
                run_id: row.get(0)?,
                spec_name: row.get(1)?,
                model_id: row.get(2)?,
                instrument: row.get(3)?,
                ts_ms: row.get(4)?,
                oos_auc: row.get(5)?,
                oos_log_loss: row.get(6)?,
                oos_brier: row.get(7)?,
                oos_balanced_acc: row.get(8)?,
                n_train: row.get(9)?,
                n_oof: row.get(10)?,
                is_winner: row.get(11)?,
            })
        })?;
        Ok(rows.collect::<duckdb::Result<Vec<_>>>()?)
    }

    /// Recent Python pipeline runs from `pipeline_runs`, newest first.
    /// Powers the dashboard's "retrain history" panel.
    pub fn recent_pipeline_runs(&self, limit: usize) -> Result<Vec<PipelineRunRow>> {
        let limit = limit.min(MAX_HISTORY_LIMIT) as i64;
        let conn = self.inner.lock();
        let mut stmt = conn.prepare(
            "SELECT run_id, command, instrument, args_json, ts_started_ms,
                    ts_finished_ms, status, elapsed_ms, error_msg
             FROM pipeline_runs
             ORDER BY ts_started_ms DESC
             LIMIT ?",
        )?;
        let rows = stmt.query_map(params![limit], |row| {
            Ok(PipelineRunRow {
                run_id: row.get(0)?,
                command: row.get(1)?,
                instrument: row.get(2)?,
                args_json: row.get(3)?,
                ts_started_ms: row.get(4)?,
                ts_finished_ms: row.get(5)?,
                status: row.get(6)?,
                elapsed_ms: row.get(7)?,
                error_msg: row.get(8)?,
            })
        })?;
        Ok(rows.collect::<duckdb::Result<Vec<_>>>()?)
    }

    /// Recent closed round-trip trades from `trade_ledger`. When
    /// `instrument` is `Some`, results are filtered; otherwise the most
    /// recent trades across every instrument are returned. Ordered
    /// ascending by exit time so the dashboard can build a cumulative-R
    /// curve directly.
    pub fn recent_trade_ledger(
        &self,
        instrument: Option<&str>,
        limit: usize,
    ) -> Result<Vec<TradeLedgerRow>> {
        let limit = limit.min(MAX_HISTORY_LIMIT) as i64;
        let conn = self.inner.lock();
        let map = |row: &duckdb::Row<'_>| -> duckdb::Result<TradeLedgerRow> {
            Ok(TradeLedgerRow {
                run_id: row.get(0)?,
                instrument: row.get(1)?,
                ts_in_ms: row.get(2)?,
                ts_out_ms: row.get(3)?,
                side: row.get(4)?,
                qty: row.get(5)?,
                entry_px: row.get(6)?,
                exit_px: row.get(7)?,
                fee: row.get(8)?,
                slip: row.get(9)?,
                realized_r: row.get(10)?,
                exit_reason: row.get(11)?,
                model_id: row.get(12)?,
                params_id: row.get(13)?,
                entry_p_long: row.get(14)?,
                entry_p_short: row.get(15)?,
                entry_calibrated: row.get(16)?,
                entry_spread_bp: row.get(17)?,
                entry_atr_14: row.get(18)?,
                exit_p_long: row.get(19)?,
                exit_p_short: row.get(20)?,
                decision_chain: row.get(21)?,
                snapshot_path: row.get(22)?,
            })
        };
        let select_cols = "run_id, instrument, ts_in_ms, ts_out_ms, side, qty, \
                           entry_px, exit_px, fee, slip, realized_r, exit_reason, \
                           model_id, params_id, entry_p_long, entry_p_short, \
                           entry_calibrated, entry_spread_bp, entry_atr_14, \
                           exit_p_long, exit_p_short, decision_chain, snapshot_path";
        let mut out: Vec<TradeLedgerRow> = if let Some(inst) = instrument {
            let sql = format!(
                "SELECT {select_cols} FROM trade_ledger \
                 WHERE instrument = ? ORDER BY ts_out_ms DESC LIMIT ?",
            );
            let mut stmt = conn.prepare(&sql)?;
            stmt.query_map(params![inst, limit], map)?
                .collect::<duckdb::Result<Vec<_>>>()?
        } else {
            let sql = format!(
                "SELECT {select_cols} FROM trade_ledger \
                 ORDER BY ts_out_ms DESC LIMIT ?",
            );
            let mut stmt = conn.prepare(&sql)?;
            stmt.query_map(params![limit], map)?
                .collect::<duckdb::Result<Vec<_>>>()?
        };
        out.reverse();
        Ok(out)
    }

    /// Recent champion-model signals for an instrument; ascending order.
    pub fn recent_champion_signals(
        &self,
        instrument: &str,
        limit: usize,
    ) -> Result<Vec<ChampionSignalRow>> {
        let limit = limit.min(MAX_HISTORY_LIMIT) as i64;
        let conn = self.inner.lock();
        let mut stmt = conn.prepare(
            "SELECT instrument, ts_ms, p_long, p_short, p_take, calibrated, model_id, kind
             FROM champion_signals
             WHERE instrument = ?
             ORDER BY ts_ms DESC
             LIMIT ?",
        )?;
        let rows = stmt.query_map(params![instrument, limit], |row| {
            Ok(ChampionSignalRow {
                instrument: row.get(0)?,
                ts_ms: row.get(1)?,
                p_long: row.get(2)?,
                p_short: row.get(3)?,
                p_take: row.get(4)?,
                calibrated: row.get(5)?,
                model_id: row.get(6)?,
                kind: row.get(7)?,
            })
        })?;
        let mut out: Vec<_> = rows.collect::<duckdb::Result<Vec<_>>>()?;
        out.reverse();
        Ok(out)
    }
}

//! Pipeline-run tracking. Wrap any unit of orchestrated work in a
//! `RunTracker` so the dashboard's "Pipeline runs" panel can show
//! what ran, when, how long it took, and whether it succeeded.
//!
//! Pure-Rust port of `research/observability/tracker.py` — the Rust
//! pipeline orchestrator (and any future Rust binary that wants to
//! be visible to the dashboard) calls this directly. The Python
//! tracker stays in place during the migration so existing Python
//! invocations keep recording rows; eventually it can be deleted
//! when no callers remain.
//!
//! Schema (matches `pipeline_runs` in `persistence::schema`):
//!
//! ```sql
//! CREATE TABLE pipeline_runs (
//!     run_id          VARCHAR PRIMARY KEY,
//!     command         VARCHAR NOT NULL,
//!     instrument      VARCHAR,
//!     args_json       VARCHAR NOT NULL,
//!     ts_started_ms   BIGINT  NOT NULL,
//!     ts_finished_ms  BIGINT,
//!     status          VARCHAR NOT NULL,   -- running | success | failed
//!     elapsed_ms      BIGINT,
//!     error_msg       VARCHAR
//! );
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! let mut t = observability::RunTracker::start(
//!     &db, "label", Some("EUR_USD"), &json!({ "n_bars": 1000 })
//! )?;
//! match do_the_work() {
//!     Ok(()) => t.success(&db),
//!     Err(e) => t.fail(&db, &format!("{e}")),
//! }
//! ```
//!
//! Failure modes are silently logged (not raised) — observability
//! shouldn't block real research work. If the DB is unavailable, the
//! tracker becomes a no-op and the caller continues.

#![deny(unsafe_code)]

use anyhow::Result;
use serde::{Deserialize, Serialize};

use persistence::Db;

/// Tracker handle. Created with `RunTracker::start`, finalised by
/// either `success(...)` or `fail(...)`. Internally records whether
/// the start INSERT succeeded so the finaliser can no-op cleanly when
/// the DB was unavailable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunTracker {
    pub run_id: String,
    pub started_ms: i64,
    pub command: String,
    pub instrument: Option<String>,
    /// True when the start INSERT was successful. Finalisers gate
    /// their UPDATE on this so a failed start doesn't try to update
    /// a row that doesn't exist.
    pub started_ok: bool,
}

impl RunTracker {
    /// Begin tracking. Inserts a row with `status='running'`. Returns
    /// the tracker even if the DB was unavailable; `started_ok=false`
    /// in that case and the finalisers will silently no-op.
    pub fn start(
        db: &Db,
        command: &str,
        instrument: Option<&str>,
        args_json: &serde_json::Value,
    ) -> Self {
        let run_id = uuid::Uuid::new_v4().simple().to_string();
        let started_ms = chrono::Utc::now().timestamp_millis();
        let args = serde_json::to_string(args_json).unwrap_or_else(|_| "{}".to_string());
        let inserted = insert_start(db, &run_id, command, instrument, &args, started_ms);
        let started_ok = match inserted {
            Ok(_) => true,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    command,
                    "observability: pipeline_runs start INSERT failed; tracker will no-op"
                );
                false
            }
        };
        Self {
            run_id,
            started_ms,
            command: command.to_string(),
            instrument: instrument.map(str::to_string),
            started_ok,
        }
    }

    pub fn success(&self, db: &Db) {
        if !self.started_ok {
            return;
        }
        if let Err(e) = finalize(db, &self.run_id, "success", None, self.started_ms) {
            tracing::warn!(
                error = %e,
                run_id = %self.run_id,
                "observability: pipeline_runs success UPDATE failed"
            );
        }
    }

    pub fn fail(&self, db: &Db, error_msg: &str) {
        if !self.started_ok {
            return;
        }
        if let Err(e) = finalize(db, &self.run_id, "failed", Some(error_msg), self.started_ms) {
            tracing::warn!(
                error = %e,
                run_id = %self.run_id,
                "observability: pipeline_runs failed UPDATE failed"
            );
        }
    }
}

fn insert_start(
    db: &Db,
    run_id: &str,
    command: &str,
    instrument: Option<&str>,
    args_json: &str,
    started_ms: i64,
) -> Result<()> {
    db.with_conn(|conn| -> duckdb::Result<()> {
        conn.execute(
            "INSERT INTO pipeline_runs
                 (run_id, command, instrument, args_json, ts_started_ms, status)
              VALUES (?, ?, ?, ?, ?, 'running')",
            duckdb::params![run_id, command, instrument, args_json, started_ms],
        )?;
        Ok(())
    })?;
    Ok(())
}

fn finalize(
    db: &Db,
    run_id: &str,
    status: &str,
    error_msg: Option<&str>,
    started_ms: i64,
) -> Result<()> {
    let finished = chrono::Utc::now().timestamp_millis();
    let elapsed = finished - started_ms;
    db.with_conn(|conn| -> duckdb::Result<()> {
        conn.execute(
            "UPDATE pipeline_runs
                 SET ts_finished_ms = ?, status = ?, elapsed_ms = ?, error_msg = ?
                 WHERE run_id = ?",
            duckdb::params![finished, status, elapsed, error_msg, run_id],
        )?;
        Ok(())
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn tmp_db(label: &str) -> (Db, PathBuf, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join(format!("{label}.duckdb"));
        let db = Db::open(&path).expect("open db");
        (db, path, dir)
    }

    #[test]
    fn success_path_writes_full_row() {
        let (db, _, _t) = tmp_db("obs_success");
        let t = RunTracker::start(
            &db,
            "label",
            Some("EUR_USD"),
            &serde_json::json!({"n_bars": 1000}),
        );
        assert!(t.started_ok);
        std::thread::sleep(std::time::Duration::from_millis(2));
        t.success(&db);

        let n_success: i64 = db.with_conn(|c| {
            c.query_row(
                "SELECT COUNT(*) FROM pipeline_runs WHERE status = 'success'",
                [],
                |r| r.get(0),
            )
            .unwrap()
        });
        assert_eq!(n_success, 1);
    }

    #[test]
    fn fail_path_records_error_message() {
        let (db, _, _t) = tmp_db("obs_fail");
        let t = RunTracker::start(
            &db,
            "train.side",
            None,
            &serde_json::json!({"k": "v"}),
        );
        t.fail(&db, "boom");
        let row: (String, Option<String>) = db.with_conn(|c| {
            c.query_row(
                "SELECT status, error_msg FROM pipeline_runs WHERE run_id = ?",
                duckdb::params![t.run_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap()
        });
        assert_eq!(row.0, "failed");
        assert_eq!(row.1.as_deref(), Some("boom"));
    }
}

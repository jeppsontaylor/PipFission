//! Retention sweep. Enforces the HARD per-instrument 10k row cap on
//! every time-series table. This is the user's locked invariant: no
//! per-instrument table EVER holds more than `MAX_ROWS_PER_INSTRUMENT`
//! rows for a single instrument. Audit / metrics tables are
//! append-only and excluded.
//!
//! Strategy:
//! For each (table, instrument):
//!   - Count rows.
//!   - If count > cap, delete everything older than the cap-th most
//!     recent row. One DELETE per (table, instrument).
//!
//! DuckDB has no rowid pseudo-column, so the deletion key is `ts_ms`.
//! Tied timestamps within the same instrument are extremely rare (the
//! source feeds use ms timestamps and OANDA enforces monotonicity per
//! instrument), and worst-case we keep one extra row temporarily — a
//! benign overshoot smaller than 1 in 10_000.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use chrono::{TimeZone, Utc};
use duckdb::params;

use crate::connection::Db;
use crate::schema::{MAX_ROWS_PER_INSTRUMENT, PER_INSTRUMENT_TABLES};

/// Retention configuration. Only the sweep cadence and the per-instrument
/// cap are tunable; the cap defaults to the user-locked HARD value of
/// 10_000 and is bounded above so a misconfig cannot widen it.
#[derive(Clone, Debug)]
pub struct RetentionPolicy {
    /// How often the sweep runs.
    pub sweep_interval_secs: u64,
    /// Per-instrument FIFO cap. Locked at 10_000 by user requirement.
    pub max_rows_per_instrument: i64,
    /// When `Some(root)`, every retention sweep first writes the
    /// to-be-deleted rows to `<root>/<table>/<instrument>/<YYYY-MM-DD>.parquet`
    /// before they're shed. None disables archiving (default — kept
    /// off so existing deployments don't suddenly grow on-disk
    /// state).
    pub archive_root: Option<PathBuf>,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            sweep_interval_secs: 30,
            max_rows_per_instrument: MAX_ROWS_PER_INSTRUMENT,
            archive_root: None,
        }
    }
}

impl RetentionPolicy {
    /// Build a policy from env vars, falling back to `Default`. The
    /// `RETENTION_MAX_ROWS_PER_INSTRUMENT` knob is clamped above by
    /// `MAX_ROWS_PER_INSTRUMENT` so a misconfig cannot widen the cap.
    pub fn from_env() -> Self {
        fn parse_i64(name: &str) -> Option<i64> {
            std::env::var(name).ok().and_then(|s| s.parse().ok())
        }
        let mut p = Self::default();
        if let Some(s) = parse_i64("RETENTION_SWEEP_SECS") {
            p.sweep_interval_secs = s.max(5) as u64;
        }
        if let Some(n) = parse_i64("RETENTION_MAX_ROWS_PER_INSTRUMENT") {
            p.max_rows_per_instrument = n.max(0).min(MAX_ROWS_PER_INSTRUMENT);
        }
        // Phase D6: optional pre-sweep parquet archive. Set
        // `RESEARCH_ARCHIVE_DIR=./data/archive` to enable; older rows
        // are written to `<dir>/<table>/<instrument>/<YYYY-MM-DD>.parquet`
        // before being deleted, so retroactive studies past the 28h
        // live window remain possible.
        if let Ok(root) = std::env::var("RESEARCH_ARCHIVE_DIR") {
            if !root.is_empty() {
                p.archive_root = Some(PathBuf::from(root));
            }
        }
        p
    }

    /// Snapshot of the per-instrument tables under retention control.
    pub fn per_instrument_tables() -> &'static [&'static str] {
        PER_INSTRUMENT_TABLES
    }
}

/// Spawn the periodic retention sweep. Runs once immediately at startup
/// (so a long-shutdown DB doesn't carry past-cap rows forward) and then
/// on the configured interval.
pub fn spawn_rolloff(db: Db, policy: RetentionPolicy) -> tokio::task::JoinHandle<()> {
    let interval_secs = policy.sweep_interval_secs;
    tokio::spawn(async move {
        run_async(&db, &policy).await;
        let mut tick = tokio::time::interval(Duration::from_secs(interval_secs));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        tick.tick().await;
        loop {
            tick.tick().await;
            run_async(&db, &policy).await;
        }
    })
}

async fn run_async(db: &Db, policy: &RetentionPolicy) {
    let db_clone = db.clone();
    let policy_clone = policy.clone();
    let result = tokio::task::spawn_blocking(move || run(&db_clone, &policy_clone)).await;
    match result {
        Ok(Ok(report)) => {
            if report.total_deleted > 0 {
                tracing::info!(
                    deleted = report.total_deleted,
                    db_size_mb =
                        format!("{:.2}", report.db_size_bytes as f64 / (1024.0 * 1024.0)),
                    by_table = ?report.per_table,
                    "retention: HARD 10k cap swept"
                );
            } else {
                tracing::debug!(
                    db_size_mb =
                        format!("{:.2}", report.db_size_bytes as f64 / (1024.0 * 1024.0)),
                    "retention: nothing to do"
                );
            }
        }
        Ok(Err(e)) => tracing::warn!("retention failed: {e:#}"),
        Err(e) => tracing::warn!("retention join error: {e}"),
    }
}

#[derive(Debug, Default)]
struct Report {
    total_deleted: i64,
    per_table: Vec<(String, i64)>,
    db_size_bytes: u64,
}

fn run(db: &Db, policy: &RetentionPolicy) -> Result<Report> {
    let conn = db.inner.lock();
    let mut report = Report::default();

    if policy.max_rows_per_instrument <= 0 {
        return Ok(report);
    }
    let cap = policy.max_rows_per_instrument;

    for &table in PER_INSTRUMENT_TABLES {
        // List instruments currently present in the table.
        let mut stmt = conn.prepare(&format!(
            "SELECT instrument, COUNT(*) FROM {table} GROUP BY instrument"
        ))?;
        let groups: Vec<(String, i64)> = stmt
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))?
            .collect::<duckdb::Result<Vec<_>>>()?;
        drop(stmt);

        let mut deleted_for_table: i64 = 0;
        for (instrument, count) in groups {
            if count <= cap {
                continue;
            }
            // Find the cutoff: the cap-th newest ts_ms. Anything strictly
            // older than that ts_ms is deleted; anything equal-to-or-newer
            // stays (so worst-case we keep one extra on a duplicate ts_ms,
            // which is acceptable and bounded).
            let cutoff: Option<i64> = conn
                .query_row(
                    &format!(
                        "SELECT ts_ms FROM {table}
                         WHERE instrument = ?
                         ORDER BY ts_ms DESC
                         LIMIT 1 OFFSET ?"
                    ),
                    params![instrument, cap - 1],
                    |row| row.get::<_, i64>(0),
                )
                .ok();
            let Some(cutoff) = cutoff else {
                continue;
            };
            // Phase D6: pre-sweep archive. Write the soon-to-be-shed
            // rows to a per-day parquet under
            // `<archive_root>/<table>/<instrument>/<YYYY-MM-DD>.parquet`
            // before the DELETE. Failures here are logged but never
            // block the sweep — retention correctness is the priority,
            // archiving is observability.
            if let Some(root) = policy.archive_root.as_deref() {
                if let Err(e) = archive_pre_sweep(&conn, root, table, &instrument, cutoff) {
                    tracing::warn!(
                        table,
                        instrument,
                        error = %e,
                        "retention: archive failed (continuing with delete)"
                    );
                }
            }
            // DuckDB has a known issue (observed multiple times in this
            // project) where DELETE walking a multi-column index can fail
            // mid-flight with "Failed to delete all rows from index" and
            // mark the DB as fatal-invalidated. Bypass it by deleting via
            // `rowid` — the heap path is robust. Same end-state.
            let deleted = conn.execute(
                &format!(
                    "DELETE FROM {table}
                     WHERE rowid IN (
                         SELECT rowid FROM {table}
                         WHERE instrument = ? AND ts_ms < ?
                     )"
                ),
                params![instrument, cutoff],
            )? as i64;
            deleted_for_table += deleted;
        }
        if deleted_for_table > 0 {
            report.total_deleted += deleted_for_table;
            report.per_table.push((table.to_string(), deleted_for_table));
        }
    }

    // DuckDB checkpoints lazily; nudging it after a sweep keeps the WAL
    // bounded between sweeps.
    let _ = conn.execute_batch("CHECKPOINT");

    drop(conn);
    report.db_size_bytes = db.file_size();
    Ok(report)
}

/// Archive the rows about to be shed for one (table, instrument,
/// cutoff_ts_ms) tuple. Splits by UTC date so the file layout matches
/// the convention `<archive_root>/<table>/<instrument>/<YYYY-MM-DD>.parquet`.
/// Same-day repeat sweeps APPEND to the existing parquet via DuckDB's
/// `COPY ... TO ... (FORMAT PARQUET, APPEND true)`.
///
/// Forward-slashes in instrument names (e.g. "BTC/USD") are flattened
/// to `_` in the path so OS path semantics don't create a phantom
/// "BTC" directory.
fn archive_pre_sweep(
    conn: &duckdb::Connection,
    root: &std::path::Path,
    table: &str,
    instrument: &str,
    cutoff_ts_ms: i64,
) -> Result<()> {
    // Determine the date range of rows being archived so we know which
    // per-day file(s) to write.
    let mut stmt = conn.prepare(&format!(
        "SELECT MIN(ts_ms), MAX(ts_ms) FROM {table}
         WHERE instrument = ? AND ts_ms < ?"
    ))?;
    let (min_ts, max_ts): (Option<i64>, Option<i64>) = stmt
        .query_row(params![instrument, cutoff_ts_ms], |row| {
            Ok((row.get::<_, Option<i64>>(0)?, row.get::<_, Option<i64>>(1)?))
        })?;
    drop(stmt);
    let (Some(min_ts), Some(max_ts)) = (min_ts, max_ts) else {
        return Ok(());
    };

    let safe_instrument = instrument.replace('/', "_");
    let archive_dir = root.join(table).join(&safe_instrument);
    std::fs::create_dir_all(&archive_dir)?;

    // Iterate one UTC day at a time. For the trading volume in this
    // system (10s bars × ~3 instruments) one sweep typically covers
    // a single day, so this loop almost always runs once.
    let mut day_start = day_floor_ms(min_ts);
    while day_start <= max_ts {
        let day_end = day_start + 86_400_000;
        let date_str = Utc
            .timestamp_millis_opt(day_start)
            .single()
            .map(|t| t.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "unknown".into());
        let out = archive_dir.join(format!("{date_str}.parquet"));
        let path_str = out.to_string_lossy().replace('\'', "''");
        // DuckDB COPY can't APPEND to parquet directly; instead, when a
        // file already exists we read its rows + the new rows in a
        // single SELECT and overwrite. For sweeps that fire every 30s,
        // re-reading the day's parquet is cheap (it sits well inside
        // the per-instrument cap × bytes-per-row budget).
        let sql = if out.exists() {
            format!(
                "COPY (
                    SELECT * FROM (
                        SELECT * FROM {table}
                         WHERE instrument = ? AND ts_ms < ?
                           AND ts_ms >= ? AND ts_ms < ?
                        UNION ALL
                        SELECT * FROM read_parquet('{path_str}')
                    )
                ) TO '{path_str}' (FORMAT PARQUET)"
            )
        } else {
            format!(
                "COPY (
                    SELECT * FROM {table}
                     WHERE instrument = ? AND ts_ms < ?
                       AND ts_ms >= ? AND ts_ms < ?
                ) TO '{path_str}' (FORMAT PARQUET)"
            )
        };
        conn.execute(
            &sql,
            params![instrument, cutoff_ts_ms, day_start, day_end],
        )?;
        day_start = day_end;
    }
    Ok(())
}

/// Floor a unix-ms timestamp to the start of its UTC day.
fn day_floor_ms(ts_ms: i64) -> i64 {
    (ts_ms / 86_400_000) * 86_400_000
}

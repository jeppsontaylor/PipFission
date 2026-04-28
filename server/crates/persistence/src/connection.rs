//! DuckDB connection wrapper. Single mutex-guarded connection backs both
//! the writer and the read-side methods used by the dashboard /api/history
//! routes. DuckDB internally parallelises query execution and releases the
//! Rust-side lock during long ops, so contention is acceptable for the
//! row volumes this system runs at (≤ 10k per instrument per table).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use duckdb::{params, Connection};
use parking_lot::Mutex;

use market_domain::{Bar10sNamed, FeatureVector, FEATURE_DIM};

use crate::schema;
use crate::{FillHistoryPoint, PriceHistoryPoint, SignalHistoryPoint, MAX_HISTORY_LIMIT};

/// Handle to the on-disk DuckDB database. Cheap to clone — internally an
/// `Arc<Mutex<Connection>>`.
#[derive(Clone)]
pub struct Db {
    pub(crate) inner: Arc<Mutex<Connection>>,
    pub(crate) path: PathBuf,
}

impl std::fmt::Debug for Db {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Db").field("path", &self.path).finish()
    }
}

impl Db {
    /// Open or create the database at `path`, run pragmas, apply the
    /// schema. Idempotent — safe to call on existing DBs.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("creating db parent dir {}", parent.display())
            })?;
        }
        let conn = Connection::open(&path)
            .with_context(|| format!("opening duckdb db {}", path.display()))?;

        conn.execute_batch(schema::PRAGMAS)
            .context("applying pragmas")?;
        // Phase 1 — CREATE TABLE / CREATE INDEX (idempotent for fresh
        // tables). New columns added after a table first existed do
        // NOT take effect via this path — DuckDB's `CREATE TABLE IF
        // NOT EXISTS` skips when the table already exists, ignoring
        // the column list.
        for stmt in schema::SCHEMA {
            conn.execute_batch(stmt)
                .with_context(|| format!("applying DDL: {stmt}"))?;
        }
        // Phase 2 — idempotent ALTER TABLE migrations. Each entry is
        // an `ALTER TABLE … ADD COLUMN IF NOT EXISTS …` statement;
        // DuckDB makes this a no-op when the column already exists,
        // so existing data is preserved on every server restart.
        // Loud failures here would lose data — earlier in this
        // project a missed migration on `trade_ledger` forced a full
        // DB rebuild, including the time-series tables. Adding
        // columns to an existing table MUST go through this list,
        // never via DROP + recreate.
        for stmt in schema::MIGRATIONS {
            if let Err(e) = conn.execute_batch(stmt) {
                tracing::warn!(
                    migration = stmt,
                    error = %e,
                    "persistence: migration failed (continuing — most likely already applied)"
                );
            }
        }

        Ok(Self {
            inner: Arc::new(Mutex::new(conn)),
            path,
        })
    }

    /// Current size of the DuckDB file in bytes. Useful for telemetry —
    /// the rolloff log line includes this so the user can verify the DB
    /// stays bounded.
    pub fn file_size(&self) -> u64 {
        std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Borrow the connection for an arbitrary closure. Used by callers
    /// that want one-off ad-hoc queries without going through the typed
    /// helpers.
    pub fn with_conn<R>(&self, f: impl FnOnce(&Connection) -> R) -> R {
        let g = self.inner.lock();
        f(&g)
    }

    /// Hydration shim kept for backwards compatibility with the existing
    /// strategy crate. The new pipeline does not persist feature vectors,
    /// so this always returns an empty map. The strategy will warm up
    /// from streaming ticks instead.
    pub fn hydrate_recent_features(
        &self,
        _per_instrument: usize,
    ) -> Result<HashMap<String, Vec<FeatureVector>>> {
        Ok(HashMap::new())
    }

    /// Same as `hydrate_recent_features` — returns empty for the new
    /// (no-feature-persistence) design. Existing callers in api-server
    /// log a "0 prefilled" line and the strategy boots cold.
    pub fn hydrate_recent_strategy_state(
        &self,
        _per_instrument: usize,
    ) -> Result<HashMap<String, Vec<(FeatureVector, f64)>>> {
        let _ = FEATURE_DIM; // keep the import live for future on-the-fly hydration
        Ok(HashMap::new())
    }

    /// Most recent N price ticks for an instrument, ordered ascending in
    /// time so the dashboard can append the live tail without re-sorting.
    /// Bounded by MAX_HISTORY_LIMIT.
    pub fn recent_price_ticks(
        &self,
        instrument: &str,
        limit: usize,
    ) -> Result<Vec<PriceHistoryPoint>> {
        let limit = limit.min(MAX_HISTORY_LIMIT) as i64;
        let conn = self.inner.lock();
        let mut stmt = conn.prepare(
            "SELECT ts_ms, bid, ask, mid, spread_bp, status
             FROM price_ticks
             WHERE instrument = ?
             ORDER BY ts_ms DESC
             LIMIT ?",
        )?;
        let rows = stmt.query_map(params![instrument, limit], |row| {
            Ok(PriceHistoryPoint {
                time_ms: row.get(0)?,
                bid: row.get(1)?,
                ask: row.get(2)?,
                mid: row.get(3)?,
                spread: row.get(4)?,
                status: row.get(5)?,
            })
        })?;
        let mut out: Vec<_> = rows.collect::<duckdb::Result<Vec<_>>>()?;
        out.reverse();
        Ok(out)
    }

    /// Most recent N closed 10s bars for an instrument, ascending order.
    pub fn recent_bars_10s(
        &self,
        instrument: &str,
        limit: usize,
    ) -> Result<Vec<Bar10sNamed>> {
        let limit = limit.min(MAX_HISTORY_LIMIT) as i64;
        let conn = self.inner.lock();
        let mut stmt = conn.prepare(
            "SELECT instrument, ts_ms, open, high, low, close, n_ticks, spread_bp_avg
             FROM bars_10s
             WHERE instrument = ?
             ORDER BY ts_ms DESC
             LIMIT ?",
        )?;
        let rows = stmt.query_map(params![instrument, limit], |row| {
            Ok(Bar10sNamed {
                instrument: row.get(0)?,
                ts_ms: row.get(1)?,
                open: row.get(2)?,
                high: row.get(3)?,
                low: row.get(4)?,
                close: row.get(5)?,
                n_ticks: {
                    let v: i64 = row.get(6)?;
                    v.max(0) as u32
                },
                spread_bp_avg: row.get(7)?,
            })
        })?;
        let mut out: Vec<_> = rows.collect::<duckdb::Result<Vec<_>>>()?;
        out.reverse();
        Ok(out)
    }

    /// Most recent N strategy signals for an instrument, ascending order.
    pub fn recent_signals(
        &self,
        instrument: &str,
        limit: usize,
    ) -> Result<Vec<SignalHistoryPoint>> {
        let limit = limit.min(MAX_HISTORY_LIMIT) as i64;
        let conn = self.inner.lock();
        let mut stmt = conn.prepare(
            "SELECT ts_ms, direction, confidence, prob_long, prob_flat, prob_short,
                    model_id, model_version
             FROM signals
             WHERE instrument = ?
             ORDER BY ts_ms DESC
             LIMIT ?",
        )?;
        let rows = stmt.query_map(params![instrument, limit], |row| {
            Ok(SignalHistoryPoint {
                time_ms: row.get(0)?,
                direction: row.get(1)?,
                confidence: row.get(2)?,
                prob_long: row.get(3)?,
                prob_flat: row.get(4)?,
                prob_short: row.get(5)?,
                model_id: row.get(6)?,
                model_version: row.get(7)?,
            })
        })?;
        let mut out: Vec<_> = rows.collect::<duckdb::Result<Vec<_>>>()?;
        out.reverse();
        Ok(out)
    }

    /// Most recent N paper fills, optionally filtered by instrument.
    pub fn recent_fills(
        &self,
        instrument: Option<&str>,
        limit: usize,
    ) -> Result<Vec<FillHistoryPoint>> {
        let limit = limit.min(MAX_HISTORY_LIMIT) as i64;
        let conn = self.inner.lock();
        let map = |row: &duckdb::Row<'_>| -> duckdb::Result<FillHistoryPoint> {
            Ok(FillHistoryPoint {
                instrument: row.get(0)?,
                time_ms: row.get(1)?,
                units: row.get(2)?,
                price: row.get(3)?,
                fee: row.get(4)?,
                mode: row.get(5)?,
                order_id: row.get(6)?,
            })
        };
        let rows: Vec<_> = if let Some(inst) = instrument {
            let mut stmt = conn.prepare(
                "SELECT instrument, ts_ms, units, price, fee, mode, order_id
                 FROM paper_fills
                 WHERE instrument = ?
                 ORDER BY ts_ms DESC
                 LIMIT ?",
            )?;
            stmt.query_map(params![inst, limit], map)?
                .collect::<duckdb::Result<Vec<_>>>()?
        } else {
            let mut stmt = conn.prepare(
                "SELECT instrument, ts_ms, units, price, fee, mode, order_id
                 FROM paper_fills
                 ORDER BY ts_ms DESC
                 LIMIT ?",
            )?;
            stmt.query_map(params![limit], map)?
                .collect::<duckdb::Result<Vec<_>>>()?
        };
        let mut out = rows;
        out.reverse();
        Ok(out)
    }

    /// Total row counts per table — for the dashboard / boot log.
    pub fn row_counts(&self) -> Result<HashMap<&'static str, i64>> {
        let conn = self.inner.lock();
        let tables: &[&'static str] = &[
            "price_ticks",
            "bars_10s",
            "labels",
            "oof_predictions",
            "signals",
            "champion_signals",
            "paper_fills",
            "trade_ledger",
            "model_metrics",
            "trader_metrics",
            "optimizer_trials",
            "lockbox_results",
            "model_artifacts",
            "fitness",
            "account_snapshots",
        ];
        let mut out = HashMap::new();
        for &t in tables {
            let n: i64 = conn.query_row(&format!("SELECT COUNT(*) FROM {t}"), [], |r| r.get(0))?;
            out.insert(t, n);
        }
        Ok(out)
    }
}

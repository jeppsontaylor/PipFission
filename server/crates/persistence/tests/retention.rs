//! End-to-end test: HARD 10k cap is enforced. Insert 11_000 rows for
//! one instrument, run retention, assert exactly 10_000 remain.
//!
//! This locks the user-stated invariant: no per-instrument table ever
//! holds more than 10_000 rows. Regressions here are blockers.

use std::path::PathBuf;

use duckdb::params;
use persistence::{Db, RetentionPolicy};

fn tmp_db_path(label: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    p.push(format!("persistence_test_{label}_{pid}_{ts}.duckdb"));
    p
}

#[test]
fn hard_cap_enforced_at_10k_per_instrument() {
    let path = tmp_db_path("cap");
    let _ = std::fs::remove_file(&path);
    let db = Db::open(&path).expect("open db");

    // Insert 11_000 ticks for one instrument. We hand-roll the INSERT
    // because the regular writer goes through the broadcast bus.
    db.with_conn(|conn| {
        let tx = conn.unchecked_transaction().expect("begin");
        for i in 0..11_000_i64 {
            tx.execute(
                "INSERT INTO price_ticks
                 (instrument, ts_ms, bid, ask, mid, spread_bp, status)
                 VALUES (?, ?, ?, ?, ?, ?, ?)",
                params!["EUR_USD", 1_000_000_i64 + i, 1.0, 1.0001, 1.00005, 1.0, "tradeable"],
            )
            .expect("insert");
        }
        tx.commit().expect("commit");
    });

    let counts = db.row_counts().expect("row_counts");
    assert_eq!(counts.get("price_ticks").copied(), Some(11_000));

    // Run the retention sweep synchronously (private fn — go via the
    // public driver, which spawns blocking work).
    let policy = RetentionPolicy::default();
    let rt = tokio::runtime::Runtime::new().expect("rt");
    rt.block_on(async {
        // Spawn the sweep, wait long enough for the immediate kick, then
        // abort. The first run is synchronous after spawn — the function
        // calls `run_async` once before entering the interval loop.
        let handle = persistence::spawn_rolloff(db.clone(), policy);
        // Yield until the immediate sweep completes; one event loop pass
        // is enough because spawn_blocking returns through tokio.
        for _ in 0..200 {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let counts = db.row_counts().unwrap_or_default();
            if counts.get("price_ticks").copied() == Some(10_000) {
                break;
            }
        }
        handle.abort();
    });

    let counts = db.row_counts().expect("row_counts after sweep");
    assert_eq!(
        counts.get("price_ticks").copied(),
        Some(10_000),
        "HARD 10k cap was breached: {:?}",
        counts.get("price_ticks")
    );

    let _ = std::fs::remove_file(&path);
}

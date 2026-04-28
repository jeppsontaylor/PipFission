//! `init_schema` — apply the full DuckDB schema to a database at a path.
//!
//! Used by Python research workflows that need the schema present
//! without booting the live OANDA server. Idempotent: running it
//! against an existing fully-initialised DB is a no-op (CREATE TABLE
//! IF NOT EXISTS / CREATE INDEX IF NOT EXISTS throughout).
//!
//! Usage:
//!     init_schema [/path/to/oanda.duckdb]
//!
//! Defaults to `./data/oanda.duckdb` next to the cwd.

use std::path::PathBuf;

use anyhow::Result;
use persistence::Db;

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let path: PathBuf = argv
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./data/oanda.duckdb"));
    let db = Db::open(&path)?;
    let counts = db.row_counts()?;
    println!("schema applied at {}", path.display());
    let mut keys: Vec<_> = counts.keys().copied().collect();
    keys.sort();
    for k in keys {
        println!("  {k:<22} {:>8}", counts[k]);
    }
    Ok(())
}

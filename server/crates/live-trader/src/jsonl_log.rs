//! Lightweight rolling JSONL logger for the agent-readable
//! `trade_logs/<version>/...` previews. Each record is one line of JSON;
//! files are capped at `MAX_LINES` most-recent records and pruned in-place
//! when over the cap.
//!
//! These files are checked into git (per operator mandate) so an agent
//! can study trading behaviour without spinning up the full stack.
//! Heavy-weight forensics (full per-trade snapshots, ticks, raw bars)
//! stay outside git in `data/trades/` and `data/oanda.duckdb`.
//!
//! Atomic write semantics: append + roll uses tmp file + rename so
//! readers never see a half-written file.

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

/// Hard cap on lines per JSONL preview file. Older lines are dropped on
/// roll-off — the full record still exists in DuckDB / `data/trades/`.
pub const MAX_LINES: usize = 2000;

/// Default repo-relative root for committed previews. Override per call
/// when needed; everything in this module accepts an explicit root for
/// testability.
pub const DEFAULT_ROOT: &str = "trade_logs";

/// The release version that segments the on-disk layout. Pulled from the
/// workspace `version` (`env!("CARGO_PKG_VERSION")` at the call site of
/// the binary that links this crate). Kept as a free function so the
/// caller can pass either the build-time version or an env override.
pub fn version_folder(version: &str) -> String {
    if version.starts_with('v') {
        version.to_string()
    } else {
        format!("v{version}")
    }
}

/// Append `record` (any `Serialize`) as one JSONL line under
/// `<root>/<version>/<sub_path>`. Creates parent dirs as needed. After
/// append, if the file exceeds `MAX_LINES`, rewrite atomically with the
/// most recent `MAX_LINES` lines kept.
///
/// `sub_path` is a slash-joined relative path like `"EUR_USD/trades.jsonl"`
/// or `"training_log.jsonl"`. Forward slashes in instrument names
/// (`"BTC/USD"`) are flattened to `_` so OS path semantics don't create
/// phantom dirs.
pub fn append<T: Serialize>(
    root: &Path,
    version: &str,
    sub_path: &str,
    record: &T,
) -> Result<()> {
    let safe_sub = sub_path.replace("//", "/");
    // Defence-in-depth: forbid `..` segments.
    if safe_sub.split('/').any(|seg| seg == ".." || seg.is_empty()) {
        anyhow::bail!("invalid sub_path {sub_path:?}");
    }
    let folder = version_folder(version);
    let target: PathBuf = root.join(&folder).join(safe_sub);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating jsonl parent {}", parent.display()))?;
    }
    let line = serde_json::to_string(record).context("serialize jsonl record")?;
    {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&target)
            .with_context(|| format!("opening jsonl file {}", target.display()))?;
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
    }
    roll_off_if_needed(&target)
        .with_context(|| format!("rolling off jsonl file {}", target.display()))?;
    Ok(())
}

/// If `path` has more than `MAX_LINES` lines, rewrite it with the most
/// recent `MAX_LINES` lines via tmp file + rename. No-op otherwise.
pub fn roll_off_if_needed(path: &Path) -> Result<()> {
    let f = OpenOptions::new().read(true).open(path)?;
    let reader = BufReader::new(f);
    let mut lines: Vec<String> = reader
        .lines()
        .filter_map(|r| r.ok())
        .filter(|l| !l.is_empty())
        .collect();
    if lines.len() <= MAX_LINES {
        return Ok(());
    }
    let drop_n = lines.len() - MAX_LINES;
    let kept: Vec<String> = lines.split_off(drop_n);
    let tmp = path.with_extension("jsonl.tmp");
    {
        let mut tf = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)?;
        for l in &kept {
            tf.write_all(l.as_bytes())?;
            tf.write_all(b"\n")?;
        }
        tf.flush()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Build the resolved JSONL path for a (root, version, sub_path) triple
/// without writing. Useful for diagnostics + tests.
pub fn resolve(root: &Path, version: &str, sub_path: &str) -> PathBuf {
    root.join(version_folder(version)).join(sub_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn appends_and_creates_dirs() {
        let dir = tempdir().unwrap();
        append(
            dir.path(),
            "1.0.1",
            "EUR_USD/trades.jsonl",
            &json!({"a": 1, "b": "hello"}),
        )
        .unwrap();
        let p = dir.path().join("v1.0.1/EUR_USD/trades.jsonl");
        assert!(p.exists());
        let content = std::fs::read_to_string(&p).unwrap();
        assert!(content.contains("\"a\":1"));
    }

    #[test]
    fn rolls_off_at_max_lines() {
        let dir = tempdir().unwrap();
        for i in 0..(MAX_LINES + 100) {
            append(
                dir.path(),
                "v1.0.1",
                "x.jsonl",
                &json!({"i": i}),
            )
            .unwrap();
        }
        let p = dir.path().join("v1.0.1/x.jsonl");
        let content = std::fs::read_to_string(&p).unwrap();
        let n_lines = content.lines().count();
        assert_eq!(n_lines, MAX_LINES, "expected exactly MAX_LINES after roll");
        // First kept line should be index 100 (we dropped 100 oldest).
        let first: serde_json::Value =
            serde_json::from_str(content.lines().next().unwrap()).unwrap();
        assert_eq!(first["i"], 100);
    }

    #[test]
    fn rejects_traversal_sub_paths() {
        let dir = tempdir().unwrap();
        let r = append(
            dir.path(),
            "v1.0.1",
            "../escape.jsonl",
            &json!({"x":1}),
        );
        assert!(r.is_err());
    }

    #[test]
    fn version_folder_normalizes() {
        assert_eq!(version_folder("1.0.1"), "v1.0.1");
        assert_eq!(version_folder("v1.0.1"), "v1.0.1");
    }
}

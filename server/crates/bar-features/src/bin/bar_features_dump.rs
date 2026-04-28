//! Tiny stdin→stdout CLI: reads a JSON array of `Bar10s` from stdin,
//! computes the 24-dim feature vector for the LAST bar, and writes
//! the result + the FEATURE_NAMES list back as JSON. Exists solely so
//! the Python test suite can verify Rust and Python feature
//! implementations agree byte-for-byte after Phase B re-engineering.
//!
//! Usage (from a test):
//!
//! ```ignore
//! echo '[{"instrument_id":0,"ts_ms":0,"open":1.1,"high":1.1,"low":1.1,"close":1.1,"n_ticks":1,"spread_bp_avg":0.5}, ...]' \
//!   | bar_features_dump
//! ```
//!
//! Output:
//!
//! ```json
//! { "feature_names": ["log_ret_1", ...], "values": [0.0, 0.0, ...] }
//! ```
//!
//! Errors go to stderr with a non-zero exit code so the Python harness
//! can surface them clearly.

use std::io::Read;

use bar_features::{recompute_last, FEATURE_NAMES};
use market_domain::Bar10s;
use serde::Serialize;

#[derive(Serialize)]
struct DumpOutput<'a> {
    feature_names: &'a [&'a str],
    values: Vec<f64>,
}

fn main() {
    let mut buf = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
        eprintln!("bar_features_dump: read stdin: {e}");
        std::process::exit(2);
    }
    let bars: Vec<Bar10s> = match serde_json::from_str(&buf) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("bar_features_dump: parse JSON: {e}");
            std::process::exit(2);
        }
    };
    let values = match recompute_last(&bars) {
        Some(arr) => arr.to_vec(),
        None => {
            eprintln!(
                "bar_features_dump: too few bars ({} < 2)",
                bars.len()
            );
            std::process::exit(3);
        }
    };
    let out = DumpOutput {
        feature_names: FEATURE_NAMES,
        values,
    };
    match serde_json::to_string(&out) {
        Ok(s) => println!("{s}"),
        Err(e) => {
            eprintln!("bar_features_dump: serialize output: {e}");
            std::process::exit(2);
        }
    }
}

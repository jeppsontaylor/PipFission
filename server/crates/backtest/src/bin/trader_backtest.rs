//! `trader_backtest` — CLI entry point for the Python optimiser.
//!
//! Reads a single JSON request from stdin, runs the deterministic
//! backtester, and writes the JSON `Report` to stdout. One process per
//! Optuna trial. Forking is the simplest contract; if optimisation
//! becomes the bottleneck we can switch to a long-lived PyO3 module
//! (M12) without changing the wire format.
//!
//! Request schema:
//! ```json
//! {
//!   "bars":   [{"ts_ms": 0, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "spread_bp_avg": 0.5, "n_ticks": 1}, ...],
//!   "probs":  [{"p_long": 0.7, "p_short": 0.1, "p_take": 0.9, "calibrated": 0.7}, ...],
//!   "sigma":  [0.005, 0.005, ...],
//!   "params": { ... TraderParams ... },
//!   "costs":  { ... Costs ... }
//! }
//! ```
//!
//! The `--print-bounds` flag exits early printing the `TraderParams`
//! search-space bounds as JSON so the Python optimiser can build its
//! Optuna trial space without re-declaring them on the Python side.

use std::io::{self, Read, Write};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use backtest::{run_backtest, Costs, Report};
use market_domain::Bar10s;
use trader::{params::BOUNDS, Probs, TraderParams};

#[derive(Deserialize)]
struct InboundBar {
    ts_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    #[serde(default)]
    n_ticks: u32,
    #[serde(default)]
    spread_bp_avg: f64,
}

impl From<InboundBar> for Bar10s {
    fn from(b: InboundBar) -> Self {
        Bar10s {
            instrument_id: 0,
            ts_ms: b.ts_ms,
            open: b.open,
            high: b.high,
            low: b.low,
            close: b.close,
            n_ticks: b.n_ticks,
            spread_bp_avg: b.spread_bp_avg,
        }
    }
}

#[derive(Deserialize)]
struct Request {
    bars: Vec<InboundBar>,
    probs: Vec<Probs>,
    sigma: Vec<f64>,
    params: TraderParams,
    #[serde(default)]
    costs: Option<Costs>,
}

#[derive(Serialize)]
struct Response {
    report: Report,
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.iter().any(|a| a == "--print-bounds") {
        let json = serde_json::to_string_pretty(BOUNDS)?;
        println!("{json}");
        return Ok(());
    }

    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf).context("read stdin")?;
    let req: Request = serde_json::from_str(&buf).context("parse request JSON")?;
    let bars: Vec<Bar10s> = req.bars.into_iter().map(Into::into).collect();
    let costs = req.costs.unwrap_or_default();
    let report = run_backtest(&bars, &req.probs, &req.sigma, req.params, costs);
    let resp = Response { report };
    let stdout = io::stdout();
    let mut h = stdout.lock();
    serde_json::to_writer(&mut h, &resp)?;
    h.write_all(b"\n")?;
    Ok(())
}

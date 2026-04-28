//! `label_opt` — CLI entry point that wraps the labeling pipeline so the
//! Python research layer (M4) can call into Rust without PyO3 bindings.
//!
//! Reads a JSON request from stdin: `{ bars: [...], cfg: {...} }`. Runs
//! EWMA σ, CUSUM event sampling, triple-barrier labelling, and the
//! constrained label optimiser; writes the chosen `LabelRow`s as JSON
//! to stdout.
//!
//! Same fork-per-call contract as `trader_backtest`. If the Python
//! research layer ends up calling this often enough to matter, we can
//! switch to PyO3 in M12.

use std::io::{self, Read};

use anyhow::{Context, Result};
use serde::Deserialize;

use labeling::{
    cusum_filter, ewma_volatility, optimise_labels, triple_barrier, BarrierConfig, LabelOptimiserConfig, LabelRow,
};
use market_domain::Bar10s;

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
#[serde(default)]
struct Config {
    /// EWMA σ span used for barrier widths and the CUSUM threshold.
    sigma_span: usize,
    /// CUSUM h = sigma * cusum_h_mult. Higher = fewer events.
    cusum_h_mult: f64,
    /// Minimum bar gap between consecutive event picks.
    min_gap: usize,
    /// Triple-barrier config.
    pt_atr: f64,
    sl_atr: f64,
    vert_horizon: usize,
    /// Cost-floor edge in fractional units (after costs). Labels with
    /// |realized_r| < min_edge get meta_y = 0 and may be dropped.
    min_edge: f64,
    /// Optimiser min hold in ms (= bars × 10_000).
    min_hold_ms: i64,
    /// Optimiser max hold in ms.
    max_hold_ms: i64,
    /// Downside-variance penalty weight in the per-trade score.
    downside_lambda: f64,
    /// Per-trade fixed turnover cost.
    turnover_cost: f64,
    /// Minority-side fraction floor for binary balance. Default 0.30 —
    /// see `LabelOptimiserConfig`. Zero disables rebalancing.
    min_minority_frac: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sigma_span: 60,
            cusum_h_mult: 2.0,
            min_gap: 1,
            pt_atr: 2.0,
            sl_atr: 2.0,
            vert_horizon: 36,
            min_edge: 0.0,
            min_hold_ms: 10_000,
            max_hold_ms: 600_000,
            downside_lambda: 4.0,
            turnover_cost: 0.0,
            min_minority_frac: 0.30,
        }
    }
}

#[derive(Deserialize)]
struct Request {
    bars: Vec<InboundBar>,
    #[serde(default)]
    cfg: Config,
}

fn main() -> Result<()> {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf).context("read stdin")?;
    let req: Request = serde_json::from_str(&buf).context("parse request JSON")?;
    let bars: Vec<Bar10s> = req.bars.into_iter().map(Into::into).collect();

    let sigma = ewma_volatility(&bars, req.cfg.sigma_span);
    let events = cusum_filter(&bars, &sigma, req.cfg.cusum_h_mult);

    let bar_cfg = BarrierConfig {
        pt_atr: req.cfg.pt_atr,
        sl_atr: req.cfg.sl_atr,
        vert_horizon: req.cfg.vert_horizon,
        min_edge: req.cfg.min_edge,
    };
    let raw_labels: Vec<LabelRow> = triple_barrier(&bars, &sigma, &events, &bar_cfg);

    let opt_cfg = LabelOptimiserConfig {
        min_hold_ms: req.cfg.min_hold_ms,
        max_hold_ms: req.cfg.max_hold_ms,
        min_edge: req.cfg.min_edge,
        downside_lambda: req.cfg.downside_lambda,
        turnover_cost: req.cfg.turnover_cost,
        min_minority_frac: req.cfg.min_minority_frac,
    };
    let chosen = optimise_labels(&raw_labels, &opt_cfg);

    let stdout = io::stdout();
    let mut h = stdout.lock();
    serde_json::to_writer(
        &mut h,
        &serde_json::json!({
            "n_bars": bars.len(),
            "n_events": events.len(),
            "n_raw_labels": raw_labels.len(),
            "labels": chosen,
        }),
    )?;
    Ok(())
}

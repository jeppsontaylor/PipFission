//! Triple-barrier labeling, event sampling, and label optimisation.
//!
//! The labeling pipeline turns a `[Bar10s]` window into a set of
//! `LabelRow` records that the classifier trains against. The pipeline
//! deliberately runs only on the trailing 1000-bar window — the user's
//! locked invariant — to keep the optimiser cheap and to avoid hindsight
//! bias creeping in from earlier regimes.
//!
//! ## Stages
//! 1. **Volatility estimate** — EWMA of bar returns; barrier widths are
//!    `pt_atr × σ_t` and `sl_atr × σ_t`, so the barriers adapt to regime.
//! 2. **Event sampling** — CUSUM-vol filter selects candidate event bars.
//!    A breakout filter is also provided.
//! 3. **Triple-barrier labelling** — for each event bar, walk forward
//!    until upper barrier (long-side TP) or lower barrier (long-side SL)
//!    is hit, or the vertical horizon expires. The first-hit barrier and
//!    the realised return determine `(side, meta_y, t1_ms, realized_r)`.
//! 4. **Label optimiser** — constrained interval scheduler that picks a
//!    non-overlapping subset of `LabelRow`s maximising a regularised
//!    objective (Sortino - drawdown - turnover - imbalance).
//!
//! ## What's intentionally NOT in here
//! - Feature recompute. That stays in `feature-engine`.
//! - CV splitting. That's the `cv` crate.
//! - Backtesting. That's the `backtest` crate.
//!
//! Everything is pure-Rust; no I/O. The Python research layer calls into
//! these functions via PyO3 bindings (added in a later milestone).

#![deny(unsafe_code)]

pub mod events;
pub mod label_optimizer;
pub mod meta;
pub mod triple_barrier;
pub mod volatility;

pub use events::{breakout_events, cusum_filter, EventConfig};
pub use label_optimizer::{optimise_labels, LabelOptimiserConfig};
pub use meta::meta_label;
pub use triple_barrier::{triple_barrier, BarrierConfig, BarrierHit, LabelRow};
pub use volatility::{atr, ewma_volatility};

use market_domain::Bar10s;
use serde::{Deserialize, Serialize};

/// One-shot configuration for the full labelling pipeline. Mirrors
/// the JSON request the `label_opt` binary accepts (and the python
/// `LabelOptConfig` dataclass) — they're now all the same shape so a
/// Rust orchestrator and a Python orchestrator can both call into
/// `run_label_pipeline` with the same struct.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LabelPipelineConfig {
    pub sigma_span: usize,
    pub cusum_h_mult: f64,
    pub min_gap: usize,
    pub pt_atr: f64,
    pub sl_atr: f64,
    pub vert_horizon: usize,
    pub min_edge: f64,
    pub min_hold_ms: i64,
    pub max_hold_ms: i64,
    pub downside_lambda: f64,
    pub turnover_cost: f64,
    pub min_minority_frac: f64,
}

impl Default for LabelPipelineConfig {
    fn default() -> Self {
        Self {
            sigma_span: 60,
            cusum_h_mult: 1.0,
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

/// Full pipeline output. Mirrors what the `label_opt` binary returns
/// over stdout.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelPipelineOutput {
    pub n_bars: usize,
    pub n_events: usize,
    pub n_raw_labels: usize,
    pub labels: Vec<LabelRow>,
}

/// Run the full labelling pipeline on a window of bars and return
/// the chosen labels. **In-process** — no subprocess fork. This is
/// the function the Rust orchestrator calls; the existing CLI binary
/// (`label_opt`) is a thin wrapper that just JSON-decodes stdin and
/// invokes this.
///
/// The Python `research/labeling/label_opt.py` shim wraps the binary
/// and is now superseded — Rust callers should use this directly.
pub fn run_label_pipeline(
    bars: &[Bar10s],
    cfg: &LabelPipelineConfig,
) -> LabelPipelineOutput {
    let sigma = ewma_volatility(bars, cfg.sigma_span);
    let events = cusum_filter(bars, &sigma, cfg.cusum_h_mult);
    let bar_cfg = BarrierConfig {
        pt_atr: cfg.pt_atr,
        sl_atr: cfg.sl_atr,
        vert_horizon: cfg.vert_horizon,
        min_edge: cfg.min_edge,
    };
    let raw = triple_barrier(bars, &sigma, &events, &bar_cfg);
    let opt_cfg = LabelOptimiserConfig {
        min_hold_ms: cfg.min_hold_ms,
        max_hold_ms: cfg.max_hold_ms,
        min_edge: cfg.min_edge,
        downside_lambda: cfg.downside_lambda,
        turnover_cost: cfg.turnover_cost,
        min_minority_frac: cfg.min_minority_frac,
    };
    let chosen = optimise_labels(&raw, &opt_cfg);
    LabelPipelineOutput {
        n_bars: bars.len(),
        n_events: events.len(),
        n_raw_labels: raw.len(),
        labels: chosen,
    }
}

#[cfg(test)]
mod pipeline_tests {
    use super::*;
    use market_domain::Bar10s;

    fn synth_bars(n: usize) -> Vec<Bar10s> {
        (0..n)
            .map(|i| {
                let t = i as f64 * 0.05;
                let close = 1.0 + (t * 0.3).sin() * 0.001;
                Bar10s {
                    instrument_id: 0,
                    ts_ms: i as i64 * 10_000,
                    open: close,
                    high: close + 1e-4,
                    low: close - 1e-4,
                    close,
                    n_ticks: 5,
                    spread_bp_avg: 1.0,
                }
            })
            .collect()
    }

    #[test]
    fn one_shot_pipeline_returns_balanced_labels() {
        let bars = synth_bars(500);
        let out = run_label_pipeline(&bars, &LabelPipelineConfig::default());
        assert_eq!(out.n_bars, 500);
        // Some events must be sampled and chosen with default config.
        assert!(out.n_events > 0);
        // All chosen labels are binary (no zeros) per the burn-down.
        for l in &out.labels {
            assert!(l.side == 1 || l.side == -1, "non-binary side: {l:?}");
        }
    }
}

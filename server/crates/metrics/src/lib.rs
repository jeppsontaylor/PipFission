//! Trading performance metrics. Identical scoring used by the
//! research/backtest path AND the live engine, so model selection and
//! production reporting can never disagree.
//!
//! All functions are pure: `&[f64]` in, scalar out. The `Summary`
//! struct collects the headline numbers and serialises to JSON for the
//! Rust→Python handoff.

#![deny(unsafe_code)]

pub mod deflated;
pub mod drawdown;
pub mod ratios;

pub use deflated::deflated_sharpe;
pub use drawdown::{equity_drawdown, max_drawdown_bp};
pub use ratios::{calmar, hit_rate, profit_factor, sharpe, sortino};

use serde::{Deserialize, Serialize};

/// Headline metrics summary. Returned by the backtester and persisted to
/// `trader_metrics`. JSON-serialisable for the Python optimiser.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Summary {
    pub n_trades: usize,
    pub net_return: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub deflated_sharpe: f64,
    pub calmar: f64,
    pub max_drawdown_bp: f64,
    pub hit_rate: f64,
    pub profit_factor: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub turnover_per_day: f64,
    /// Total holding time / total elapsed time across the run.
    pub exposure: f64,
    /// Whether the strategy actually traded (n_trades > 0). The optimiser
    /// uses this to enforce a minimum trade-count constraint.
    pub traded: bool,
}

impl Summary {
    /// Build the headline metrics from a per-trade return series and an
    /// equity curve. `n_periods_per_year` annualises Sharpe/Sortino;
    /// for 10s bars on 24/5 forex it's roughly `8640 * 5`.
    pub fn from_trade_returns(
        per_trade_r: &[f64],
        equity: &[f64],
        n_periods_per_year: f64,
        elapsed_days: f64,
    ) -> Self {
        let n_trades = per_trade_r.len();
        let net_return: f64 = per_trade_r.iter().sum();
        let s = sharpe(per_trade_r, n_periods_per_year);
        let so = sortino(per_trade_r, n_periods_per_year);
        let max_dd_bp = max_drawdown_bp(equity);
        let cal = calmar(s, max_dd_bp);
        let (wins, losses): (Vec<f64>, Vec<f64>) =
            per_trade_r.iter().partition(|&&r| r > 0.0);
        let avg_win = if wins.is_empty() {
            0.0
        } else {
            wins.iter().sum::<f64>() / wins.len() as f64
        };
        let avg_loss = if losses.is_empty() {
            0.0
        } else {
            losses.iter().sum::<f64>() / losses.len() as f64
        };
        // Clamp profit_factor to a large finite value: serde_json emits
        // `null` for non-finite floats, which crashes downstream Python
        // consumers that expect a number. Anything beyond 1e6 is "all
        // wins, no losses" — the ranking is preserved by the cap.
        let pf_raw = profit_factor(per_trade_r);
        let pf = if pf_raw.is_finite() {
            pf_raw
        } else if pf_raw > 0.0 {
            1.0e6
        } else {
            0.0
        };
        let hr = hit_rate(per_trade_r);
        let turnover = if elapsed_days > 0.0 {
            n_trades as f64 / elapsed_days
        } else {
            0.0
        };
        // Conservative DSR with skew=0, kurt=3, n_trials=1 (caller can
        // override via `deflated_sharpe` directly when n_trials > 1).
        let dsr = deflated_sharpe(s, per_trade_r.len().max(1), 1, 0.0, 3.0);
        Self {
            n_trades,
            net_return,
            sharpe: s,
            sortino: so,
            deflated_sharpe: dsr,
            calmar: cal,
            max_drawdown_bp: max_dd_bp,
            hit_rate: hr,
            profit_factor: pf,
            avg_win,
            avg_loss,
            turnover_per_day: turnover,
            exposure: 0.0,
            traded: n_trades > 0,
        }
    }
}

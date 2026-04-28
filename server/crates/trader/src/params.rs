//! `TraderParams` — the thing the optimiser tunes — plus the `Probs`
//! input row the trader consumes per bar.
//!
//! Bounds on each field are exported as a static `BOUNDS` table so the
//! Python optimiser can build its search space directly from this
//! crate (read via the `trader_backtest` binary's `--print-bounds`
//! flag — see `backtest::bin`).

use serde::{Deserialize, Serialize};

/// Per-bar input to the trader: the calibrated probabilities for
/// long / short / take-trade plus the raw "side" probability spread
/// the optimiser thresholds against.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Probs {
    pub p_long: f64,
    pub p_short: f64,
    pub p_take: f64,
    /// Calibrated probability of the chosen side (max(p_long, p_short)
    /// after Platt/sigmoid calibration). Used by the trader's
    /// `min_conf_margin` rule.
    pub calibrated: f64,
}

/// Tunable trader parameters. Every field has a concrete bound so the
/// optimiser can sample legal points. Defaults pick a "sensible no-op"
/// configuration that won't trade on noise.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TraderParams {
    /// p_long must clear this to consider a long entry.
    pub long_threshold: f64,
    /// p_short must clear this to consider a short entry.
    pub short_threshold: f64,
    /// p_take must clear this for the meta-model to allow a trade.
    pub take_threshold: f64,
    /// |p_long − p_short| must clear this to enter.
    pub min_conf_margin: f64,
    /// Stop-loss in ATR multiples (per-bar volatility units).
    pub stop_loss_atr: f64,
    /// Take-profit in ATR multiples.
    pub take_profit_atr: f64,
    /// Trailing-stop ATR multiple. 0 disables trailing.
    pub trailing_stop_atr: f64,
    /// Minimum holding period in bars before any exit (other than stop).
    pub min_hold_bars: u32,
    /// Maximum holding period in bars; force-exit at this horizon.
    pub max_hold_bars: u32,
    /// Bars to wait before re-entering after any exit.
    pub cooldown_bars: u32,
    /// Maximum fraction of account equity per position.
    pub max_position_frac: f64,
    /// Daily loss kill-switch in basis points.
    pub daily_loss_limit_bp: f64,
    /// Pause new entries once peak-to-trough drawdown exceeds this in bp.
    pub max_dd_pause_bp: f64,
    /// Skip trades when the bar's spread exceeds this in bp.
    pub spread_max_bp: f64,
    /// Skip trades when the latest tick is older than this in ms.
    pub stale_data_ms: i64,
}

impl Default for TraderParams {
    /// Phase C "hold longer + higher conviction" defaults. The system
    /// previously over-traded on noisy classifier flips at min_hold=3
    /// (30 seconds on 10s bars). These defaults push the trader into
    /// 2-30 minute swing-trade behaviour:
    ///
    ///  * `min_hold_bars=12` — must hold 2 minutes before any
    ///    Reverse exit. Damps noise-driven flips.
    ///  * `max_hold_bars=180` — caps holds at 30 minutes; lets winning
    ///    trades breathe.
    ///  * `min_conf_margin=0.15` — requires a 15% gap between p_long
    ///    and p_short to enter. Stricter long-vs-short discrimination.
    ///  * `long/short_threshold=0.62` — higher conviction entries.
    ///  * `cooldown_bars=12` — 2 min wind-down after exit so the next
    ///    entry isn't a knee-jerk reaction to the closing bar.
    ///  * `take_profit_atr=2.5` — wider profit target to give winners
    ///    room. Stop-loss stays at 1.5×ATR (asymmetric reward:risk).
    ///
    /// Each field is also pinned by `BOUNDS` below so the optimizer
    /// can't undo these structural choices.
    fn default() -> Self {
        Self {
            long_threshold: 0.62,
            short_threshold: 0.62,
            take_threshold: 0.50,
            min_conf_margin: 0.15,
            stop_loss_atr: 1.5,
            take_profit_atr: 2.5,
            trailing_stop_atr: 0.0,
            min_hold_bars: 12,
            max_hold_bars: 180,
            cooldown_bars: 12,
            max_position_frac: 0.10,
            daily_loss_limit_bp: 200.0,
            max_dd_pause_bp: 500.0,
            spread_max_bp: 5.0,
            stale_data_ms: 30_000,
        }
    }
}

/// Inclusive lower/upper bounds for every numeric field. Fed to the
/// Optuna search space.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Bound {
    pub name: &'static str,
    pub lo: f64,
    pub hi: f64,
    pub is_int: bool,
}

// Phase C "no-scalp guarantee" bounds. The optimiser used to converge
// to scalp-style params (`min_hold=1, max_hold=12`) because tight
// holds minimise drawdown in-sample, even though they generate
// constant noise-driven Reverse exits in production. Floors below
// make scalp configurations unrepresentable in the search space:
//
//   * `min_hold_bars` ≥ 10  — every trade holds at least ~100 seconds.
//   * `max_hold_bars` ≥ 60  — the optimiser can't pick a 5-bar ceiling.
//   * `min_conf_margin` ≥ 0.05 — entries always require at least a
//     5% probability gap, even when the optimiser would prefer
//     looser entry discipline.
//   * `cooldown_bars` ≥ 6   — there's always a cooldown after an exit.
pub const BOUNDS: &[Bound] = &[
    Bound { name: "long_threshold", lo: 0.50, hi: 0.95, is_int: false },
    Bound { name: "short_threshold", lo: 0.50, hi: 0.95, is_int: false },
    Bound { name: "take_threshold", lo: 0.40, hi: 0.95, is_int: false },
    Bound { name: "min_conf_margin", lo: 0.05, hi: 0.40, is_int: false },
    Bound { name: "stop_loss_atr", lo: 0.30, hi: 4.00, is_int: false },
    Bound { name: "take_profit_atr", lo: 0.30, hi: 6.00, is_int: false },
    Bound { name: "trailing_stop_atr", lo: 0.00, hi: 5.00, is_int: false },
    Bound { name: "min_hold_bars", lo: 10.0, hi: 60.0, is_int: true },
    Bound { name: "max_hold_bars", lo: 60.0, hi: 360.0, is_int: true },
    Bound { name: "cooldown_bars", lo: 6.0, hi: 60.0, is_int: true },
    Bound { name: "max_position_frac", lo: 0.01, hi: 1.00, is_int: false },
    Bound { name: "daily_loss_limit_bp", lo: 25.0, hi: 1500.0, is_int: false },
    Bound { name: "max_dd_pause_bp", lo: 50.0, hi: 5000.0, is_int: false },
    Bound { name: "spread_max_bp", lo: 0.5, hi: 30.0, is_int: false },
    Bound { name: "stale_data_ms", lo: 1000.0, hi: 120_000.0, is_int: true },
];

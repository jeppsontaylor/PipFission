//! Risk gates. Pre-trade and per-bar checks that veto trades regardless
//! of what the model suggests. The state machine consults these on every
//! new bar; gates can be triggered open / closed independently of the
//! model's signal.

use crate::params::TraderParams;
use crate::state::Reason;

/// Result of a risk-gate check on a single bar.
#[derive(Clone, Copy, Debug)]
pub enum RiskOutcome {
    /// Trading allowed.
    Ok,
    /// Trading blocked. The reason is surfaced in `TradeEvent::Skip`.
    Blocked(Reason),
}

/// Bar-aware risk state carried across calls. Tracks running daily P&L,
/// running drawdown vs the day's high-water mark, and consecutive-loss
/// streak so the kill-switches can fire deterministically.
#[derive(Clone, Copy, Debug, Default)]
pub struct RiskGates {
    /// Cumulative realised P&L for the current day in basis points.
    pub day_pnl_bp: f64,
    /// High-water mark of `day_pnl_bp` so far.
    pub day_peak_bp: f64,
    /// Consecutive loss streak (resets on any winning trade).
    pub consec_losses: u32,
    /// Wall-clock day-of-year boundary for resetting daily counters.
    /// Stored as the epoch-day integer; rolling over resets day_pnl/peak.
    pub epoch_day: i64,
}

impl RiskGates {
    /// Roll daily counters if we've crossed midnight UTC.
    pub fn maybe_roll_day(&mut self, ts_ms: i64) {
        let day = ts_ms.div_euclid(86_400_000);
        if day != self.epoch_day {
            self.day_pnl_bp = 0.0;
            self.day_peak_bp = 0.0;
            self.consec_losses = 0;
            self.epoch_day = day;
        }
    }

    /// Update running P&L after a closed trade. `realized_r` is the net
    /// per-trade return; `1.0` is treated as 100% so we convert to bp.
    pub fn record_trade(&mut self, realized_r: f64) {
        let pnl_bp = realized_r * 10_000.0;
        self.day_pnl_bp += pnl_bp;
        if self.day_pnl_bp > self.day_peak_bp {
            self.day_peak_bp = self.day_pnl_bp;
        }
        if realized_r < 0.0 {
            self.consec_losses += 1;
        } else if realized_r > 0.0 {
            self.consec_losses = 0;
        }
    }

    /// Pre-trade gate. Returns `Blocked(reason)` if any kill-switch fires
    /// at this bar; otherwise `Ok`.
    pub fn check(
        &self,
        params: &TraderParams,
        bar_ts_ms: i64,
        spread_bp: f64,
        latest_tick_age_ms: i64,
    ) -> RiskOutcome {
        if spread_bp > params.spread_max_bp {
            return RiskOutcome::Blocked(Reason::SpreadTooWide);
        }
        if latest_tick_age_ms > params.stale_data_ms {
            return RiskOutcome::Blocked(Reason::StaleData);
        }
        if self.day_pnl_bp <= -params.daily_loss_limit_bp {
            return RiskOutcome::Blocked(Reason::DailyLossKill);
        }
        let dd_bp = (self.day_peak_bp - self.day_pnl_bp).max(0.0);
        if dd_bp >= params.max_dd_pause_bp {
            return RiskOutcome::Blocked(Reason::DrawdownPause);
        }
        let _ = bar_ts_ms;
        RiskOutcome::Ok
    }
}

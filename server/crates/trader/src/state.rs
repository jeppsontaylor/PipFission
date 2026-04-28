//! State machine. Drives the trader through Flat → Long/Short → Flat
//! transitions according to `TraderParams`, the latest `Probs`, and the
//! `RiskGates`. Emits zero or one `TradeEvent` per bar.

use serde::{Deserialize, Serialize};

use market_domain::Bar10s;

use crate::params::{Probs, TraderParams};
use crate::risk::{RiskGates, RiskOutcome};

/// Position side.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Side {
    Long,
    Short,
}

/// FSM state. `Cooldown(n)` tracks the remaining bars before re-entry
/// is allowed. `Long(entry)` / `Short(entry)` carry the bar index at
/// entry so min/max-hold checks are O(1).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum State {
    Flat,
    Long { entry_idx: u32 },
    Short { entry_idx: u32 },
    Cooldown { remaining: u32 },
}

impl Default for State {
    fn default() -> Self {
        State::Flat
    }
}

/// Reason a transition occurred. Drives dashboard tooltips and explains
/// "why didn't we trade" in the optimiser logs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Reason {
    /// Probability + threshold rules cleared.
    Signal,
    /// Stop-loss barrier hit.
    StopLoss,
    /// Take-profit barrier hit.
    TakeProfit,
    /// Trailing-stop barrier hit.
    TrailingStop,
    /// Max hold reached.
    MaxHold,
    /// Reverse signal triggered an early exit.
    Reverse,
    /// Risk gate blocked entry: spread too wide.
    SpreadTooWide,
    /// Risk gate blocked entry: stale market data.
    StaleData,
    /// Risk gate: daily loss kill-switch fired.
    DailyLossKill,
    /// Risk gate: drawdown-pause active.
    DrawdownPause,
    /// In cooldown after a recent exit.
    Cooldown,
    /// Probability rules failed to clear thresholds.
    BelowThreshold,
    /// Below the minimum holding period; can't exit voluntarily yet.
    MinHold,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum TradeEvent {
    /// Position opened at this bar's close.
    Open {
        side: Side,
        bar_idx: u32,
        entry_px: f64,
        reason: Reason,
    },
    /// Position closed at this bar's close.
    Close {
        bar_idx: u32,
        exit_px: f64,
        realized_r: f64,
        reason: Reason,
    },
    /// Trader chose not to trade this bar; reason recorded for later
    /// auditing. Skipped events DON'T flow into trade_ledger.
    Skip { bar_idx: u32, reason: Reason },
}

/// Trader instance. One per (instrument); spawn many in parallel for a
/// multi-instrument trader pool.
#[derive(Clone, Debug)]
pub struct Trader {
    pub params: TraderParams,
    pub state: State,
    pub risk: RiskGates,
    /// Entry close + barrier widths captured at open. Re-evaluated at
    /// every bar in `Long`/`Short`.
    open: Option<OpenContext>,
}

#[derive(Clone, Copy, Debug)]
struct OpenContext {
    entry_px: f64,
    sigma: f64,
    /// Best price seen since entry (for trailing stop).
    extreme_px: f64,
}

impl Trader {
    pub fn new(params: TraderParams) -> Self {
        Self {
            params,
            state: State::Flat,
            risk: RiskGates::default(),
            open: None,
        }
    }

    /// Process one closed bar. `sigma` is the per-bar EWMA volatility
    /// used for ATR-style barrier widths. `latest_tick_age_ms` is the
    /// gap between the bar's close and the latest tick (for staleness
    /// gating in the live path; backtester passes 0).
    pub fn on_bar(
        &mut self,
        bar_idx: u32,
        bar: &Bar10s,
        probs: &Probs,
        sigma: f64,
        latest_tick_age_ms: i64,
    ) -> TradeEvent {
        self.risk.maybe_roll_day(bar.ts_ms);
        match self.state {
            State::Cooldown { remaining } => {
                let next = remaining.saturating_sub(1);
                self.state = if next == 0 {
                    State::Flat
                } else {
                    State::Cooldown { remaining: next }
                };
                TradeEvent::Skip {
                    bar_idx,
                    reason: Reason::Cooldown,
                }
            }
            State::Flat => {
                match self
                    .risk
                    .check(&self.params, bar.ts_ms, bar.spread_bp_avg, latest_tick_age_ms)
                {
                    RiskOutcome::Blocked(r) => TradeEvent::Skip { bar_idx, reason: r },
                    RiskOutcome::Ok => self.maybe_enter(bar_idx, bar, probs, sigma),
                }
            }
            State::Long { entry_idx } => self.maybe_exit_long(bar_idx, bar, probs, sigma, entry_idx),
            State::Short { entry_idx } => {
                self.maybe_exit_short(bar_idx, bar, probs, sigma, entry_idx)
            }
        }
    }

    fn maybe_enter(
        &mut self,
        bar_idx: u32,
        bar: &Bar10s,
        probs: &Probs,
        sigma: f64,
    ) -> TradeEvent {
        let p = &self.params;
        if probs.p_take < p.take_threshold {
            return TradeEvent::Skip {
                bar_idx,
                reason: Reason::BelowThreshold,
            };
        }
        let margin = (probs.p_long - probs.p_short).abs();
        if margin < p.min_conf_margin {
            return TradeEvent::Skip {
                bar_idx,
                reason: Reason::BelowThreshold,
            };
        }
        let go_long = probs.p_long >= p.long_threshold && probs.p_long > probs.p_short;
        let go_short = probs.p_short >= p.short_threshold && probs.p_short > probs.p_long;
        if go_long {
            self.state = State::Long { entry_idx: bar_idx };
            self.open = Some(OpenContext {
                entry_px: bar.close,
                sigma: sigma.max(1e-12),
                extreme_px: bar.close,
            });
            return TradeEvent::Open {
                side: Side::Long,
                bar_idx,
                entry_px: bar.close,
                reason: Reason::Signal,
            };
        }
        if go_short {
            self.state = State::Short { entry_idx: bar_idx };
            self.open = Some(OpenContext {
                entry_px: bar.close,
                sigma: sigma.max(1e-12),
                extreme_px: bar.close,
            });
            return TradeEvent::Open {
                side: Side::Short,
                bar_idx,
                entry_px: bar.close,
                reason: Reason::Signal,
            };
        }
        TradeEvent::Skip {
            bar_idx,
            reason: Reason::BelowThreshold,
        }
    }

    fn maybe_exit_long(
        &mut self,
        bar_idx: u32,
        bar: &Bar10s,
        probs: &Probs,
        _sigma: f64,
        entry_idx: u32,
    ) -> TradeEvent {
        let p = self.params;
        let ctx = match self.open {
            Some(c) => c,
            None => {
                self.state = State::Flat;
                return TradeEvent::Skip {
                    bar_idx,
                    reason: Reason::BelowThreshold,
                };
            }
        };
        let mut new_extreme = ctx.extreme_px.max(bar.high);
        let bars_held = bar_idx.saturating_sub(entry_idx);
        let stop_px = ctx.entry_px * (1.0 - p.stop_loss_atr * ctx.sigma);
        let tp_px = ctx.entry_px * (1.0 + p.take_profit_atr * ctx.sigma);
        let trailing_px = if p.trailing_stop_atr > 0.0 {
            new_extreme * (1.0 - p.trailing_stop_atr * ctx.sigma)
        } else {
            f64::NEG_INFINITY
        };
        // Adverse fill assumption: SL takes precedence when both bands
        // are touched within the same bar.
        if bar.low <= stop_px {
            return self.close_long(bar_idx, stop_px, ctx, Reason::StopLoss);
        }
        if bar.high >= tp_px {
            return self.close_long(bar_idx, tp_px, ctx, Reason::TakeProfit);
        }
        if bar.low <= trailing_px && p.trailing_stop_atr > 0.0 {
            return self.close_long(bar_idx, trailing_px, ctx, Reason::TrailingStop);
        }
        if bars_held >= p.max_hold_bars {
            return self.close_long(bar_idx, bar.close, ctx, Reason::MaxHold);
        }
        if bars_held >= p.min_hold_bars && probs.p_short >= p.short_threshold {
            return self.close_long(bar_idx, bar.close, ctx, Reason::Reverse);
        }
        // Stay long; persist the running extreme.
        self.open = Some(OpenContext {
            extreme_px: new_extreme.max(bar.close),
            ..ctx
        });
        let _ = &mut new_extreme;
        TradeEvent::Skip {
            bar_idx,
            reason: Reason::MinHold,
        }
    }

    fn maybe_exit_short(
        &mut self,
        bar_idx: u32,
        bar: &Bar10s,
        probs: &Probs,
        _sigma: f64,
        entry_idx: u32,
    ) -> TradeEvent {
        let p = self.params;
        let ctx = match self.open {
            Some(c) => c,
            None => {
                self.state = State::Flat;
                return TradeEvent::Skip {
                    bar_idx,
                    reason: Reason::BelowThreshold,
                };
            }
        };
        let new_extreme = ctx.extreme_px.min(bar.low);
        let bars_held = bar_idx.saturating_sub(entry_idx);
        let stop_px = ctx.entry_px * (1.0 + p.stop_loss_atr * ctx.sigma);
        let tp_px = ctx.entry_px * (1.0 - p.take_profit_atr * ctx.sigma);
        let trailing_px = if p.trailing_stop_atr > 0.0 {
            new_extreme * (1.0 + p.trailing_stop_atr * ctx.sigma)
        } else {
            f64::INFINITY
        };
        if bar.high >= stop_px {
            return self.close_short(bar_idx, stop_px, ctx, Reason::StopLoss);
        }
        if bar.low <= tp_px {
            return self.close_short(bar_idx, tp_px, ctx, Reason::TakeProfit);
        }
        if bar.high >= trailing_px && p.trailing_stop_atr > 0.0 {
            return self.close_short(bar_idx, trailing_px, ctx, Reason::TrailingStop);
        }
        if bars_held >= p.max_hold_bars {
            return self.close_short(bar_idx, bar.close, ctx, Reason::MaxHold);
        }
        if bars_held >= p.min_hold_bars && probs.p_long >= p.long_threshold {
            return self.close_short(bar_idx, bar.close, ctx, Reason::Reverse);
        }
        self.open = Some(OpenContext {
            extreme_px: new_extreme.min(bar.close),
            ..ctx
        });
        TradeEvent::Skip {
            bar_idx,
            reason: Reason::MinHold,
        }
    }

    fn close_long(
        &mut self,
        bar_idx: u32,
        exit_px: f64,
        ctx: OpenContext,
        reason: Reason,
    ) -> TradeEvent {
        let realized_r = exit_px / ctx.entry_px - 1.0;
        self.risk.record_trade(realized_r);
        self.state = State::Cooldown {
            remaining: self.params.cooldown_bars,
        };
        self.open = None;
        TradeEvent::Close {
            bar_idx,
            exit_px,
            realized_r,
            reason,
        }
    }

    fn close_short(
        &mut self,
        bar_idx: u32,
        exit_px: f64,
        ctx: OpenContext,
        reason: Reason,
    ) -> TradeEvent {
        let realized_r = ctx.entry_px / exit_px - 1.0;
        self.risk.record_trade(realized_r);
        self.state = State::Cooldown {
            remaining: self.params.cooldown_bars,
        };
        self.open = None;
        TradeEvent::Close {
            bar_idx,
            exit_px,
            realized_r,
            reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(idx: i64, c: f64, h: f64, l: f64) -> Bar10s {
        Bar10s {
            instrument_id: 0,
            ts_ms: idx * 10_000,
            open: c,
            high: h,
            low: l,
            close: c,
            n_ticks: 1,
            spread_bp_avg: 0.5,
        }
    }

    #[test]
    fn enters_long_when_thresholds_clear() {
        let mut t = Trader::new(TraderParams::default());
        let probs = Probs {
            p_long: 0.8,
            p_short: 0.1,
            p_take: 0.9,
            calibrated: 0.8,
        };
        let ev = t.on_bar(0, &bar(0, 100.0, 100.5, 99.5), &probs, 0.005, 0);
        match ev {
            TradeEvent::Open { side: Side::Long, .. } => {}
            _ => panic!("expected long open, got {ev:?}"),
        }
        assert!(matches!(t.state, State::Long { .. }));
    }

    #[test]
    fn skip_below_take_threshold() {
        let mut t = Trader::new(TraderParams::default());
        let probs = Probs {
            p_long: 0.99,
            p_short: 0.0,
            p_take: 0.01,
            calibrated: 0.99,
        };
        let ev = t.on_bar(0, &bar(0, 100.0, 100.5, 99.5), &probs, 0.005, 0);
        assert!(matches!(
            ev,
            TradeEvent::Skip {
                reason: Reason::BelowThreshold,
                ..
            }
        ));
        assert!(matches!(t.state, State::Flat));
    }

    #[test]
    fn cooldown_blocks_re_entry_after_close() {
        let mut params = TraderParams::default();
        params.cooldown_bars = 3;
        params.min_hold_bars = 0;
        params.stop_loss_atr = 0.0001; // immediate stop
        let mut t = Trader::new(params);
        let probs = Probs {
            p_long: 0.9,
            p_short: 0.0,
            p_take: 0.99,
            calibrated: 0.9,
        };
        let _ = t.on_bar(0, &bar(0, 100.0, 100.5, 99.5), &probs, 0.005, 0);
        // Next bar with low < stop → close.
        let ev = t.on_bar(1, &bar(1, 99.0, 99.5, 98.5), &probs, 0.005, 0);
        assert!(matches!(
            ev,
            TradeEvent::Close {
                reason: Reason::StopLoss,
                ..
            }
        ));
        // We're in Cooldown; further bars must Skip.
        let ev = t.on_bar(2, &bar(2, 100.0, 100.5, 99.5), &probs, 0.005, 0);
        assert!(matches!(ev, TradeEvent::Skip { .. }));
    }
}

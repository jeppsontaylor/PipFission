//! Triple-barrier labelling (López de Prado §3.4).
//!
//! For each event bar `i`, define three barriers:
//!   - **Upper**  at `entry_close · (1 + pt_atr · σ_i)`  — long-side TP / short-side SL.
//!   - **Lower**  at `entry_close · (1 - sl_atr · σ_i)`  — long-side SL / short-side TP.
//!   - **Vertical** at bar `i + vert_horizon` (or end-of-window).
//!
//! Walk forward from `i+1`. The first bar whose [low, high] range
//! contains an upper-barrier hit before any lower-barrier hit produces a
//! `+1` (long-favourable) label; the first lower-barrier hit produces
//! `-1`. If neither barrier is touched before the vertical horizon, the
//! label is the **sign of the close-to-close return at the vertical**.
//!
//! **Binary mode (post-burn-down)**: `side` is always `±1`, never `0`.
//! On a vertical hit with exactly zero realised return (rare; occurs only
//! when entry close == horizon close), the side is bucketed as `+1` by
//! convention. The downstream classifier is binary IN/OUT (long vs short),
//! so a `0` would be unrepresentable anyway.
//!
//! `meta_y` is `1` whenever `realized_r` exceeds a configurable
//! `min_edge` (after costs). Callers pass net costs through `BarrierConfig::min_edge`.
//!
//! When both barriers are crossed by the same bar (intra-bar both-touch),
//! we adopt the *adverse* assumption — the loss barrier is taken — to
//! match the backtester's adverse-fill rule.

use serde::{Deserialize, Serialize};

use market_domain::Bar10s;

#[derive(Clone, Copy, Debug)]
pub struct BarrierConfig {
    /// Profit-take multiplier on per-bar volatility.
    pub pt_atr: f64,
    /// Stop-loss multiplier on per-bar volatility.
    pub sl_atr: f64,
    /// Maximum forward bars before the vertical barrier closes the trade.
    pub vert_horizon: usize,
    /// Net edge a trade must clear, in absolute return units, to count as
    /// `meta_y = 1`. Captures expected costs (commission + spread + slip).
    pub min_edge: f64,
}

impl Default for BarrierConfig {
    fn default() -> Self {
        Self {
            pt_atr: 2.0,
            sl_atr: 2.0,
            vert_horizon: 36, // 6 minutes at 10s bars
            min_edge: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BarrierHit {
    /// Profit-take barrier (long entry's upper / short entry's lower).
    Pt,
    /// Stop-loss barrier (long entry's lower / short entry's upper).
    Sl,
    /// Vertical (time) barrier — exit at horizon.
    Vert,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LabelRow {
    /// Event-bar timestamp.
    pub ts_ms: i64,
    /// Vertical-barrier (or first-hit barrier) timestamp. Always > ts_ms.
    pub t1_ms: i64,
    /// Side of the most-favourable trade at this event: -1 = short
    /// favourable, +1 = long favourable. Binary classifier — `0` is
    /// reserved for legacy rows on the wire and never produced by new
    /// labelling runs.
    pub side: i8,
    /// `1` if `|realized_r| >= min_edge`, else `0`.
    pub meta_y: u8,
    /// Realised return in price-relative units.
    pub realized_r: f64,
    /// Which barrier closed the trade.
    pub barrier_hit: BarrierHit,
}

/// Apply triple-barrier labelling to `events_idx` over `bars`. `sigma`
/// must be the same length as `bars` (typically EWMA σ from the
/// `volatility` module). Returns one `LabelRow` per event index.
pub fn triple_barrier(
    bars: &[Bar10s],
    sigma: &[f64],
    events_idx: &[usize],
    cfg: &BarrierConfig,
) -> Vec<LabelRow> {
    debug_assert_eq!(bars.len(), sigma.len());
    let mut out = Vec::with_capacity(events_idx.len());
    if bars.is_empty() || cfg.vert_horizon == 0 {
        return out;
    }
    for &i in events_idx {
        if i + 1 >= bars.len() {
            continue;
        }
        let entry = bars[i].close;
        let s = sigma[i].max(1e-12);
        let upper = entry * (1.0 + cfg.pt_atr * s);
        let lower = entry * (1.0 - cfg.sl_atr * s);
        let last_idx = (i + cfg.vert_horizon).min(bars.len() - 1);

        let mut hit_kind = BarrierHit::Vert;
        let mut hit_ts = bars[last_idx].ts_ms;
        let mut hit_close = bars[last_idx].close;
        let mut side: i8 = 0;

        for j in (i + 1)..=last_idx {
            let high = bars[j].high;
            let low = bars[j].low;
            let upper_touched = high >= upper;
            let lower_touched = low <= lower;
            if upper_touched && lower_touched {
                // Intra-bar both touched → adverse assumption: stop loss
                // takes precedence (long-side SL = short-side TP).
                hit_kind = BarrierHit::Sl;
                hit_ts = bars[j].ts_ms;
                hit_close = lower;
                side = -1;
                break;
            }
            if upper_touched {
                hit_kind = BarrierHit::Pt;
                hit_ts = bars[j].ts_ms;
                hit_close = upper;
                side = 1;
                break;
            }
            if lower_touched {
                hit_kind = BarrierHit::Sl;
                hit_ts = bars[j].ts_ms;
                hit_close = lower;
                side = -1;
                break;
            }
        }

        if hit_kind == BarrierHit::Vert {
            // Vertical hit: binary side = sign of realised return. A truly
            // zero return is bucketed as long (`+1`); the classifier is
            // binary so there is no neutral class to fall into.
            let r = hit_close - entry;
            side = if r >= 0.0 { 1 } else { -1 };
        }

        let realized_r = match hit_kind {
            BarrierHit::Pt => upper / entry - 1.0,
            BarrierHit::Sl => lower / entry - 1.0,
            BarrierHit::Vert => hit_close / entry - 1.0,
        };

        let meta_y = if realized_r.abs() >= cfg.min_edge { 1 } else { 0 };

        out.push(LabelRow {
            ts_ms: bars[i].ts_ms,
            t1_ms: hit_ts,
            side,
            meta_y,
            realized_r,
            barrier_hit: hit_kind,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_bar(ts: i64, c: f64, h: f64, l: f64) -> Bar10s {
        Bar10s {
            instrument_id: 0,
            ts_ms: ts,
            open: c,
            high: h,
            low: l,
            close: c,
            n_ticks: 1,
            spread_bp_avg: 0.0,
        }
    }

    #[test]
    fn t1_strictly_greater_than_t0() {
        let bars: Vec<_> = (0..50)
            .map(|i| mk_bar(i * 10_000, 1.0 + (i as f64) * 0.001, 1.0 + (i as f64) * 0.001 + 0.001, 1.0 + (i as f64) * 0.001 - 0.001))
            .collect();
        let sigma = vec![0.005; bars.len()];
        let events = vec![5, 10, 20, 30];
        let cfg = BarrierConfig::default();
        let labels = triple_barrier(&bars, &sigma, &events, &cfg);
        for label in &labels {
            assert!(
                label.t1_ms > label.ts_ms,
                "t1 must be strictly greater than t0: {label:?}"
            );
        }
    }

    #[test]
    fn pt_barrier_is_recognised() {
        // Construct a series where the close climbs steadily so the PT
        // barrier from event index 5 with sigma=0.01 and pt_atr=1 is
        // touched within the horizon.
        let mut bars = Vec::new();
        for i in 0..50 {
            let c = 1.0 + (i as f64) * 0.005;
            bars.push(mk_bar(i * 10_000, c, c + 0.001, c - 0.001));
        }
        let sigma = vec![0.01; bars.len()];
        let events = vec![5];
        let cfg = BarrierConfig {
            pt_atr: 1.0,
            sl_atr: 5.0,
            vert_horizon: 20,
            min_edge: 0.0,
        };
        let labels = triple_barrier(&bars, &sigma, &events, &cfg);
        assert_eq!(labels.len(), 1);
        let l = labels[0];
        assert_eq!(l.barrier_hit, BarrierHit::Pt);
        assert_eq!(l.side, 1);
        assert!(l.realized_r > 0.0);
    }

    #[test]
    fn sl_barrier_is_recognised() {
        // Series falling steadily — SL on a long-direction triple-barrier
        // should be hit.
        let mut bars = Vec::new();
        for i in 0..50 {
            let c = 1.0 - (i as f64) * 0.005;
            bars.push(mk_bar(i * 10_000, c, c + 0.001, c - 0.001));
        }
        let sigma = vec![0.01; bars.len()];
        let events = vec![5];
        let cfg = BarrierConfig {
            pt_atr: 5.0,
            sl_atr: 1.0,
            vert_horizon: 20,
            min_edge: 0.0,
        };
        let labels = triple_barrier(&bars, &sigma, &events, &cfg);
        assert_eq!(labels.len(), 1);
        let l = labels[0];
        assert_eq!(l.barrier_hit, BarrierHit::Sl);
        assert_eq!(l.side, -1);
        assert!(l.realized_r < 0.0);
    }
}

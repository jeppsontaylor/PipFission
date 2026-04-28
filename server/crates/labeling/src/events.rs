//! Candidate event sampling. The classifier learns over a *subset* of
//! bars chosen by these filters — this keeps the training set focused
//! on bars where something economically meaningful happened, and keeps
//! the long/short label balance close to 50/50.
//!
//! Two filters are provided:
//!   - **CUSUM** (López de Prado §2.5.2). Tracks cumulative log-returns
//!     since the last reset; fires whenever |cum| crosses a per-bar
//!     volatility-scaled threshold. This is the default.
//!   - **Breakout**. Fires when the close exits a Donchian channel of
//!     the last `lookback` bars. Cheap and complementary.

use market_domain::Bar10s;

/// Tunable knobs for the event sampler. `cusum_h_mult` is multiplied by
/// the per-bar EWMA σ to set the CUSUM threshold; `breakout_lookback` is
/// the Donchian window. Either filter is optional.
#[derive(Clone, Copy, Debug)]
pub struct EventConfig {
    pub cusum_enabled: bool,
    pub cusum_h_mult: f64,
    pub breakout_enabled: bool,
    pub breakout_lookback: usize,
    /// Minimum gap (in bars) between consecutive events. Prevents the
    /// optimiser seeing nearly-identical adjacent events and inflating
    /// label count. Default: 1 (no gap).
    pub min_gap: usize,
}

impl Default for EventConfig {
    fn default() -> Self {
        Self {
            cusum_enabled: true,
            cusum_h_mult: 2.0,
            breakout_enabled: false,
            breakout_lookback: 20,
            min_gap: 1,
        }
    }
}

/// Symmetric CUSUM filter on close-to-close log returns. `sigma` must be
/// the same length as `bars` and is typically `volatility::ewma_volatility`.
/// Returns the indices of bars at which the filter fires (resets `s_pos`
/// or `s_neg` to zero on each fire).
pub fn cusum_filter(bars: &[Bar10s], sigma: &[f64], h_mult: f64) -> Vec<usize> {
    debug_assert_eq!(bars.len(), sigma.len());
    let mut out = Vec::new();
    if bars.len() < 2 {
        return out;
    }
    let mut s_pos = 0.0_f64;
    let mut s_neg = 0.0_f64;
    for i in 1..bars.len() {
        let r = (bars[i].close / bars[i - 1].close).ln();
        let h = sigma[i] * h_mult;
        s_pos = (s_pos + r).max(0.0);
        s_neg = (s_neg + r).min(0.0);
        if h > 0.0 && s_pos > h {
            out.push(i);
            s_pos = 0.0;
            s_neg = 0.0;
        } else if h > 0.0 && -s_neg > h {
            out.push(i);
            s_pos = 0.0;
            s_neg = 0.0;
        }
    }
    out
}

/// Donchian-channel breakout filter. Fires when bar `i`'s close strictly
/// exceeds the highest close in `[i-lookback, i)` or strictly undercuts
/// the lowest close in that window.
pub fn breakout_events(bars: &[Bar10s], lookback: usize) -> Vec<usize> {
    let mut out = Vec::new();
    if lookback == 0 || bars.len() <= lookback {
        return out;
    }
    for i in lookback..bars.len() {
        let lo = i - lookback;
        let hi_window = bars[lo..i].iter().map(|b| b.close).fold(f64::MIN, f64::max);
        let lo_window = bars[lo..i].iter().map(|b| b.close).fold(f64::MAX, f64::min);
        if bars[i].close > hi_window || bars[i].close < lo_window {
            out.push(i);
        }
    }
    out
}

/// Combine multiple event index lists, deduplicate, sort, and enforce
/// `min_gap` between consecutive picks. Public so tests / Python can
/// call it directly with custom event sources.
pub fn merge_and_gap(mut all: Vec<usize>, min_gap: usize) -> Vec<usize> {
    all.sort_unstable();
    all.dedup();
    if min_gap <= 1 || all.is_empty() {
        return all;
    }
    let mut out = Vec::with_capacity(all.len());
    let mut last = usize::MAX;
    for idx in all {
        if last == usize::MAX || idx >= last + min_gap {
            out.push(idx);
            last = idx;
        }
    }
    out
}

/// One-shot event sampler combining the configured filters.
pub fn sample_events(bars: &[Bar10s], sigma: &[f64], cfg: &EventConfig) -> Vec<usize> {
    let mut all = Vec::new();
    if cfg.cusum_enabled {
        all.extend(cusum_filter(bars, sigma, cfg.cusum_h_mult));
    }
    if cfg.breakout_enabled {
        all.extend(breakout_events(bars, cfg.breakout_lookback));
    }
    merge_and_gap(all, cfg.min_gap)
}

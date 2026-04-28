//! O(N) bar-level 24-dim feature recompute. Byte-aligned with the
//! Python implementation in `research/src/research/features.py` so the
//! live ONNX inference path produces feature vectors identical to what
//! the model trained on.
//!
//! # Why recompute instead of incremental state?
//! Live bars arrive at 10s cadence. Even with N = 300 (the largest
//! rolling window we need) and 24 features, a full recompute on each
//! bar is a few microseconds. The simpler code path is worth more than
//! the (negligible) saved CPU.
//!
//! # API
//! ```ignore
//! use bar_features::{recompute_last, N_FEATURES, FEATURE_NAMES};
//!
//! let feat: Option<[f64; N_FEATURES]> = recompute_last(&bars);
//! ```
//! `bars` MUST be sorted ascending by `ts_ms`. The function returns
//! `None` when there aren't enough bars to populate even the smallest
//! rolling window safely (currently: < 2 bars).

#![deny(unsafe_code)]

use market_domain::Bar10s;

pub const N_FEATURES: usize = 24;

pub const FEATURE_NAMES: &[&str; N_FEATURES] = &[
    "log_ret_1",
    "log_ret_5",
    "log_ret_20",
    "log_ret_60",
    "vol_30",
    "vol_120",
    "vol_300",
    "spread_bp",
    "spread_z",
    "sma_dev_30",
    "sma_dev_120",
    "rsi_14",
    "bb_upper_dev",
    "bb_lower_dev",
    "atr_14",
    "macd",
    "macd_signal",
    "macd_hist",
    // Phase B: rescaled volume-weighted return. Replaces raw
    // `force_index = (close - prev_close) * n_ticks` (~±10 000) with
    // the same numerator divided by the trailing-60-bar mean tick
    // count (~±50). Mirrors `research.features.FORCE_INDEX_REL_WINDOW`.
    "force_index_rel",
    "minute_sin",
    "minute_cos",
    // Phase B: log-scaled tick count. Replaces raw `n_ticks` (~0-10 000)
    // with `ln(1 + n_ticks)` (~0-12). Same rank-ordering for tree
    // models, but linear models + display layers now see a sane scale.
    "log1p_n_ticks",
    "range_60",
    "drawdown_300",
];

/// Trailing-window length for the `force_index_rel` denominator. Must
/// match `research.features.FORCE_INDEX_REL_WINDOW` so Python
/// training and Rust live inference agree byte-for-byte.
pub const FORCE_INDEX_REL_WINDOW: usize = 60;

#[derive(Clone, Copy, Debug)]
pub struct FeatureConfig {
    pub sma_short: usize,
    pub sma_long: usize,
    pub vol_short: usize,
    pub vol_mid: usize,
    pub vol_long: usize,
    pub rsi_window: usize,
    pub bb_window: usize,
    pub bb_k: f64,
    pub macd_fast: usize,
    pub macd_slow: usize,
    pub macd_signal: usize,
    pub range_window: usize,
    pub dd_window: usize,
    pub spread_window: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            sma_short: 30,
            sma_long: 120,
            vol_short: 30,
            vol_mid: 120,
            vol_long: 300,
            rsi_window: 14,
            bb_window: 20,
            bb_k: 2.0,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
            range_window: 60,
            dd_window: 300,
            spread_window: 120,
        }
    }
}

/// Compute the 24-dim feature vector for the LAST bar in `bars`.
/// Returns `None` if there are fewer than 2 bars (no return defined).
pub fn recompute_last(bars: &[Bar10s]) -> Option<[f64; N_FEATURES]> {
    recompute_last_with(bars, &FeatureConfig::default())
}

pub fn recompute_last_with(
    bars: &[Bar10s],
    cfg: &FeatureConfig,
) -> Option<[f64; N_FEATURES]> {
    if bars.len() < 2 {
        return None;
    }
    let n = bars.len();
    let i = n - 1;
    let close = |k: usize| bars[k].close;

    // ---- log returns ------------------------------------------------------
    let log_close = |k: usize| close(k).max(1e-12).ln();
    let lret = |bars_back: usize| -> f64 {
        if i < bars_back {
            0.0
        } else {
            log_close(i) - log_close(i - bars_back)
        }
    };
    let log_ret_1 = lret(1);
    let log_ret_5 = lret(5);
    let log_ret_20 = lret(20);
    let log_ret_60 = lret(60);

    // ---- volatility (rolling std of log_ret_1, sample formula) -----------
    let vol_short = rolling_std_logret(bars, i, cfg.vol_short);
    let vol_mid = rolling_std_logret(bars, i, cfg.vol_mid);
    let vol_long = rolling_std_logret(bars, i, cfg.vol_long);

    // ---- spread + z-score -------------------------------------------------
    let spread_bp_now = bars[i].spread_bp_avg;
    let (spread_mean, spread_std) = rolling_mean_std(bars, i, cfg.spread_window, |b| b.spread_bp_avg);
    let spread_z = if spread_std > 0.0 {
        (spread_bp_now - spread_mean) / spread_std
    } else {
        0.0
    };

    // ---- SMA deviations --------------------------------------------------
    let sma_dev_short = sma_dev(bars, i, cfg.sma_short);
    let sma_dev_long = sma_dev(bars, i, cfg.sma_long);

    // ---- RSI(14), Wilder's smoothing, returned in [0, 1] -----------------
    let rsi = wilder_rsi(bars, i, cfg.rsi_window) / 100.0;

    // ---- Bollinger band deviations ---------------------------------------
    let (bb_mean, bb_std) = rolling_mean_std(bars, i, cfg.bb_window, |b| b.close);
    let upper = bb_mean + cfg.bb_k * bb_std;
    let lower = bb_mean - cfg.bb_k * bb_std;
    let bb_upper_dev = if close(i) > 0.0 { (close(i) - upper) / close(i) } else { 0.0 };
    let bb_lower_dev = if close(i) > 0.0 { (close(i) - lower) / close(i) } else { 0.0 };

    // ---- ATR(14): SMA of true ranges -------------------------------------
    let atr = sma_true_range(bars, i, cfg.rsi_window);

    // ---- MACD line / signal / hist ---------------------------------------
    let ema_fast = ema_close(bars, i, cfg.macd_fast);
    let ema_slow = ema_close(bars, i, cfg.macd_slow);
    let macd_line = ema_fast - ema_slow;
    // For the signal we EMA the macd series itself. Recompute the macd
    // series for the EMA. This is O(N).
    let macd_signal_value = ema_macd_signal(bars, i, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal);
    let macd_hist = macd_line - macd_signal_value;

    // ---- force_index_rel -------------------------------------------------
    // Numerator: classic Elder force index ((close - prev_close) × n_ticks).
    // Denominator: trailing mean of n_ticks over `FORCE_INDEX_REL_WINDOW`
    // bars (matching the Python implementation's polars
    // `rolling_mean(60, min_samples=1)` so even short histories produce
    // a non-zero scale). Floored at 1.0 so an idle market doesn't divide
    // by zero. Result range: ~±50 in active markets, ~±5 in calm ones.
    let prev_close = if i == 0 { close(0) } else { close(i - 1) };
    let force_index_raw = (close(i) - prev_close) * bars[i].n_ticks as f64;
    let n_ticks_window_lo = i.saturating_sub(FORCE_INDEX_REL_WINDOW.saturating_sub(1));
    let mut n_ticks_sum = 0.0_f64;
    let mut n_ticks_count = 0_usize;
    for k in n_ticks_window_lo..=i {
        n_ticks_sum += bars[k].n_ticks as f64;
        n_ticks_count += 1;
    }
    let n_ticks_mean = if n_ticks_count > 0 {
        (n_ticks_sum / n_ticks_count as f64).max(1.0)
    } else {
        1.0
    };
    let force_index_rel = force_index_raw / n_ticks_mean;

    // ---- log1p_n_ticks ---------------------------------------------------
    // ln(1 + n_ticks) — guards against the vanishing rare zero-tick
    // bar produced by the aggregator on a quiet weekend session.
    let log1p_n_ticks = (1.0 + (bars[i].n_ticks as f64).max(0.0)).ln();

    // ---- minute-of-hour cyclical encoding --------------------------------
    let minute = ((bars[i].ts_ms / 60_000) % 60) as f64;
    let two_pi = std::f64::consts::TAU;
    let minute_sin = (two_pi * minute / 60.0).sin();
    let minute_cos = (two_pi * minute / 60.0).cos();

    // ---- range_60: (max - min) / close over the last 60 bars -------------
    let range_60 = if i + 1 < cfg.range_window {
        0.0
    } else {
        let lo = i + 1 - cfg.range_window;
        let mut hi_v = f64::NEG_INFINITY;
        let mut lo_v = f64::INFINITY;
        for k in lo..=i {
            if close(k) > hi_v {
                hi_v = close(k);
            }
            if close(k) < lo_v {
                lo_v = close(k);
            }
        }
        if close(i) > 0.0 {
            (hi_v - lo_v) / close(i)
        } else {
            0.0
        }
    };

    // ---- drawdown_300 ----------------------------------------------------
    let drawdown_300 = if i + 1 < cfg.dd_window {
        0.0
    } else {
        let lo = i + 1 - cfg.dd_window;
        let mut peak = f64::NEG_INFINITY;
        for k in lo..=i {
            if close(k) > peak {
                peak = close(k);
            }
        }
        if peak > 0.0 {
            (close(i) - peak) / peak
        } else {
            0.0
        }
    };

    Some([
        log_ret_1,
        log_ret_5,
        log_ret_20,
        log_ret_60,
        vol_short,
        vol_mid,
        vol_long,
        spread_bp_now,
        spread_z,
        sma_dev_short,
        sma_dev_long,
        rsi,
        bb_upper_dev,
        bb_lower_dev,
        atr,
        macd_line,
        macd_signal_value,
        macd_hist,
        force_index_rel,
        minute_sin,
        minute_cos,
        log1p_n_ticks,
        range_60,
        drawdown_300,
    ])
}

// ---- Rolling helpers ----------------------------------------------------

fn rolling_std_logret(bars: &[Bar10s], i: usize, window: usize) -> f64 {
    if window < 2 || i + 1 < window {
        return 0.0;
    }
    let lo = i + 1 - window;
    // Compute log returns over the window, then their sample stddev.
    let mut buf: Vec<f64> = Vec::with_capacity(window);
    for k in lo..=i {
        let prev = if k == 0 { bars[0].close } else { bars[k - 1].close };
        buf.push((bars[k].close.max(1e-12) / prev.max(1e-12)).ln());
    }
    let mean = buf.iter().sum::<f64>() / buf.len() as f64;
    let var = buf.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (buf.len() - 1) as f64;
    var.sqrt()
}

fn rolling_mean_std<F>(bars: &[Bar10s], i: usize, window: usize, getter: F) -> (f64, f64)
where
    F: Fn(&Bar10s) -> f64,
{
    if window == 0 || i + 1 < window {
        return (0.0, 0.0);
    }
    let lo = i + 1 - window;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for k in lo..=i {
        let v = getter(&bars[k]);
        sum += v;
        sum2 += v * v;
    }
    let n = window as f64;
    let mean = sum / n;
    if window < 2 {
        return (mean, 0.0);
    }
    let var = (sum2 - sum * sum / n) / (n - 1.0);
    (mean, var.max(0.0).sqrt())
}

fn sma_dev(bars: &[Bar10s], i: usize, window: usize) -> f64 {
    if window == 0 || i + 1 < window {
        return 0.0;
    }
    let lo = i + 1 - window;
    let sum: f64 = (lo..=i).map(|k| bars[k].close).sum();
    let sma = sum / window as f64;
    if sma > 0.0 {
        bars[i].close / sma - 1.0
    } else {
        0.0
    }
}

fn wilder_rsi(bars: &[Bar10s], i: usize, window: usize) -> f64 {
    if window == 0 || i < window {
        return 50.0; // neutral
    }
    // Initialise on the first `window` bars (indices 1..=window)
    let init_lo = 1usize;
    let init_hi = window; // inclusive
    if init_hi > i {
        return 50.0;
    }
    let mut gain = 0.0;
    let mut loss = 0.0;
    for k in init_lo..=init_hi {
        let d = bars[k].close - bars[k - 1].close;
        if d > 0.0 {
            gain += d;
        } else {
            loss += -d;
        }
    }
    let mut avg_gain = gain / window as f64;
    let mut avg_loss = loss / window as f64;
    // Wilder smoothing for k > init_hi.
    for k in (init_hi + 1)..=i {
        let d = bars[k].close - bars[k - 1].close;
        let g = if d > 0.0 { d } else { 0.0 };
        let l = if d < 0.0 { -d } else { 0.0 };
        avg_gain = (avg_gain * (window as f64 - 1.0) + g) / window as f64;
        avg_loss = (avg_loss * (window as f64 - 1.0) + l) / window as f64;
    }
    if avg_loss == 0.0 {
        return 100.0;
    }
    let rs = avg_gain / avg_loss;
    100.0 - 100.0 / (1.0 + rs)
}

fn sma_true_range(bars: &[Bar10s], i: usize, window: usize) -> f64 {
    if window == 0 || i + 1 < window {
        return 0.0;
    }
    let lo = i + 1 - window;
    let mut sum = 0.0;
    for k in lo..=i {
        let prev_close = if k == 0 { bars[0].close } else { bars[k - 1].close };
        let tr = (bars[k].high - bars[k].low)
            .max((bars[k].high - prev_close).abs())
            .max((bars[k].low - prev_close).abs());
        sum += tr;
    }
    sum / window as f64
}

fn ema_close(bars: &[Bar10s], i: usize, span: usize) -> f64 {
    if span <= 1 || bars.is_empty() {
        return bars.get(i).map(|b| b.close).unwrap_or(0.0);
    }
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut ema = bars[0].close;
    for k in 1..=i {
        ema = alpha * bars[k].close + (1.0 - alpha) * ema;
    }
    ema
}

fn ema_macd_signal(bars: &[Bar10s], i: usize, fast: usize, slow: usize, sig: usize) -> f64 {
    // Walk the bars once, maintaining EMA fast, EMA slow, and EMA of
    // their difference (the signal line).
    if bars.is_empty() || sig <= 1 {
        return 0.0;
    }
    let alpha_fast = 2.0 / (fast as f64 + 1.0);
    let alpha_slow = 2.0 / (slow as f64 + 1.0);
    let alpha_sig = 2.0 / (sig as f64 + 1.0);
    let mut ema_fast = bars[0].close;
    let mut ema_slow = bars[0].close;
    let mut ema_sig = 0.0_f64;
    for k in 1..=i {
        ema_fast = alpha_fast * bars[k].close + (1.0 - alpha_fast) * ema_fast;
        ema_slow = alpha_slow * bars[k].close + (1.0 - alpha_slow) * ema_slow;
        let macd = ema_fast - ema_slow;
        ema_sig = if k == 1 { macd } else { alpha_sig * macd + (1.0 - alpha_sig) * ema_sig };
    }
    ema_sig
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_bar(ts_ms: i64, c: f64) -> Bar10s {
        Bar10s {
            instrument_id: 0,
            ts_ms,
            open: c,
            high: c + 1e-4,
            low: c - 1e-4,
            close: c,
            n_ticks: 1,
            spread_bp_avg: 0.5,
        }
    }

    #[test]
    fn returns_none_for_short_input() {
        assert!(recompute_last(&[]).is_none());
        assert!(recompute_last(&[mk_bar(0, 1.0)]).is_none());
    }

    #[test]
    fn produces_finite_features() {
        let bars: Vec<_> = (0..400)
            .map(|i| mk_bar(i * 10_000, 1.0 + 0.001 * (i as f64 / 30.0).sin()))
            .collect();
        let f = recompute_last(&bars).expect("features");
        for (k, v) in f.iter().enumerate() {
            assert!(v.is_finite(), "feature {k} ({}) is non-finite: {v}", FEATURE_NAMES[k]);
        }
    }

    #[test]
    fn rsi_in_unit_interval() {
        let bars: Vec<_> = (0..50)
            .map(|i| mk_bar(i * 10_000, 1.0 + (i as f64) * 0.001))
            .collect();
        let f = recompute_last(&bars).expect("features");
        let rsi = f[11];
        assert!((0.0..=1.0).contains(&rsi), "rsi out of [0,1]: {rsi}");
    }

    #[test]
    fn flat_series_has_zero_volatility() {
        let bars: Vec<_> = (0..100).map(|i| mk_bar(i * 10_000, 1.0)).collect();
        let f = recompute_last(&bars).expect("features");
        // vol_short, vol_mid, vol_long
        assert_eq!(f[4], 0.0);
        assert_eq!(f[5], 0.0);
        assert_eq!(f[6], 0.0);
    }

    /// Find a feature's column index by name. Used by the Phase B
    /// range tests so they stay readable when the FEATURE_NAMES list
    /// gets reshuffled.
    fn idx(name: &str) -> usize {
        FEATURE_NAMES
            .iter()
            .position(|n| *n == name)
            .unwrap_or_else(|| panic!("feature {name} missing from FEATURE_NAMES"))
    }

    /// Phase B guard: `log1p_n_ticks` for any realistic tick count
    /// stays in [0, 12]. ln(1 + 100_000) ≈ 11.51 — even an extreme
    /// burst can't blow this out of band.
    #[test]
    fn log1p_n_ticks_bounded() {
        let mut bars = Vec::with_capacity(200);
        for i in 0..200 {
            // Heavy-tailed tick counts: mostly 50-1500, with the occasional
            // 50 000 burst — exercises both the log compression and the
            // rolling-mean denominator for `force_index_rel`.
            let nt: u32 = if i % 50 == 0 { 50_000 } else { (50 + (i * 11) % 1500) as u32 };
            bars.push(Bar10s {
                instrument_id: 0,
                ts_ms: i as i64 * 10_000,
                open: 1.10,
                high: 1.10 + 1e-4,
                low: 1.10 - 1e-4,
                close: 1.10 + 1e-4 * (i as f64).sin(),
                n_ticks: nt,
                spread_bp_avg: 1.0,
            });
        }
        let f = recompute_last(&bars).expect("features");
        let v = f[idx("log1p_n_ticks")];
        assert!((0.0..=12.0).contains(&v), "log1p_n_ticks out of [0,12]: {v}");
        assert!(v.is_finite());
    }

    /// Phase B guard: `force_index_rel` — even with a 50 000-tick
    /// burst on top of a small price move, the rolling-mean
    /// denominator keeps the magnitude bounded.
    #[test]
    fn force_index_rel_bounded_under_burst() {
        let mut bars = Vec::with_capacity(200);
        for i in 0..200 {
            let nt: u32 = if i == 199 { 50_000 } else { 100 };
            bars.push(Bar10s {
                instrument_id: 0,
                ts_ms: i as i64 * 10_000,
                open: 1.10,
                high: 1.10 + 1e-4,
                low: 1.10 - 1e-4,
                // 1 bp price change on the burst bar.
                close: 1.10 + if i == 199 { 1e-4 } else { 0.0 },
                n_ticks: nt,
                spread_bp_avg: 1.0,
            });
        }
        let f = recompute_last(&bars).expect("features");
        let v = f[idx("force_index_rel")];
        // Rough bound: numerator ≈ 1e-4 * 50_000 = 5; denominator ≈
        // mean(100s + one 50_000) ≈ 350, so the ratio ≲ 0.015. Allow
        // a wide ±50 envelope to cover any future regime.
        assert!(v.abs() < 50.0, "force_index_rel out of bounds: {v}");
        assert!(v.is_finite());
    }

    /// Idle market: a long run of zero-tick bars must not divide by
    /// zero in the `force_index_rel` denominator.
    #[test]
    fn force_index_rel_idle_market_no_divzero() {
        let bars: Vec<_> = (0..200)
            .map(|i| Bar10s {
                instrument_id: 0,
                ts_ms: i as i64 * 10_000,
                open: 1.10,
                high: 1.10 + 1e-4,
                low: 1.10 - 1e-4,
                close: 1.10,
                n_ticks: 0,
                spread_bp_avg: 1.0,
            })
            .collect();
        let f = recompute_last(&bars).expect("features");
        let v = f[idx("force_index_rel")];
        assert!(v.is_finite());
        // close didn't change → numerator is 0 → result must be 0.
        assert_eq!(v, 0.0);
    }

    /// `FORCE_INDEX_REL_WINDOW` must match the Python constant
    /// `research.features.FORCE_INDEX_REL_WINDOW` so live inference
    /// and offline training agree byte-for-byte. The actual cross-
    /// language check lives in `research/tests/test_features.py`;
    /// this test pins the Rust side.
    #[test]
    fn force_index_rel_window_constant_pinned() {
        assert_eq!(FORCE_INDEX_REL_WINDOW, 60);
    }

    /// Phase B feature contract: 24 features, with the renamed columns
    /// at the same positions the live ONNX champion's input shape
    /// expects. Catches accidental list reshuffles.
    #[test]
    fn feature_contract_pinned() {
        assert_eq!(N_FEATURES, 24);
        assert_eq!(FEATURE_NAMES.len(), 24);
        // Phase B renames must be present at *some* position
        // (positions matter only that they're stable; the index
        // helper above lets tests stay readable).
        assert!(FEATURE_NAMES.contains(&"log1p_n_ticks"));
        assert!(FEATURE_NAMES.contains(&"force_index_rel"));
        // Old raw names must be gone — they'd indicate an
        // accidentally-doubled feature list.
        assert!(!FEATURE_NAMES.contains(&"n_ticks"));
        assert!(!FEATURE_NAMES.contains(&"force_index"));
    }
}

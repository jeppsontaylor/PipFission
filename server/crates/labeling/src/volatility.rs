//! Volatility estimators used to scale triple-barrier widths and to
//! threshold CUSUM event filters. All inputs are slices of bar data;
//! all outputs are aligned series of the same length.

use market_domain::Bar10s;

/// Exponentially-weighted standard deviation of close-to-close log
/// returns. `span` is the equivalent SMA window: alpha = 2 / (span + 1).
/// Element 0 of the output is 0.0 (no return defined for the first bar);
/// element i for i >= 1 is the EWMA σ over returns up to (and including)
/// bar i. The estimator uses Welford-style EWMA of squared returns,
/// which is numerically stable for streaming use.
pub fn ewma_volatility(bars: &[Bar10s], span: usize) -> Vec<f64> {
    let n = bars.len();
    let mut out = vec![0.0; n];
    if n < 2 || span < 2 {
        return out;
    }
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut s2 = 0.0_f64;
    for i in 1..n {
        let r = (bars[i].close / bars[i - 1].close).ln();
        let r2 = r * r;
        s2 = if i == 1 { r2 } else { alpha * r2 + (1.0 - alpha) * s2 };
        out[i] = s2.sqrt();
    }
    out
}

/// Simple Welles-Wilder ATR (mean of true ranges) over a fixed window.
/// Returns 0.0 for indices < window. Used as an alternative barrier
/// width when log-return-based vol underestimates intraday range.
pub fn atr(bars: &[Bar10s], window: usize) -> Vec<f64> {
    let n = bars.len();
    let mut out = vec![0.0; n];
    if n == 0 || window == 0 {
        return out;
    }
    let mut buf: Vec<f64> = Vec::with_capacity(window);
    for i in 0..n {
        let prev_close = if i == 0 { bars[0].close } else { bars[i - 1].close };
        let tr = (bars[i].high - bars[i].low)
            .max((bars[i].high - prev_close).abs())
            .max((bars[i].low - prev_close).abs());
        if buf.len() == window {
            buf.remove(0);
        }
        buf.push(tr);
        if buf.len() == window {
            out[i] = buf.iter().sum::<f64>() / window as f64;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_bar(c: f64) -> Bar10s {
        Bar10s {
            instrument_id: 0,
            ts_ms: 0,
            open: c,
            high: c,
            low: c,
            close: c,
            n_ticks: 1,
            spread_bp_avg: 0.0,
        }
    }

    #[test]
    fn ewma_handles_constant_series() {
        let bars: Vec<_> = (0..50).map(|_| mk_bar(1.0)).collect();
        let v = ewma_volatility(&bars, 10);
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn ewma_grows_with_volatility() {
        // Alternating ±1% returns produce a non-trivial vol estimate.
        let mut bars = Vec::new();
        let mut p = 1.0;
        for i in 0..100 {
            p *= if i % 2 == 0 { 1.01 } else { 0.99 };
            bars.push(mk_bar(p));
        }
        let v = ewma_volatility(&bars, 10);
        assert!(v.last().copied().unwrap_or(0.0) > 0.0);
    }
}

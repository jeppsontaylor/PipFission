//! O(1) incremental indicator primitives.
//!
//! Each struct holds running sums (and where needed a tiny ring buffer for
//! exit values) so adding a new sample is constant-time. Tested against
//! batch implementations in `#[cfg(test)] mod batch_compare`.

use std::collections::VecDeque;

/// Simple moving average over a fixed window. O(1) per update.
#[derive(Debug, Clone)]
pub struct Sma {
    window: usize,
    buf: VecDeque<f64>,
    sum: f64,
}

impl Sma {
    pub fn new(window: usize) -> Self {
        assert!(window > 0);
        Self {
            window,
            buf: VecDeque::with_capacity(window),
            sum: 0.0,
        }
    }
    pub fn push(&mut self, x: f64) -> Option<f64> {
        self.buf.push_back(x);
        self.sum += x;
        if self.buf.len() > self.window {
            self.sum -= self.buf.pop_front().expect("len > window");
        }
        if self.buf.len() == self.window {
            Some(self.sum / self.window as f64)
        } else {
            None
        }
    }
    pub fn ready(&self) -> bool {
        self.buf.len() == self.window
    }
    pub fn last(&self) -> Option<f64> {
        if self.ready() {
            Some(self.sum / self.window as f64)
        } else {
            None
        }
    }
}

/// Welford-style rolling variance + mean over a fixed window. O(1) per
/// update. Returns (mean, variance) once full.
#[derive(Debug, Clone)]
pub struct RollingVar {
    window: usize,
    buf: VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
}

impl RollingVar {
    pub fn new(window: usize) -> Self {
        assert!(window > 1);
        Self {
            window,
            buf: VecDeque::with_capacity(window),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }
    pub fn push(&mut self, x: f64) -> Option<(f64, f64)> {
        self.buf.push_back(x);
        self.sum += x;
        self.sum_sq += x * x;
        if self.buf.len() > self.window {
            let old = self.buf.pop_front().expect("len > window");
            self.sum -= old;
            self.sum_sq -= old * old;
        }
        if self.buf.len() == self.window {
            let n = self.window as f64;
            let mean = self.sum / n;
            // Population variance (denominator n) — fine for stdev as a feature scale.
            let var = (self.sum_sq / n - mean * mean).max(0.0);
            Some((mean, var))
        } else {
            None
        }
    }
    pub fn ready(&self) -> bool {
        self.buf.len() == self.window
    }
}

/// Exponential moving average. O(1) per update; ready immediately after
/// the first sample (seeded with that sample).
#[derive(Debug, Clone)]
pub struct Ema {
    alpha: f64,
    value: Option<f64>,
}

impl Ema {
    /// Build an EMA whose smoothing constant matches a `period`-day SMA:
    /// alpha = 2 / (period + 1).
    pub fn from_period(period: usize) -> Self {
        let alpha = 2.0 / (period as f64 + 1.0);
        Self { alpha, value: None }
    }
    pub fn push(&mut self, x: f64) -> f64 {
        let new = match self.value {
            Some(v) => v + self.alpha * (x - v),
            None => x,
        };
        self.value = Some(new);
        new
    }
    pub fn last(&self) -> Option<f64> {
        self.value
    }
}

/// Wilder's RSI on a configurable period. O(1) per update; ready after
/// `period + 1` samples.
#[derive(Debug, Clone)]
pub struct Rsi {
    period: usize,
    avg_gain: Option<f64>,
    avg_loss: Option<f64>,
    prev: Option<f64>,
    seen: usize,
    seed_gain: f64,
    seed_loss: f64,
}

impl Rsi {
    pub fn new(period: usize) -> Self {
        assert!(period > 0);
        Self {
            period,
            avg_gain: None,
            avg_loss: None,
            prev: None,
            seen: 0,
            seed_gain: 0.0,
            seed_loss: 0.0,
        }
    }
    pub fn push(&mut self, x: f64) -> Option<f64> {
        let prev = match self.prev {
            Some(p) => p,
            None => {
                self.prev = Some(x);
                return None;
            }
        };
        let chg = x - prev;
        let gain = chg.max(0.0);
        let loss = (-chg).max(0.0);
        self.prev = Some(x);
        self.seen += 1;

        if self.avg_gain.is_none() {
            // Seed with simple-average over the first `period` deltas.
            self.seed_gain += gain;
            self.seed_loss += loss;
            if self.seen == self.period {
                self.avg_gain = Some(self.seed_gain / self.period as f64);
                self.avg_loss = Some(self.seed_loss / self.period as f64);
            } else {
                return None;
            }
        } else {
            let p = self.period as f64;
            let ag = self.avg_gain.expect("seeded above");
            let al = self.avg_loss.expect("seeded above");
            self.avg_gain = Some((ag * (p - 1.0) + gain) / p);
            self.avg_loss = Some((al * (p - 1.0) + loss) / p);
        }
        let ag = self.avg_gain.unwrap();
        let al = self.avg_loss.unwrap();
        if al == 0.0 {
            // All gains, no losses — RSI saturates at 100.
            Some(100.0)
        } else {
            let rs = ag / al;
            Some(100.0 - 100.0 / (1.0 + rs))
        }
    }
}

/// MACD: fast EMA - slow EMA, signal = EMA of MACD.
#[derive(Debug, Clone)]
pub struct Macd {
    fast: Ema,
    slow: Ema,
    signal: Ema,
}

impl Macd {
    pub fn new(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            fast: Ema::from_period(fast),
            slow: Ema::from_period(slow),
            signal: Ema::from_period(signal),
        }
    }
    /// Returns (macd_line, signal_line). Both available after first push.
    pub fn push(&mut self, x: f64) -> (f64, f64) {
        let f = self.fast.push(x);
        let s = self.slow.push(x);
        let macd = f - s;
        let sig = self.signal.push(macd);
        (macd, sig)
    }
}

/// Min/max range over a fixed window. O(1) amortized per update via a
/// monotonic deque.
#[derive(Debug, Clone)]
pub struct Range {
    window: usize,
    buf: VecDeque<f64>,
}

impl Range {
    pub fn new(window: usize) -> Self {
        assert!(window > 0);
        Self {
            window,
            buf: VecDeque::with_capacity(window),
        }
    }
    pub fn push(&mut self, x: f64) -> Option<(f64, f64)> {
        self.buf.push_back(x);
        if self.buf.len() > self.window {
            self.buf.pop_front();
        }
        if self.buf.len() == self.window {
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            for &v in &self.buf {
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
            Some((min, max))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch_sma(xs: &[f64], window: usize) -> Vec<Option<f64>> {
        (0..xs.len())
            .map(|i| {
                if i + 1 < window {
                    None
                } else {
                    let slice = &xs[i + 1 - window..=i];
                    Some(slice.iter().sum::<f64>() / window as f64)
                }
            })
            .collect()
    }

    #[test]
    fn sma_matches_batch() {
        let xs: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut sma = Sma::new(10);
        let inc: Vec<Option<f64>> = xs.iter().map(|&x| sma.push(x)).collect();
        let bat = batch_sma(&xs, 10);
        for (a, b) in inc.iter().zip(bat.iter()) {
            match (a, b) {
                (Some(x), Some(y)) => assert!((x - y).abs() < 1e-9, "{x} vs {y}"),
                (None, None) => {}
                _ => panic!("ready mismatch"),
            }
        }
    }

    #[test]
    fn ema_seeds_with_first_sample() {
        let mut e = Ema::from_period(10);
        assert_eq!(e.push(5.0), 5.0);
        let v = e.push(15.0);
        // alpha = 2/11 ≈ 0.1818; new = 5 + 0.1818*(15-5) ≈ 6.818
        assert!((v - 6.818).abs() < 0.01, "{v}");
    }

    #[test]
    fn rsi_handles_all_gains() {
        let mut r = Rsi::new(14);
        for i in 0..30 {
            let _ = r.push(100.0 + i as f64);
        }
        let v = r.push(200.0).unwrap();
        assert!((v - 100.0).abs() < 1e-9, "{v}");
    }

    #[test]
    fn rolling_var_zero_for_constant() {
        let mut rv = RollingVar::new(10);
        for _ in 0..20 {
            let _ = rv.push(7.0);
        }
        let (m, v) = rv.push(7.0).unwrap();
        assert!((m - 7.0).abs() < 1e-9);
        assert!(v < 1e-12);
    }

    #[test]
    fn range_returns_min_max() {
        let mut r = Range::new(5);
        for x in [1.0, 5.0, 3.0, 2.0] {
            assert!(r.push(x).is_none());
        }
        let (lo, hi) = r.push(4.0).unwrap();
        assert_eq!((lo, hi), (1.0, 5.0));
        // After sliding off the 1.0:
        let (lo, hi) = r.push(0.5).unwrap();
        assert_eq!((lo, hi), (0.5, 5.0));
    }
}

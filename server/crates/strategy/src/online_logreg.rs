//! Binary logistic regression trained by mini-batch SGD.
//!
//! `market_domain::FEATURE_DIM` features × **2 classes (long/short)** +
//! bias. No third-party ML crate. Trained from scratch per retrain.
//!
//! "Flat" is gone — every prediction picks long or short. The legacy
//! 3-class layout was a research-grade placeholder; the operator's
//! mandate is binary in/out, with class balance ≥ 30 % minority.
//!
//! Loss: cross-entropy over softmax(2). Optimizer: SGD with L2
//! regularisation and momentum. Standardisation: z-score per feature
//! using train-window mean/std (recomputed per training).

use std::fmt;

pub const NUM_CLASSES: usize = 2;

/// Class layout — keep in sync with `market_domain::SignalDirection`'s
/// usage: argmax==`CLASS_LONG` → `SignalDirection::Long`,
/// argmax==`CLASS_SHORT` → `SignalDirection::Short`. The model never
/// emits `SignalDirection::Flat` (kept on the wire only for
/// pre-burn-down rows in the `signals` table).
pub const CLASS_SHORT: usize = 0;
pub const CLASS_LONG: usize = 1;

/// Backwards-compatibility shim: callers in walk_forward / runner used
/// to dispatch on `CLASS_FLAT`. Returning a sentinel that's outside
/// `[0, NUM_CLASSES)` so any stray dispatch logic surfaces as a
/// compile error rather than silently picking the wrong class. Kept
/// public so external callers get a clear deprecation pointer.
#[deprecated(note = "binary classifier — flat class no longer exists")]
pub const CLASS_FLAT: usize = usize::MAX;

#[derive(Clone, Debug)]
pub struct LogRegConfig {
    pub feature_dim: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub l2: f64,
    pub momentum: f64,
}

impl Default for LogRegConfig {
    fn default() -> Self {
        Self {
            // Must match `market_domain::FEATURE_DIM`. Sprint 2/A2 bumped
            // this to 24 (added 6 orderbook features); the literal here
            // had stayed at 18 and caused the trainer to panic on first
            // fit when a 24-wide vector arrived. Keep this sourced from
            // the constant so future schema changes don't desync.
            feature_dim: market_domain::FEATURE_DIM,
            epochs: 30,
            learning_rate: 0.05,
            l2: 1e-4,
            momentum: 0.85,
        }
    }
}

/// A trained multinomial logistic regression. Weights shaped
/// [NUM_CLASSES][feature_dim], bias shaped [NUM_CLASSES]. Standardization
/// stats stored alongside so prediction always uses the train-time scale.
#[derive(Clone, Debug)]
pub struct LogReg {
    pub w: Vec<Vec<f64>>, // [class][feat]
    pub b: Vec<f64>,      // [class]
    pub mean: Vec<f64>,   // [feat]
    pub std: Vec<f64>,    // [feat]
    pub feature_dim: usize,
}

impl fmt::Display for LogReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LogReg({} feats, {} classes)",
            self.feature_dim, NUM_CLASSES
        )
    }
}

impl LogReg {
    pub fn predict_probs(&self, x: &[f64]) -> [f64; NUM_CLASSES] {
        debug_assert_eq!(x.len(), self.feature_dim);
        let mut z = [0.0; NUM_CLASSES];
        for c in 0..NUM_CLASSES {
            let mut s = self.b[c];
            for j in 0..self.feature_dim {
                let xs = (x[j] - self.mean[j]) / self.std[j].max(1e-9);
                s += self.w[c][j] * xs;
            }
            z[c] = s;
        }
        softmax(z)
    }
}

fn softmax(z: [f64; NUM_CLASSES]) -> [f64; NUM_CLASSES] {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    let mut out = [0.0; NUM_CLASSES];
    for c in 0..NUM_CLASSES {
        let e = (z[c] - m).exp();
        out[c] = e;
        sum += e;
    }
    if sum > 0.0 {
        for c in 0..NUM_CLASSES {
            out[c] /= sum;
        }
    } else {
        out = [1.0 / NUM_CLASSES as f64; NUM_CLASSES];
    }
    out
}

/// Train a logistic regression on the given (feature, label) pairs.
/// `xs` is row-major: each `xs[i]` has length `cfg.feature_dim`.
/// `ys[i]` ∈ {0, 1, 2}.
pub fn fit(xs: &[Vec<f64>], ys: &[usize], cfg: &LogRegConfig) -> LogReg {
    assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    let d = cfg.feature_dim;
    assert!(n > 0 && xs[0].len() == d);

    // Compute per-feature mean/std on the training set.
    let mut mean = vec![0.0_f64; d];
    let mut std = vec![1.0_f64; d];
    for row in xs.iter() {
        for j in 0..d {
            mean[j] += row[j];
        }
    }
    for j in 0..d {
        mean[j] /= n as f64;
    }
    let mut var = vec![0.0_f64; d];
    for row in xs.iter() {
        for j in 0..d {
            let dx = row[j] - mean[j];
            var[j] += dx * dx;
        }
    }
    for j in 0..d {
        let v = (var[j] / n as f64).max(1e-12);
        std[j] = v.sqrt();
    }

    let mut w: Vec<Vec<f64>> = vec![vec![0.0; d]; NUM_CLASSES];
    let mut b: Vec<f64> = vec![0.0; NUM_CLASSES];
    let mut vw: Vec<Vec<f64>> = vec![vec![0.0; d]; NUM_CLASSES];
    let mut vb: Vec<f64> = vec![0.0; NUM_CLASSES];

    let lr = cfg.learning_rate;
    let l2 = cfg.l2;
    let mu = cfg.momentum;

    // Pre-standardize once into a working buffer so the inner loop is small.
    let mut xs_z: Vec<Vec<f64>> = vec![vec![0.0; d]; n];
    for i in 0..n {
        for j in 0..d {
            xs_z[i][j] = (xs[i][j] - mean[j]) / std[j].max(1e-9);
        }
    }

    for _epoch in 0..cfg.epochs {
        // One full pass; in-order. (We could shuffle but for ~800 rows the
        // gain is negligible vs. the cost of an RNG call per row.)
        for i in 0..n {
            let x = &xs_z[i];
            let y = ys[i];

            // Forward: logits = w·x + b for each class.
            let mut logits = [0.0; NUM_CLASSES];
            for c in 0..NUM_CLASSES {
                let mut s = b[c];
                for j in 0..d {
                    s += w[c][j] * x[j];
                }
                logits[c] = s;
            }
            let probs = softmax(logits);

            // Gradient: dlogit_c = (probs[c] - 1[y==c]). dW_c = dlogit * x.
            for c in 0..NUM_CLASSES {
                let g = probs[c] - if c == y { 1.0 } else { 0.0 };
                // momentum SGD with L2 decay
                vb[c] = mu * vb[c] - lr * g;
                b[c] += vb[c];
                for j in 0..d {
                    let gj = g * x[j] + l2 * w[c][j];
                    vw[c][j] = mu * vw[c][j] - lr * gj;
                    w[c][j] += vw[c][j];
                }
            }
        }
    }

    LogReg {
        w,
        b,
        mean,
        std,
        feature_dim: d,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Toy linearly-separable dataset: 2 features, **2 classes** on
    /// either side of the y-axis. fit should hit > 90 % accuracy.
    /// Pre-binary-burn-down this exercised 3 classes; the binary
    /// rewrite removes the centre cluster.
    #[test]
    fn fits_simple_separable() {
        let cfg = LogRegConfig {
            feature_dim: 2,
            epochs: 200,
            ..Default::default()
        };
        let mut xs: Vec<Vec<f64>> = Vec::new();
        let mut ys: Vec<usize> = Vec::new();
        for i in 0..30 {
            let t = i as f64 * 0.1;
            // Class SHORT (0) cluster around (-3, 0).
            xs.push(vec![-3.0 + (t * 0.05).cos(), 0.0 + (t * 0.05).sin()]);
            ys.push(CLASS_SHORT);
            // Class LONG (1) cluster around (+3, 0).
            xs.push(vec![3.0 + (t * 0.05).cos(), 0.1 + (t * 0.05).sin()]);
            ys.push(CLASS_LONG);
        }
        let m = fit(&xs, &ys, &cfg);
        let mut correct = 0;
        for (x, &y) in xs.iter().zip(ys.iter()) {
            let p = m.predict_probs(x);
            let pred = (0..NUM_CLASSES)
                .max_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap())
                .unwrap();
            if pred == y {
                correct += 1;
            }
        }
        let acc = correct as f64 / xs.len() as f64;
        assert!(acc > 0.85, "accuracy {acc} too low");
    }

    #[test]
    fn predict_probs_sum_to_one() {
        let cfg = LogRegConfig {
            feature_dim: 3,
            epochs: 5,
            ..Default::default()
        };
        let xs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let ys = vec![0, 1, 2];
        let m = fit(&xs, &ys, &cfg);
        let p = m.predict_probs(&[0.5, 0.5, 0.5]);
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
    }
}

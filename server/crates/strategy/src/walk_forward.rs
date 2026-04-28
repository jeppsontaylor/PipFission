//! Walk-forward training harness with a purge gap.
//!
//! Given a contiguous time-ordered buffer of `(features, mid)` pairs and a
//! **binary** labeling rule (predict next-N-tick direction: long vs short,
//! no flat), this module:
//! 1. Builds (X, y) pairs after labeling-look-ahead.
//! 2. Splits into train and OOS windows with a purge gap equal to the
//!    look-ahead so labels can't leak.
//! 3. Fits a binary logistic regression on train.
//! 4. Scores accuracy + log-loss + directional sharpe on OOS.

use crate::online_logreg::{fit, LogReg, LogRegConfig, CLASS_LONG, CLASS_SHORT, NUM_CLASSES};
use market_domain::{FitnessMetrics, FEATURE_DIM};

/// Number of *future* feature ticks the label looks ahead. At ~100Hz/inst
/// synthetic this is ~0.1s; at 4Hz live ~2.5s. Tuned for Sprint 2 demo to
/// catch enough movement to learn anything within reason.
pub const LABEL_HORIZON_TICKS: usize = 10;

#[derive(Clone, Debug)]
pub struct LabeledSample {
    pub feat: [f64; FEATURE_DIM],
    pub label: usize,
    /// Forward log-return that produced the label. Used for sharpe scoring.
    pub forward_lr: f64,
}

/// Build labeled samples from a (feature_vector, mid) ring. `mids[i]` is
/// the mid at the time `feats[i]` was emitted. Returns one sample per
/// `i` for which `i + LABEL_HORIZON_TICKS < N`.
pub fn build_labeled(feats: &[[f64; FEATURE_DIM]], mids: &[f64]) -> Vec<LabeledSample> {
    assert_eq!(feats.len(), mids.len());
    let n = feats.len();
    if n <= LABEL_HORIZON_TICKS {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n - LABEL_HORIZON_TICKS);
    for i in 0..(n - LABEL_HORIZON_TICKS) {
        let now = mids[i];
        let later = mids[i + LABEL_HORIZON_TICKS];
        if now <= 0.0 || later <= 0.0 {
            continue;
        }
        let lr = (later / now).ln();
        // Binary labels: every sample is either long or short. A zero
        // forward return is rare (≤ 1 in 10⁴ at 4Hz live) and gets bucketed
        // as long by convention — the optimizer-stage label balance check
        // (Phase B → C) will rebalance regardless.
        let label = if lr >= 0.0 { CLASS_LONG } else { CLASS_SHORT };
        out.push(LabeledSample {
            feat: feats[i],
            label,
            forward_lr: lr,
        });
    }
    out
}

#[derive(Clone, Debug)]
pub struct WalkForwardResult {
    pub model: LogReg,
    pub train: FitnessMetrics,
    pub oos: FitnessMetrics,
    pub train_window: (usize, usize),
    pub oos_window: (usize, usize),
}

/// Train + evaluate. Splits `samples` into `[0..train_end)` for training
/// and `[oos_start..oos_end)` for OOS, with `purge` indices skipped between.
pub fn train_walk_forward(
    samples: &[LabeledSample],
    train_end: usize,
    purge: usize,
    oos_end: usize,
    cfg: &LogRegConfig,
) -> Option<WalkForwardResult> {
    if samples.is_empty()
        || train_end < 50
        || train_end + purge >= oos_end
        || oos_end > samples.len()
    {
        return None;
    }
    let oos_start = train_end + purge;

    let train_xs: Vec<Vec<f64>> = samples[..train_end]
        .iter()
        .map(|s| s.feat.to_vec())
        .collect();
    let train_ys: Vec<usize> = samples[..train_end].iter().map(|s| s.label).collect();
    let model = fit(&train_xs, &train_ys, cfg);

    let train = score(&model, &samples[..train_end]);
    let oos = score(&model, &samples[oos_start..oos_end]);

    Some(WalkForwardResult {
        model,
        train,
        oos,
        train_window: (0, train_end),
        oos_window: (oos_start, oos_end),
    })
}

fn score(model: &LogReg, samples: &[LabeledSample]) -> FitnessMetrics {
    if samples.is_empty() {
        return FitnessMetrics::default();
    }
    let mut correct = 0;
    let mut log_loss_sum = 0.0;
    let mut class_dist = [0_usize; NUM_CLASSES];
    let mut returns: Vec<f64> = Vec::with_capacity(samples.len());

    for s in samples {
        let p = model.predict_probs(&s.feat);
        let argmax = (0..NUM_CLASSES)
            .max_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap())
            .unwrap();
        if argmax == s.label {
            correct += 1;
        }
        class_dist[argmax] += 1;
        let py = p[s.label].max(1e-12);
        log_loss_sum += -py.ln();
        // Binary directional return: long bets +lr, short bets -lr.
        // No flat class — argmax ∈ {CLASS_LONG, CLASS_SHORT}.
        let bet = if argmax == CLASS_LONG {
            s.forward_lr
        } else {
            -s.forward_lr
        };
        returns.push(bet);
    }
    let n = samples.len() as f64;
    let mean_ret = returns.iter().sum::<f64>() / n;
    let var_ret = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / n;
    let sharpe = if var_ret > 1e-18 {
        mean_ret / var_ret.sqrt()
    } else {
        0.0
    };
    // Wire format for class_distribution stays [long, flat, short] so the
    // dashboard + signals table don't break. Binary classifier never emits
    // flat — that bucket is always 0.
    let wire_class_dist = [class_dist[CLASS_LONG], 0_usize, class_dist[CLASS_SHORT]];
    FitnessMetrics {
        samples: samples.len(),
        accuracy: correct as f64 / n,
        log_loss: log_loss_sum / n,
        sharpe,
        class_distribution: wire_class_dist,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_labeled_skips_short_buffer() {
        let feats = vec![[0.0; FEATURE_DIM]; 5];
        let mids = vec![1.0; 5];
        let s = build_labeled(&feats, &mids);
        assert!(s.is_empty(), "horizon=10 vs 5 samples");
    }

    #[test]
    fn build_labeled_binary_split() {
        let mut feats = vec![[0.0; FEATURE_DIM]; 30];
        let mut mids = vec![1.0; 30];
        // Constant mid → forward log-return is exactly 0 → bucketed as LONG
        // by the binary convention (lr >= 0.0).
        for s in build_labeled(&feats, &mids) {
            assert_eq!(s.label, CLASS_LONG);
        }
        // Big positive move starting at index 20 → samples at indices 10..=19
        // (which look ahead 10 ticks to indices 20..=29) get CLASS_LONG.
        for i in 20..30 {
            mids[i] = 1.10;
        }
        let labeled = build_labeled(&feats, &mids);
        for s in &labeled[10..20] {
            assert_eq!(s.label, CLASS_LONG, "expected long for bullish look-ahead");
        }
        // Big negative move at the tail → bearish samples get CLASS_SHORT.
        for i in 20..30 {
            mids[i] = 0.90;
        }
        let labeled = build_labeled(&feats, &mids);
        for s in &labeled[10..20] {
            assert_eq!(s.label, CLASS_SHORT, "expected short for bearish look-ahead");
        }
        feats[0][0] = 0.1; // silence unused warning
    }
}

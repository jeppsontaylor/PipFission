//! Purged K-Fold cross-validation.
//!
//! For each contiguous fold of the index range, hold it out as the test
//! set, then purge any training sample whose `[t0, t1]` overlaps any
//! test sample's `[t0, t1]`. Apply an embargo of `embargo_pct · N`
//! samples *after* the test fold (purged from training).
//!
//! Inputs:
//!   - `t0`, `t1`: parallel slices, length N, sorted ascending by `t0`.
//!     Each represents a labelled bar's start and label-horizon-end.
//!   - `n_splits`: number of folds.
//!   - `embargo_pct`: fraction of N to embargo after each test fold.
//!
//! Output: `Vec<(train_idx, test_idx)>` of length `n_splits`.

use crate::SplitConfig;

/// Generate the purged K-Fold splits.
pub fn purged_kfold(t0: &[i64], t1: &[i64], cfg: &SplitConfig) -> Vec<(Vec<usize>, Vec<usize>)> {
    assert_eq!(t0.len(), t1.len(), "t0 and t1 must be parallel");
    let n = t0.len();
    if n == 0 || cfg.n_splits == 0 {
        return Vec::new();
    }
    let n_splits = cfg.n_splits.min(n);
    let embargo = (cfg.embargo_pct * n as f64).round() as usize;
    // Sample-count fold boundaries. Last fold absorbs any remainder.
    let fold_size = n / n_splits;
    let mut folds: Vec<(usize, usize)> = Vec::with_capacity(n_splits);
    let mut start = 0;
    for k in 0..n_splits {
        let end = if k + 1 == n_splits { n } else { start + fold_size };
        folds.push((start, end));
        start = end;
    }

    let mut out = Vec::with_capacity(n_splits);
    for &(test_lo, test_hi) in &folds {
        let test_idx: Vec<usize> = (test_lo..test_hi).collect();
        // Purge from train any sample whose [t0, t1] overlaps any test
        // sample's [t0, t1]. Equivalent: train sample i is allowed iff
        // t1[i] < min(t0[test]) OR t0[i] > max(t1[test]).
        let test_t0_min = (test_lo..test_hi)
            .map(|j| t0[j])
            .min()
            .unwrap_or(i64::MIN);
        let test_t1_max = (test_lo..test_hi)
            .map(|j| t1[j])
            .max()
            .unwrap_or(i64::MAX);
        let embargo_lo_excl = test_hi;
        let embargo_hi_excl = (test_hi + embargo).min(n);
        let mut train_idx: Vec<usize> = Vec::with_capacity(n - test_idx.len());
        for i in 0..n {
            // Skip test indices and embargoed samples (post-test only;
            // pre-test purge is handled by the t0/t1 overlap check below).
            if i >= test_lo && i < test_hi {
                continue;
            }
            if i >= embargo_lo_excl && i < embargo_hi_excl {
                continue;
            }
            // Overlap check: keep i in train only if its label horizon
            // does not intersect the test fold's combined horizon.
            let i_t0 = t0[i];
            let i_t1 = t1[i];
            let overlaps = i_t1 >= test_t0_min && i_t0 <= test_t1_max;
            if overlaps {
                continue;
            }
            train_idx.push(i);
        }
        out.push((train_idx, test_idx));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_overlap_between_train_and_test_label_horizons() {
        // 60 samples, label horizon = 5 bars, 5 folds, 1% embargo.
        let n = 60;
        let t0: Vec<i64> = (0..n).map(|i| i as i64 * 10).collect();
        let t1: Vec<i64> = (0..n).map(|i| (i as i64 + 5) * 10).collect();
        let cfg = SplitConfig {
            n_splits: 5,
            embargo_pct: 0.01,
        };
        let splits = purged_kfold(&t0, &t1, &cfg);
        assert_eq!(splits.len(), 5);
        for (train, test) in splits {
            for &i in &train {
                for &j in &test {
                    let train_overlaps_test = t1[i] >= t0[j] && t0[i] <= t1[j];
                    assert!(
                        !train_overlaps_test,
                        "train idx {i} (t0={}, t1={}) overlaps test idx {j} (t0={}, t1={})",
                        t0[i], t1[i], t0[j], t1[j]
                    );
                }
            }
        }
    }

    #[test]
    fn empty_inputs_yield_empty_splits() {
        let cfg = SplitConfig::default();
        let splits = purged_kfold(&[], &[], &cfg);
        assert!(splits.is_empty());
    }
}

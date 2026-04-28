//! Combinatorial Purged Cross-Validation (López de Prado §7.5).
//!
//! Partition the index range into `n_groups` contiguous groups. For
//! every C(n_groups, n_test_groups) combination of test groups, build
//! one (train_idx, test_idx) split using the same overlap-purge +
//! post-test-embargo rules as Purged K-Fold.
//!
//! The combinatorial expansion produces many more (and better-mixed)
//! train/test splits per dataset, which is what the literature shows
//! best mitigates backtest overfitting on financial time series.

use crate::SplitConfig;

/// Run CPCV. `n_test_groups` is the number of groups taken as test in
/// each combination; usually 2.
pub fn combinatorial_purged_cv(
    t0: &[i64],
    t1: &[i64],
    cfg: &SplitConfig,
    n_test_groups: usize,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    assert_eq!(t0.len(), t1.len(), "t0 and t1 must be parallel");
    let n = t0.len();
    if n == 0 || cfg.n_splits == 0 || n_test_groups == 0 || n_test_groups > cfg.n_splits {
        return Vec::new();
    }
    let n_groups = cfg.n_splits.min(n);
    let embargo = (cfg.embargo_pct * n as f64).round() as usize;

    // Group boundaries by sample count.
    let group_size = n / n_groups;
    let mut group_ranges: Vec<(usize, usize)> = Vec::with_capacity(n_groups);
    let mut start = 0;
    for g in 0..n_groups {
        let end = if g + 1 == n_groups { n } else { start + group_size };
        group_ranges.push((start, end));
        start = end;
    }

    let mut splits = Vec::new();
    for combo in combinations(n_groups, n_test_groups) {
        // Build the test index set as the union of the chosen groups, and
        // record each group's [t0_min, t1_max] separately so the overlap
        // check is per-group rather than against the union's bounding
        // box. This matters when the chosen groups are non-contiguous —
        // taking the union's bounds would purge everything between the
        // groups and could leave the training set empty.
        let mut test_idx: Vec<usize> = Vec::new();
        let mut test_group_bounds: Vec<(i64, i64)> = Vec::with_capacity(combo.len());
        for &g in &combo {
            let (lo, hi) = group_ranges[g];
            if lo == hi {
                continue;
            }
            test_idx.extend(lo..hi);
            let t0_min = (lo..hi).map(|j| t0[j]).min().unwrap();
            let t1_max = (lo..hi).map(|j| t1[j]).max().unwrap();
            test_group_bounds.push((t0_min, t1_max));
        }
        if test_idx.is_empty() {
            continue;
        }

        // Embargoes: for each test group, embargo a contiguous suffix.
        let mut embargo_set = std::collections::HashSet::new();
        for &g in &combo {
            let (_lo, hi) = group_ranges[g];
            for i in hi..((hi + embargo).min(n)) {
                embargo_set.insert(i);
            }
        }
        let test_set: std::collections::HashSet<usize> = test_idx.iter().copied().collect();

        let mut train_idx: Vec<usize> = Vec::with_capacity(n - test_idx.len());
        for i in 0..n {
            if test_set.contains(&i) || embargo_set.contains(&i) {
                continue;
            }
            let mut overlaps_any = false;
            for &(g_t0_min, g_t1_max) in &test_group_bounds {
                if t1[i] >= g_t0_min && t0[i] <= g_t1_max {
                    overlaps_any = true;
                    break;
                }
            }
            if overlaps_any {
                continue;
            }
            train_idx.push(i);
        }
        splits.push((train_idx, test_idx));
    }
    splits
}

/// Generate all C(n, k) ascending-order index combinations.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if k == 0 || k > n {
        return out;
    }
    let mut buf = vec![0usize; k];
    fn rec(out: &mut Vec<Vec<usize>>, buf: &mut [usize], pos: usize, start: usize, n: usize) {
        if pos == buf.len() {
            out.push(buf.to_vec());
            return;
        }
        for i in start..=(n - (buf.len() - pos)) {
            buf[pos] = i;
            rec(out, buf, pos + 1, i + 1, n);
        }
    }
    rec(&mut out, &mut buf, 0, 0, n);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combinations_count_matches_binomial() {
        // C(5, 2) = 10, C(6, 3) = 20.
        assert_eq!(combinations(5, 2).len(), 10);
        assert_eq!(combinations(6, 3).len(), 20);
    }

    #[test]
    fn cpcv_produces_expected_split_count() {
        let n = 60;
        let t0: Vec<i64> = (0..n).map(|i| i as i64 * 10).collect();
        let t1: Vec<i64> = (0..n).map(|i| (i as i64 + 3) * 10).collect();
        let cfg = SplitConfig {
            n_splits: 6,
            embargo_pct: 0.01,
        };
        let splits = combinatorial_purged_cv(&t0, &t1, &cfg, 2);
        assert_eq!(splits.len(), 15); // C(6, 2)
        for (train, test) in &splits {
            assert!(!test.is_empty());
            assert!(!train.is_empty());
        }
    }

    #[test]
    fn cpcv_train_does_not_overlap_test_horizons() {
        let n = 60;
        let t0: Vec<i64> = (0..n).map(|i| i as i64 * 10).collect();
        let t1: Vec<i64> = (0..n).map(|i| (i as i64 + 5) * 10).collect();
        let cfg = SplitConfig {
            n_splits: 6,
            embargo_pct: 0.02,
        };
        let splits = combinatorial_purged_cv(&t0, &t1, &cfg, 2);
        for (train, test) in splits {
            for &i in &train {
                for &j in &test {
                    let train_overlaps_test = t1[i] >= t0[j] && t0[i] <= t1[j];
                    assert!(
                        !train_overlaps_test,
                        "leak: train {i} -> test {j}"
                    );
                }
            }
        }
    }
}

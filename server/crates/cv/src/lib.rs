//! Time-series cross-validation for financial ML.
//!
//! Random k-fold leaks across overlapping label horizons. This crate
//! implements two splitters that don't:
//!
//! - **Purged K-Fold** (López de Prado §7.4). Splits the index into K
//!   contiguous folds; for each fold:
//!     * the test fold is held out
//!     * any train sample whose label horizon `[t0, t1]` overlaps a test
//!       label horizon is *purged* (dropped) from training
//!     * an *embargo* of `embargo_pct * N` samples after the test fold
//!       is also dropped, to prevent feature leakage from autocorrelated
//!       returns
//!
//! - **Combinatorial Purged CV** (López de Prado §7.5). All `C(K, k)`
//!   ways of holding out `k` of the K groups, producing many more
//!   train/test splits per dataset.
//!
//! Both splitters take parallel `t0` and `t1` slices (the bar timestamps
//! and the label-horizon-end timestamps from triple-barrier labelling)
//! and return `Vec<(Vec<usize>, Vec<usize>)>` of (train_idx, test_idx).
//! The slices must be sorted ascending by `t0`.

#![deny(unsafe_code)]

pub mod cpcv;
pub mod purged_kfold;

pub use cpcv::combinatorial_purged_cv;
pub use purged_kfold::purged_kfold;

/// Common configuration for both splitters.
#[derive(Clone, Copy, Debug)]
pub struct SplitConfig {
    pub n_splits: usize,
    pub embargo_pct: f64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            n_splits: 6,
            embargo_pct: 0.01,
        }
    }
}

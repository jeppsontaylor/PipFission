"""Purged K-Fold + Combinatorial Purged CV — pure Python.

Mirrors the algorithms in `server/crates/cv/src/`. We keep both
implementations because the Python optimiser path (Optuna inner CV)
must run in-process to share the trained sklearn estimators with the
calibrator. The Rust `cv` crate is the live-engine path (M3, M10).
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterator

import numpy as np


def purged_kfold(
    t0: np.ndarray,
    t1: np.ndarray,
    n_splits: int = 6,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) pairs. `t0` and `t1` must be parallel
    arrays sorted ascending by `t0`."""
    assert t0.shape == t1.shape
    n = len(t0)
    if n == 0 or n_splits == 0:
        return []
    n_splits = min(n_splits, n)
    embargo = int(round(embargo_pct * n))
    fold_size = n // n_splits

    folds: list[tuple[int, int]] = []
    start = 0
    for k in range(n_splits):
        end = n if k + 1 == n_splits else start + fold_size
        folds.append((start, end))
        start = end

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for test_lo, test_hi in folds:
        test_idx = np.arange(test_lo, test_hi)
        if test_idx.size == 0:
            continue
        test_t0_min = int(t0[test_lo:test_hi].min())
        test_t1_max = int(t1[test_lo:test_hi].max())
        emb_lo, emb_hi = test_hi, min(test_hi + embargo, n)

        keep = np.ones(n, dtype=bool)
        keep[test_lo:test_hi] = False
        keep[emb_lo:emb_hi] = False
        # Overlap purge.
        overlap = (t1 >= test_t0_min) & (t0 <= test_t1_max)
        keep[overlap] = False
        train_idx = np.where(keep)[0]
        out.append((train_idx, test_idx))
    return out


def combinatorial_purged_cv(
    t0: np.ndarray,
    t1: np.ndarray,
    n_splits: int = 6,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """C(n_splits, n_test_groups) splits with per-group overlap purging
    (so non-contiguous test groups don't accidentally purge the entire
    middle of the dataset).
    """
    assert t0.shape == t1.shape
    n = len(t0)
    if n == 0 or n_splits == 0 or n_test_groups == 0 or n_test_groups > n_splits:
        return []
    n_groups = min(n_splits, n)
    embargo = int(round(embargo_pct * n))

    group_size = n // n_groups
    groups: list[tuple[int, int]] = []
    start = 0
    for g in range(n_groups):
        end = n if g + 1 == n_groups else start + group_size
        groups.append((start, end))
        start = end

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for combo in combinations(range(n_groups), n_test_groups):
        test_idx_parts: list[np.ndarray] = []
        bounds: list[tuple[int, int]] = []
        for g in combo:
            lo, hi = groups[g]
            if lo == hi:
                continue
            test_idx_parts.append(np.arange(lo, hi))
            bounds.append((int(t0[lo:hi].min()), int(t1[lo:hi].max())))
        if not test_idx_parts:
            continue
        test_idx = np.concatenate(test_idx_parts)

        keep = np.ones(n, dtype=bool)
        keep[test_idx] = False
        for g in combo:
            lo, hi = groups[g]
            for i in range(hi, min(hi + embargo, n)):
                keep[i] = False
        # Per-group overlap purge.
        for t0_min, t1_max in bounds:
            overlap = (t1 >= t0_min) & (t0 <= t1_max)
            keep[overlap] = False
        train_idx = np.where(keep)[0]
        out.append((train_idx, test_idx))
    return out

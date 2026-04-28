"""Probability of Backtest Overfitting via Combinatorially Symmetric
Cross-Validation (Bailey, Borwein, López de Prado, Zhu, 2014).

Given a matrix `M` of shape `(n_periods, n_strategies)` of returns,
the CSCV procedure splits the rows into S equal blocks, takes every
combination of S/2 blocks for in-sample / S/2 for out-of-sample,
ranks each strategy in-sample, and reports how often the in-sample
champion is below median out-of-sample.

PBO ∈ [0, 1]: 0 = no overfitting; 0.5 = no skill; > 0.5 = champion
strategies tend to be over-fit and underperform OOS.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def probability_of_backtest_overfitting(returns: np.ndarray, n_blocks: int = 16) -> float:
    """`returns` shape: (n_periods, n_strategies). Each column is one
    candidate strategy's per-period returns; columns must be aligned.
    """
    if returns.ndim != 2:
        raise ValueError("returns must be 2-D (n_periods, n_strategies)")
    n_periods, n_strats = returns.shape
    if n_strats < 2 or n_periods < n_blocks * 2:
        return 0.5
    if n_blocks % 2 != 0:
        n_blocks -= 1
    block_size = n_periods // n_blocks
    if block_size == 0:
        return 0.5
    # Trim to exact block boundary.
    trimmed = returns[: block_size * n_blocks]
    blocks = [trimmed[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
    half = n_blocks // 2
    overfits = 0
    n_combos = 0
    for combo in combinations(range(n_blocks), half):
        is_idx = list(combo)
        oos_idx = [i for i in range(n_blocks) if i not in combo]
        is_block = np.vstack([blocks[i] for i in is_idx])
        oos_block = np.vstack([blocks[i] for i in oos_idx])
        # Sharpe per strategy on each block.
        is_sr = is_block.mean(axis=0) / (is_block.std(axis=0, ddof=1) + 1e-12)
        oos_sr = oos_block.mean(axis=0) / (oos_block.std(axis=0, ddof=1) + 1e-12)
        # Rank-of-the-IS-best in OOS.
        best = int(np.argmax(is_sr))
        ranks_oos = (oos_sr.argsort().argsort()[best]) / (n_strats - 1)
        # Convert to logit-rank: > 0.5 means below median in OOS.
        logit = np.log(max(ranks_oos, 1e-9) / max(1 - ranks_oos, 1e-9))
        if logit < 0:
            overfits += 1
        n_combos += 1
    return overfits / max(1, n_combos)

"""Sanity tests for DSR + PBO."""
from __future__ import annotations

import numpy as np

from research.stats.deflated_sharpe import deflated_sharpe, deflated_sharpe_from_returns
from research.stats.pbo import probability_of_backtest_overfitting


def test_dsr_in_unit_interval():
    v = deflated_sharpe(1.5, 252, 10, 0.0, 3.0)
    assert 0.0 <= v <= 1.0


def test_dsr_decreases_with_more_trials():
    a = deflated_sharpe(1.5, 252, 1, 0.0, 3.0)
    b = deflated_sharpe(1.5, 252, 100, 0.0, 3.0)
    assert b < a


def test_dsr_from_returns_runs():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, 500)
    v = deflated_sharpe_from_returns(rets, n_trials=10)
    assert 0.0 <= v <= 1.0


def test_pbo_random_strategies_are_no_skill():
    # 100 random-walk strategies → PBO should hover around 0.5.
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 1, (320, 100))
    pbo = probability_of_backtest_overfitting(rets, n_blocks=16)
    assert 0.3 <= pbo <= 0.7

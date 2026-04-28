"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

Mirrors the Rust implementation in `metrics::deflated`. Used by the
research reports and by the optimiser's tiebreaker scoring.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def deflated_sharpe(
    sr: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Returns DSR ∈ [0, 1] — calibrated p-value that the true Sharpe
    is positive after correcting for selection bias from `n_trials`.
    """
    if n_obs < 2 or n_trials < 1:
        return 0.0
    if n_trials > 1:
        z = stats.norm.ppf(1.0 - 1.0 / n_trials)
        gamma = 0.5772156649  # Euler-Mascheroni
        expected_max_sr_h0 = (1.0 - gamma) * z + gamma * stats.norm.ppf(
            1.0 - 1.0 / (n_trials * math.e)
        )
    else:
        expected_max_sr_h0 = 0.0
    var_correction = (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr) / (n_obs - 1)
    if var_correction <= 0.0:
        return 0.0
    return float(stats.norm.cdf((sr - expected_max_sr_h0) / math.sqrt(var_correction)))


def deflated_sharpe_from_returns(
    returns: np.ndarray, n_trials: int, n_periods_per_year: float = 252.0
) -> float:
    """Convenience: compute Sharpe / skew / kurt from a return series and
    plug into the DSR formula."""
    if returns.size < 2:
        return 0.0
    mean = returns.mean()
    sd = returns.std(ddof=1)
    if sd <= 0.0:
        return 0.0
    sr = mean / sd * math.sqrt(n_periods_per_year)
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))
    return deflated_sharpe(sr, returns.size, n_trials, skew, kurt)

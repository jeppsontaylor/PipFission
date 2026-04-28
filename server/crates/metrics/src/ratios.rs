//! Risk-adjusted return ratios. All functions take a per-period (per-trade
//! or per-bar) return slice and an annualisation factor, and return a
//! scalar. Empty / single-element inputs return 0 to avoid pathological
//! division by zero in the optimiser.

/// Annualised Sharpe ratio. `n_periods_per_year` should be the number of
/// independent return observations per year — for per-trade returns this
/// is harder to estimate; pass `1.0` to get unannualised Sharpe.
pub fn sharpe(returns: &[f64], n_periods_per_year: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let var = returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / (returns.len() - 1) as f64;
    let sd = var.sqrt();
    if sd <= 0.0 {
        return 0.0;
    }
    (mean / sd) * n_periods_per_year.sqrt()
}

/// Annualised Sortino ratio. Downside-only standard deviation in the
/// denominator. Returns 0 if no negative returns exist (degenerate;
/// caller treats this as "noise").
pub fn sortino(returns: &[f64], n_periods_per_year: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let downside_var = returns
        .iter()
        .filter(|r| **r < 0.0)
        .map(|r| r.powi(2))
        .sum::<f64>()
        / returns.len() as f64;
    let dsd = downside_var.sqrt();
    if dsd <= 0.0 {
        return 0.0;
    }
    (mean / dsd) * n_periods_per_year.sqrt()
}

/// Calmar = Sharpe / |max drawdown|. Drawdown is in basis points; we
/// scale to a fraction internally so the ratio stays dimensionless.
pub fn calmar(sharpe_value: f64, max_dd_bp: f64) -> f64 {
    let dd_frac = (max_dd_bp / 10_000.0).abs();
    if dd_frac <= 0.0 {
        return 0.0;
    }
    sharpe_value / dd_frac
}

/// Profit factor = sum(positive returns) / |sum(negative returns)|.
/// Returns `f64::INFINITY` when no losses exist; callers should clamp.
pub fn profit_factor(returns: &[f64]) -> f64 {
    let (gross_win, gross_loss): (f64, f64) = returns.iter().fold((0.0, 0.0), |(w, l), &r| {
        if r > 0.0 {
            (w + r, l)
        } else {
            (w, l + r)
        }
    });
    if gross_loss == 0.0 {
        if gross_win == 0.0 {
            return 0.0;
        }
        return f64::INFINITY;
    }
    gross_win / gross_loss.abs()
}

/// Fraction of trades that had a strictly positive return.
pub fn hit_rate(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    returns.iter().filter(|r| **r > 0.0).count() as f64 / returns.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharpe_zero_for_empty() {
        assert_eq!(sharpe(&[], 1.0), 0.0);
        assert_eq!(sharpe(&[0.5], 1.0), 0.0);
    }

    #[test]
    fn sortino_zero_with_no_downside() {
        // All positive returns: downside vol = 0, return 0 (degenerate).
        assert_eq!(sortino(&[0.01, 0.02, 0.03, 0.04], 1.0), 0.0);
    }

    #[test]
    fn profit_factor_handles_only_wins() {
        assert!(profit_factor(&[0.01, 0.02]).is_infinite());
    }

    #[test]
    fn hit_rate_basic() {
        assert!((hit_rate(&[0.01, -0.02, 0.005]) - 2.0 / 3.0).abs() < 1e-9);
    }
}

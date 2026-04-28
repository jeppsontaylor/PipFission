//! Deflated Sharpe Ratio (Bailey & López de Prado, 2014).
//!
//! Adjusts an observed Sharpe for selection bias from running many
//! trials, plus skew/kurtosis of the return series. Implemented as the
//! probability that the *true* Sharpe exceeds zero, given:
//!   - the observed Sharpe `sr`
//!   - the number of return observations `n`
//!   - the number of trials run during the search `n_trials`
//!   - the return series' skewness `skew` and kurtosis `kurt` (4 ≈ normal)
//!
//! The DSR is in [0, 1] and behaves as a calibrated p-value: 0.95 means
//! "95% confidence the true Sharpe is positive after correcting for the
//! search". The Python research layer reports both the raw Sharpe and
//! the DSR; the optimiser uses DSR as a tiebreaker.

use std::f64::consts::{E, PI};

/// Standard-normal CDF via the error-function approximation in
/// Abramowitz & Stegun 26.2.17. Sufficient precision for ranking
/// trader configurations.
fn norm_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / 2.0_f64.sqrt();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0
        - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    0.5 * (1.0 + sign * y)
}

/// Inverse standard-normal CDF, Beasley-Springer-Moro approximation.
/// Used to derive the expected maximum Sharpe across `n_trials`.
fn norm_ppf(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
}

/// Deflated Sharpe Ratio. `sr` is the observed Sharpe; `n` is the number
/// of returns; `n_trials` is the number of distinct strategies tried in
/// search (e.g. number of Optuna trials, or grid-search points); `skew`
/// is the third moment of returns; `kurt` is the fourth moment (3 = Gaussian).
pub fn deflated_sharpe(sr: f64, n: usize, n_trials: usize, skew: f64, kurt: f64) -> f64 {
    if n < 2 || n_trials == 0 {
        return 0.0;
    }
    // Expected maximum Sharpe under H0 (true SR=0) across n_trials.
    let euler_mascheroni = 0.5772156649;
    let z = norm_ppf(1.0 - 1.0 / n_trials as f64);
    let expected_max_sr_h0 = if n_trials > 1 {
        (1.0 - euler_mascheroni) * z
            + euler_mascheroni * norm_ppf(1.0 - 1.0 / (n_trials as f64 * E))
    } else {
        0.0
    };
    // SR variance correction for higher moments.
    let var_correction =
        (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr) / (n as f64 - 1.0);
    if var_correction <= 0.0 {
        return 0.0;
    }
    let denom = var_correction.sqrt();
    if denom <= 0.0 {
        return 0.0;
    }
    let _ = PI; // kept for any future closed-form refinements.
    norm_cdf((sr - expected_max_sr_h0) / denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsr_in_unit_interval() {
        let v = deflated_sharpe(1.5, 252, 10, 0.0, 3.0);
        assert!((0.0..=1.0).contains(&v), "DSR out of [0,1]: {v}");
    }

    #[test]
    fn dsr_drops_when_more_trials_run() {
        let dsr_1 = deflated_sharpe(1.5, 252, 1, 0.0, 3.0);
        let dsr_100 = deflated_sharpe(1.5, 252, 100, 0.0, 3.0);
        assert!(dsr_100 < dsr_1);
    }

    #[test]
    fn dsr_zero_for_short_series() {
        assert_eq!(deflated_sharpe(2.0, 1, 1, 0.0, 3.0), 0.0);
    }
}

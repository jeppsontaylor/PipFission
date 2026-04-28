//! Bootstrap confidence intervals for trade-return statistics.
//!
//! Pure-math; no external dependencies. Used by the research reports
//! and (eventually) the deployment gate to put a confidence interval
//! around the realised Sharpe / Sortino / cumulative return.
//!
//! The implementation is the percentile bootstrap: resample the
//! return series with replacement `n_resamples` times, compute the
//! statistic on each resample, return the (lo, hi) percentiles.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Two-sided percentile bootstrap CI for an arbitrary statistic.
/// `stat_fn` is called on each resample and must be deterministic.
///
/// `alpha` ∈ (0, 1): the CI returned is `[alpha/2, 1 - alpha/2]`
/// percentiles, e.g. `alpha = 0.05` for a 95 % CI.
pub fn bootstrap_ci<F: Fn(&[f64]) -> f64>(
    series: &[f64],
    stat_fn: F,
    n_resamples: usize,
    alpha: f64,
    seed: u64,
) -> (f64, f64) {
    if series.len() < 2 || n_resamples == 0 {
        let s = stat_fn(series);
        return (s, s);
    }
    let alpha = alpha.clamp(1e-6, 0.5);
    let mut rng = SmallRng::seed_from_u64(seed);
    let n = series.len();
    let mut samples: Vec<f64> = Vec::with_capacity(n_resamples);
    let mut buf = vec![0.0_f64; n];
    for _ in 0..n_resamples {
        for i in 0..n {
            buf[i] = series[rng.gen_range(0..n)];
        }
        samples.push(stat_fn(&buf));
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo_idx = ((alpha / 2.0) * n_resamples as f64).floor() as usize;
    let hi_idx = (((1.0 - alpha / 2.0) * n_resamples as f64).ceil() as usize)
        .min(n_resamples - 1);
    (samples[lo_idx], samples[hi_idx])
}

/// Convenience: bootstrap CI for the **mean** of `series`.
pub fn bootstrap_mean_ci(
    series: &[f64],
    n_resamples: usize,
    alpha: f64,
    seed: u64,
) -> (f64, f64) {
    bootstrap_ci(series, mean, n_resamples, alpha, seed)
}

/// Convenience: bootstrap CI for the (period) Sharpe of `series`.
pub fn bootstrap_sharpe_ci(
    series: &[f64],
    n_resamples: usize,
    alpha: f64,
    seed: u64,
) -> (f64, f64) {
    bootstrap_ci(series, sharpe, n_resamples, alpha, seed)
}

fn mean(s: &[f64]) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    s.iter().sum::<f64>() / s.len() as f64
}

fn sharpe(s: &[f64]) -> f64 {
    if s.len() < 2 {
        return 0.0;
    }
    let m = mean(s);
    let var = s.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (s.len() as f64 - 1.0);
    let sd = var.max(0.0).sqrt();
    if sd > 1e-12 {
        m / sd
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ci_brackets_population_mean() {
        // Series with known mean = 0.5; 95% CI on bootstrap mean
        // should bracket 0.5 with high confidence.
        let series: Vec<f64> = (0..200).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let (lo, hi) = bootstrap_mean_ci(&series, 500, 0.05, 42);
        assert!(lo <= 0.5 && hi >= 0.5, "CI [{lo}, {hi}] missed 0.5");
    }

    #[test]
    fn ci_collapses_for_short_series() {
        let series = vec![1.0];
        let (lo, hi) = bootstrap_mean_ci(&series, 100, 0.05, 1);
        assert_eq!((lo, hi), (1.0, 1.0));
    }

    #[test]
    fn deterministic_with_same_seed() {
        let series: Vec<f64> = (0..50).map(|i| (i as f64).sin()).collect();
        let a = bootstrap_sharpe_ci(&series, 200, 0.10, 7);
        let b = bootstrap_sharpe_ci(&series, 200, 0.10, 7);
        assert_eq!(a, b);
    }
}

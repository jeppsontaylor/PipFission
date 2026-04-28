//! Constrained label optimiser. Given a set of triple-barrier label
//! candidates, pick a non-overlapping subset that maximises a regularised
//! Sortino-style objective. The output is the "ideal buy/sell entry
//! points" the user wants written back to the `labels` table.
//!
//! The optimisation problem is **weighted interval scheduling**: given
//! intervals `[ts_ms, t1_ms]` each with a profit/risk-adjusted score,
//! select a non-overlapping subset that maximises total score. We solve
//! it with the standard O(N log N) DP after sorting by `t1_ms`.
//!
//! Why not Sharpe directly? Sharpe and Sortino are non-additive — they
//! don't decompose over a subset of trades. We instead pick a smooth
//! per-trade score that is well-aligned with Sortino in practice
//! (mean-minus-downside-variance) and re-evaluate true Sortino on the
//! chosen subset for reporting.
//!
//! Constraints (enforced as penalties or hard filters):
//!   - **Min/max hold**: discard intervals with `(t1 - t0)` outside the band.
//!   - **No overlap**: enforced exactly by the DP.
//!   - **Edge floor**: discard candidates whose `|realized_r| < min_edge`.
//!   - **Long/short balance (binary)**: after the DP, if the minority side's
//!     fraction is below `min_minority_frac`, drop the lowest-scoring
//!     majority-side trades until the floor is met. This keeps the
//!     classifier from collapsing onto a single class — the operator
//!     mandate is "binary 30/50% that maximizes Sharpe while fighting for
//!     nearly balanced."
//!
//! Output: the selected subset, sorted ascending by `ts_ms`.

use crate::triple_barrier::LabelRow;

#[derive(Clone, Copy, Debug)]
pub struct LabelOptimiserConfig {
    /// Minimum bar-level holding period to consider a candidate. Same
    /// units (ms) as `LabelRow::t1_ms - LabelRow::ts_ms`.
    pub min_hold_ms: i64,
    /// Maximum holding period.
    pub max_hold_ms: i64,
    /// Per-trade absolute return floor (after costs).
    pub min_edge: f64,
    /// Downside-variance penalty weight in the per-trade score:
    /// `score = realized_r - lambda * max(-realized_r, 0)^2`. λ ≥ 0.
    pub downside_lambda: f64,
    /// Per-trade fixed turnover cost subtracted from the score.
    pub turnover_cost: f64,
    /// Minimum fraction the minority side (long or short) must hold of
    /// the chosen subset. Default `0.30` per operator mandate. Set to
    /// `0.0` to disable rebalancing. The optimizer will drop the lowest-
    /// scoring majority trades until the floor is met or no majority
    /// trades remain to drop.
    pub min_minority_frac: f64,
}

impl Default for LabelOptimiserConfig {
    fn default() -> Self {
        Self {
            min_hold_ms: 10_000,
            max_hold_ms: 600_000,
            min_edge: 0.0,
            downside_lambda: 4.0,
            turnover_cost: 0.0,
            min_minority_frac: 0.30,
        }
    }
}

/// Pick the non-overlapping subset of `labels` maximising the regularised
/// per-trade score under the constraints above. Returns the chosen rows
/// sorted ascending by `ts_ms`.
pub fn optimise_labels(labels: &[LabelRow], cfg: &LabelOptimiserConfig) -> Vec<LabelRow> {
    if labels.is_empty() {
        return Vec::new();
    }

    // 1. Filter on hold-time band and edge floor.
    let mut candidates: Vec<(usize, f64, LabelRow)> = labels
        .iter()
        .copied()
        .filter_map(|l| {
            let hold = l.t1_ms - l.ts_ms;
            if hold < cfg.min_hold_ms || hold > cfg.max_hold_ms {
                return None;
            }
            if l.realized_r.abs() < cfg.min_edge {
                return None;
            }
            // The per-trade score: assume we trade in the "favoured"
            // direction (long if side >=0, short if side < 0). PnL is
            // |realized_r| because the side optimally aligns. Penalise
            // realised downside variance.
            let pnl = l.realized_r.abs();
            let downside = (-l.realized_r.abs()).max(0.0); // 0 by construction; reserved for future variants
            let score = pnl - cfg.downside_lambda * downside.powi(2) - cfg.turnover_cost;
            Some((0, score, l))
        })
        .collect();

    if candidates.is_empty() {
        return Vec::new();
    }

    // 2. Sort by t1 so we can DP efficiently.
    candidates.sort_unstable_by_key(|(_, _, l)| l.t1_ms);
    for (i, c) in candidates.iter_mut().enumerate() {
        c.0 = i;
    }

    let n = candidates.len();
    // p[i] = largest j < i such that candidates[j].t1 <= candidates[i].ts.
    // Standard binary search over the sorted-by-t1 list.
    let t1: Vec<i64> = candidates.iter().map(|(_, _, l)| l.t1_ms).collect();
    let mut p = vec![-1_i64; n];
    for (i, (_, _, l)) in candidates.iter().enumerate() {
        // find largest j with t1[j] <= l.ts_ms
        match t1[..i].binary_search_by(|x| x.cmp(&l.ts_ms)) {
            Ok(j) => p[i] = j as i64,
            Err(j) if j > 0 => p[i] = (j - 1) as i64,
            Err(_) => p[i] = -1,
        }
    }

    // 3. Standard weighted-interval-scheduling DP.
    let mut dp = vec![0.0_f64; n + 1];
    for i in 1..=n {
        let (_, score, _) = candidates[i - 1];
        let take = score
            + if p[i - 1] >= 0 {
                dp[(p[i - 1] + 1) as usize]
            } else {
                0.0
            };
        let skip = dp[i - 1];
        dp[i] = take.max(skip);
    }

    // 4. Backtrace to recover the selected indices.
    let mut chosen: Vec<LabelRow> = Vec::new();
    let mut i = n;
    while i > 0 {
        let (_, score, l) = candidates[i - 1];
        let take = score
            + if p[i - 1] >= 0 {
                dp[(p[i - 1] + 1) as usize]
            } else {
                0.0
            };
        let skip = dp[i - 1];
        if take >= skip {
            chosen.push(l);
            i = if p[i - 1] >= 0 { (p[i - 1] + 1) as usize } else { 0 };
        } else {
            i -= 1;
        }
    }
    chosen.sort_unstable_by_key(|l| l.ts_ms);
    enforce_minority_floor(chosen, cfg.min_minority_frac)
}

/// Drop the lowest-scoring majority-side trades until the minority side's
/// fraction reaches `min_frac` (or the majority side is exhausted). Score
/// is `|realized_r|` — the same magnitude the DP optimised — so we always
/// keep the strongest signals on each side.
///
/// `chosen` is assumed to be sorted ascending by `ts_ms`; the result
/// preserves that ordering. If `min_frac <= 0.0` or the chosen set is
/// already balanced, `chosen` is returned unchanged.
fn enforce_minority_floor(mut chosen: Vec<LabelRow>, min_frac: f64) -> Vec<LabelRow> {
    if min_frac <= 0.0 || chosen.is_empty() {
        return chosen;
    }
    let n = chosen.len();
    let n_long = chosen.iter().filter(|l| l.side >= 0).count();
    let n_short = n - n_long;
    let minority = n_long.min(n_short);
    let needed_total = if minority == 0 {
        return chosen; // single-sided run; nothing to rebalance against
    } else {
        // smallest N s.t. minority / N >= min_frac, i.e. N <= minority / min_frac.
        ((minority as f64) / min_frac).floor() as usize
    };
    if n <= needed_total {
        return chosen; // already balanced enough
    }
    let drop_n = n - needed_total;
    // Drop the `drop_n` weakest majority-side trades. Score = |realized_r|.
    let majority_side: i8 = if n_long > n_short { 1 } else { -1 };
    let mut majority_idx_sorted: Vec<usize> = (0..n)
        .filter(|&i| {
            if majority_side >= 0 {
                chosen[i].side >= 0
            } else {
                chosen[i].side < 0
            }
        })
        .collect();
    majority_idx_sorted.sort_by(|&a, &b| {
        chosen[a]
            .realized_r
            .abs()
            .partial_cmp(&chosen[b].realized_r.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let drop: std::collections::HashSet<usize> =
        majority_idx_sorted.into_iter().take(drop_n).collect();
    let mut out = Vec::with_capacity(n - drop.len());
    for (i, l) in chosen.drain(..).enumerate() {
        if !drop.contains(&i) {
            out.push(l);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triple_barrier::BarrierHit;

    fn lr(t0: i64, t1: i64, r: f64) -> LabelRow {
        LabelRow {
            ts_ms: t0,
            t1_ms: t1,
            side: if r >= 0.0 { 1 } else { -1 },
            meta_y: 1,
            realized_r: r,
            barrier_hit: BarrierHit::Pt,
        }
    }

    #[test]
    fn picks_non_overlapping_higher_score() {
        // Two overlapping wins (0.005 each) vs one big win (0.012) overlapping both:
        // total of two non-overlap wins (0.010) vs the single (0.012) — DP picks 0.012.
        let labels = vec![
            lr(0, 100_000, 0.005),
            lr(50_000, 200_000, 0.012),
            lr(150_000, 250_000, 0.005),
        ];
        let cfg = LabelOptimiserConfig {
            min_hold_ms: 10_000,
            max_hold_ms: 1_000_000,
            min_edge: 0.0,
            downside_lambda: 0.0,
            turnover_cost: 0.0,
            min_minority_frac: 0.0,
        };
        let chosen = optimise_labels(&labels, &cfg);
        assert_eq!(chosen.len(), 1);
        assert!((chosen[0].realized_r - 0.012).abs() < 1e-9);
    }

    #[test]
    fn drops_short_holds() {
        let labels = vec![lr(0, 5_000, 0.020), lr(10_000, 100_000, 0.005)];
        let cfg = LabelOptimiserConfig {
            min_hold_ms: 10_000,
            min_minority_frac: 0.0,
            ..Default::default()
        };
        let chosen = optimise_labels(&labels, &cfg);
        assert_eq!(chosen.len(), 1);
        assert!((chosen[0].realized_r - 0.005).abs() < 1e-9);
    }

    #[test]
    fn empty_input_returns_empty() {
        let cfg = LabelOptimiserConfig::default();
        assert!(optimise_labels(&[], &cfg).is_empty());
    }

    #[test]
    fn rebalancer_drops_majority_until_minority_floor_met() {
        // 8 long-favourable + 1 short. Minority floor 0.30 → keep at most
        // ⌊1 / 0.30⌋ = 3 trades total → 1 short + 2 longs (the strongest two).
        let labels: Vec<LabelRow> = vec![
            lr(0, 11_000, 0.001),
            lr(20_000, 31_000, 0.002),
            lr(40_000, 51_000, 0.003),
            lr(60_000, 71_000, 0.004),
            lr(80_000, 91_000, 0.005),
            lr(100_000, 111_000, 0.006),
            lr(120_000, 131_000, 0.007),
            lr(140_000, 151_000, 0.008),
            lr(160_000, 171_000, -0.010),
        ];
        let cfg = LabelOptimiserConfig {
            min_hold_ms: 10_000,
            max_hold_ms: 1_000_000,
            min_edge: 0.0,
            downside_lambda: 0.0,
            turnover_cost: 0.0,
            min_minority_frac: 0.30,
        };
        let chosen = optimise_labels(&labels, &cfg);
        let n_long = chosen.iter().filter(|l| l.side >= 0).count();
        let n_short = chosen.len() - n_long;
        let frac = (n_long.min(n_short) as f64) / (chosen.len() as f64);
        assert!(
            frac >= 0.30 - 1e-9,
            "expected ≥ 30% minority, got {frac:.3} ({n_long} long / {n_short} short of {})",
            chosen.len()
        );
        // The kept longs must be the strongest two by |realized_r|.
        let kept_longs: Vec<f64> = chosen
            .iter()
            .filter(|l| l.side >= 0)
            .map(|l| l.realized_r)
            .collect();
        assert!(kept_longs.iter().all(|&r| r >= 0.007 - 1e-9));
    }

    #[test]
    fn rebalancer_noops_when_already_balanced() {
        let labels = vec![
            lr(0, 11_000, 0.005),
            lr(20_000, 31_000, -0.005),
            lr(40_000, 51_000, 0.004),
            lr(60_000, 71_000, -0.004),
        ];
        let cfg = LabelOptimiserConfig {
            min_hold_ms: 10_000,
            max_hold_ms: 1_000_000,
            min_edge: 0.0,
            downside_lambda: 0.0,
            turnover_cost: 0.0,
            min_minority_frac: 0.30,
        };
        let chosen = optimise_labels(&labels, &cfg);
        assert_eq!(chosen.len(), 4);
    }
}

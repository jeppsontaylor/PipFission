//! Bar-replay loop. Walks `bars` in order, feeds each closed bar to the
//! `Trader`, applies `Costs` on entry/exit fills, and accumulates an
//! equity curve + per-trade ledger. Returns a `Report` carrying the
//! `metrics::Summary` so the optimiser only needs scalar comparisons.

use serde::{Deserialize, Serialize};

use market_domain::Bar10s;
use metrics::Summary;
use trader::{Probs, Reason, Side, State, TradeEvent, Trader, TraderParams};

use crate::fill::Costs;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TradeRecord {
    pub side: Side,
    pub entry_idx: u32,
    pub exit_idx: u32,
    pub entry_ts_ms: i64,
    pub exit_ts_ms: i64,
    pub entry_px: f64,
    pub exit_px: f64,
    pub gross_r: f64,
    pub net_r: f64,
    pub exit_reason: Reason,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub summary: Summary,
    pub ledger: Vec<TradeRecord>,
    pub equity: Vec<f64>,
    /// Average bar-level OOF entropy across the run (diagnostic).
    pub avg_entropy: f64,
}

/// 10s × 8640 / day × 5 trading days = ~43_200 ten-second bars per
/// trading week of 24/5 forex; 252 weeks/year.
const BARS_PER_YEAR_24_5: f64 = 43_200.0 * 252.0 / 5.0;

/// Run the backtest. Lengths must match: `bars`, `probs`, `sigma` are
/// parallel slices of the same length.
pub fn run_backtest(
    bars: &[Bar10s],
    probs: &[Probs],
    sigma: &[f64],
    params: TraderParams,
    costs: Costs,
) -> Report {
    assert_eq!(bars.len(), probs.len(), "bars/probs length mismatch");
    assert_eq!(bars.len(), sigma.len(), "bars/sigma length mismatch");
    let n = bars.len();
    let mut tr = Trader::new(params);
    let mut ledger: Vec<TradeRecord> = Vec::new();
    let mut equity: Vec<f64> = Vec::with_capacity(n);
    let mut acct: f64 = 1.0;
    let cost_per_round = costs.round_trip_frac();
    let mut entry_idx: Option<u32> = None;
    let mut entry_px: f64 = 0.0;
    let mut entry_side: Option<Side> = None;
    let mut entry_ts_ms: i64 = 0;
    let mut total_entropy = 0.0_f64;

    for i in 0..n {
        let bar = &bars[i];
        let p = &probs[i];
        let s = sigma[i];
        // Entropy diagnostic — flat probabilities mean the model has no edge.
        let q = [p.p_long.max(1e-9), p.p_short.max(1e-9), p.p_take.max(1e-9)];
        total_entropy += -q.iter().map(|x| x * x.ln()).sum::<f64>();

        let event = tr.on_bar(i as u32, bar, p, s, 0);
        match event {
            TradeEvent::Open { side, bar_idx, entry_px: ex, .. } => {
                entry_idx = Some(bar_idx);
                entry_px = ex;
                entry_side = Some(side);
                entry_ts_ms = bar.ts_ms;
            }
            TradeEvent::Close {
                bar_idx,
                exit_px,
                realized_r,
                reason,
            } => {
                if let (Some(ei), Some(side)) = (entry_idx, entry_side) {
                    // realized_r is gross; apply round-trip cost net of fills.
                    let net = realized_r - cost_per_round;
                    acct *= 1.0 + net;
                    ledger.push(TradeRecord {
                        side,
                        entry_idx: ei,
                        exit_idx: bar_idx,
                        entry_ts_ms,
                        exit_ts_ms: bar.ts_ms,
                        entry_px,
                        exit_px,
                        gross_r: realized_r,
                        net_r: net,
                        exit_reason: reason,
                    });
                    entry_idx = None;
                    entry_side = None;
                }
            }
            TradeEvent::Skip { .. } => {}
        }
        equity.push(acct);
    }

    // Force-close any open position at the last bar at close price.
    if let (Some(_), Some(side)) = (entry_idx, entry_side) {
        let last = bars.last().expect("non-empty bars");
        let exit_px = last.close;
        let gross_r = match side {
            Side::Long => exit_px / entry_px - 1.0,
            Side::Short => entry_px / exit_px - 1.0,
        };
        let net = gross_r - cost_per_round;
        acct *= 1.0 + net;
        ledger.push(TradeRecord {
            side,
            entry_idx: entry_idx.unwrap(),
            exit_idx: (n - 1) as u32,
            entry_ts_ms,
            exit_ts_ms: last.ts_ms,
            entry_px,
            exit_px,
            gross_r,
            net_r: net,
            exit_reason: Reason::MaxHold,
        });
        // Update the last point of the equity curve to reflect close.
        if let Some(last_eq) = equity.last_mut() {
            *last_eq = acct;
        }
    }

    let per_trade: Vec<f64> = ledger.iter().map(|t| t.net_r).collect();
    let elapsed_days = if bars.len() >= 2 {
        let span_ms = (bars.last().unwrap().ts_ms - bars.first().unwrap().ts_ms).max(1);
        span_ms as f64 / 86_400_000.0
    } else {
        1.0 / 1440.0
    };
    let summary = Summary::from_trade_returns(&per_trade, &equity, BARS_PER_YEAR_24_5, elapsed_days);
    let _ = State::Flat; // keep the import live for the public API surface
    Report {
        summary,
        ledger,
        equity,
        avg_entropy: if n == 0 { 0.0 } else { total_entropy / n as f64 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rising_bars(n: usize) -> Vec<Bar10s> {
        (0..n)
            .map(|i| {
                let p = 1.0 + (i as f64) * 0.0005;
                Bar10s {
                    instrument_id: 0,
                    ts_ms: i as i64 * 10_000,
                    open: p,
                    high: p + 0.0001,
                    low: p - 0.0001,
                    close: p,
                    n_ticks: 1,
                    spread_bp_avg: 0.5,
                }
            })
            .collect()
    }

    #[test]
    fn determinism_two_runs_match_byte_for_byte() {
        let bars = rising_bars(200);
        let probs: Vec<Probs> = (0..bars.len())
            .map(|i| Probs {
                p_long: 0.7 + (i as f64 * 0.001).sin() * 0.05,
                p_short: 0.2,
                p_take: 0.8,
                calibrated: 0.7,
            })
            .collect();
        let sigma = vec![0.005; bars.len()];
        let params = TraderParams::default();
        let costs = Costs::default();
        let r1 = run_backtest(&bars, &probs, &sigma, params, costs);
        let r2 = run_backtest(&bars, &probs, &sigma, params, costs);
        assert_eq!(r1.ledger.len(), r2.ledger.len());
        for (a, b) in r1.ledger.iter().zip(r2.ledger.iter()) {
            assert_eq!(a.entry_idx, b.entry_idx);
            assert_eq!(a.exit_idx, b.exit_idx);
            assert!((a.net_r - b.net_r).abs() < 1e-15);
        }
        assert!((r1.summary.net_return - r2.summary.net_return).abs() < 1e-12);
    }

    #[test]
    fn rising_market_with_long_signal_makes_money() {
        let bars = rising_bars(400);
        let probs: Vec<Probs> = (0..bars.len())
            .map(|_| Probs {
                p_long: 0.85,
                p_short: 0.05,
                p_take: 0.95,
                calibrated: 0.85,
            })
            .collect();
        let sigma = vec![0.005; bars.len()];
        let params = TraderParams {
            long_threshold: 0.55,
            short_threshold: 0.55,
            take_threshold: 0.50,
            min_conf_margin: 0.10,
            stop_loss_atr: 1.5,
            take_profit_atr: 2.0,
            min_hold_bars: 1,
            max_hold_bars: 60,
            cooldown_bars: 1,
            ..TraderParams::default()
        };
        let costs = Costs {
            commission_bp: 0.0,
            spread_bp: 0.0,
            slippage_bp: 0.0,
        };
        let r = run_backtest(&bars, &probs, &sigma, params, costs);
        assert!(
            r.summary.n_trades > 0,
            "expected at least one trade, got {:?}",
            r.summary
        );
        assert!(
            r.summary.net_return > 0.0,
            "expected positive net return on a steadily rising market, got {}",
            r.summary.net_return
        );
    }
}

//! Synthetic paper book: VWAP positions per instrument + cash + mark-to-market.
//!
//! M1: pure state machine. The estimator-spawn task moved here is gone — that
//! lives in `api-server::estimator_runner` since it needs the bus and the
//! AppState's `latest_prices`. This crate is I/O-free.

use std::collections::HashMap;

use market_domain::{EstimateTick, TransactionEvent};

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Position {
    /// Net long units (negative = short).
    pub units: f64,
    /// Volume-weighted average entry price.
    pub avg_price: f64,
    /// Realized P/L attributed to this instrument so far.
    pub realized: f64,
}

/// VWAP-tracked positions per instrument plus a single cash bucket.
///
/// `seeded` is true once `seed_cash` has been called. `apply_oanda_fill`
/// silently no-ops on non-`ORDER_FILL` kinds (financing/transfer/dividend
/// adjustments are recognized and applied to cash directly; everything
/// else is ignored).
#[derive(Default, Debug)]
pub struct PaperBook {
    positions: HashMap<String, Position>,
    cash: f64,
    seeded: bool,
}

impl PaperBook {
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed cash from the first OANDA account snapshot so the very first
    /// estimate ties to the actual NAV.
    pub fn seed_cash(&mut self, balance: f64) {
        self.cash = balance;
        self.seeded = true;
    }

    pub fn is_seeded(&self) -> bool {
        self.seeded
    }

    pub fn cash(&self) -> f64 {
        self.cash
    }

    pub fn position(&self, instrument: &str) -> Option<&Position> {
        self.positions.get(instrument)
    }

    pub fn positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }

    /// Apply an internal paper fill: synthetic version of an order fill,
    /// without going through OANDA. Same VWAP/realized-PL logic as
    /// `apply_oanda_fill`, just driven from a `(units, price)` pair.
    pub fn apply_paper_fill(&mut self, instrument: &str, units: i64, price: f64) {
        let units_f = units as f64;
        if units_f == 0.0 {
            return;
        }
        let pos = self.positions.entry(instrument.to_string()).or_default();
        let new_units = pos.units + units_f;
        let mut realized = 0.0;
        if pos.units == 0.0 {
            pos.avg_price = price;
        } else if pos.units.signum() == units_f.signum() {
            // Same direction — VWAP in.
            let total = pos.units + units_f;
            if total != 0.0 {
                pos.avg_price = (pos.avg_price * pos.units + price * units_f) / total;
            }
        } else {
            // Opposing — realize PL on the closed portion.
            let closed = units_f.abs().min(pos.units.abs()).copysign(pos.units);
            realized = closed * (price - pos.avg_price);
            if new_units == 0.0 {
                pos.avg_price = 0.0;
            } else if pos.units.signum() != new_units.signum() {
                // Flipped through zero. New side starts at fill price.
                pos.avg_price = price;
            }
        }
        pos.units = new_units;
        pos.realized += realized;
        self.cash += realized;
    }

    /// Apply an OANDA transaction event. Updates positions on ORDER_FILL,
    /// adjusts cash on financing / transfer / dividend events, ignores
    /// everything else.
    pub fn apply_oanda_fill(&mut self, tx: &TransactionEvent) {
        if tx.kind != "ORDER_FILL" {
            // Cash-only events.
            if matches!(
                tx.kind.as_str(),
                "TRANSFER_FUNDS" | "DAILY_FINANCING" | "DIVIDEND_ADJUSTMENT"
            ) {
                if let Some(pl) = tx.pl {
                    self.cash += pl;
                }
            }
            return;
        }
        let instrument = match &tx.instrument {
            Some(i) => i.clone(),
            None => return,
        };
        let units = tx.units.unwrap_or(0.0);
        let price = tx.price.unwrap_or(0.0);
        let realized = tx.pl.unwrap_or(0.0);

        let pos = self.positions.entry(instrument).or_default();

        // If reducing or flipping the position, realize P/L on the closed portion.
        let new_units = pos.units + units;
        if pos.units == 0.0 {
            pos.avg_price = price;
        } else if pos.units.signum() == units.signum() {
            // Same direction — VWAP the new fill in.
            let total = pos.units + units;
            if total != 0.0 {
                pos.avg_price = (pos.avg_price * pos.units + price * units) / total;
            }
        } else {
            // Opposing direction — partially or fully closing. The fill's
            // `pl` already tells us the realized amount, so we just update
            // units; if we cross zero, the next leg starts a new VWAP at
            // `price`.
            if new_units == 0.0 {
                pos.avg_price = 0.0;
            } else if pos.units.signum() != new_units.signum() {
                // Flipped through zero. New side starts at fill price.
                pos.avg_price = price;
            }
            // If same sign as before but smaller magnitude, avg_price unchanged.
        }
        pos.units = new_units;
        pos.realized += realized;
        self.cash += realized;
    }

    /// Mark every open position to market against the supplied price oracle
    /// and return the resulting estimate tick. Returns `None` until the
    /// book has been seeded.
    pub fn mark_to_market<F>(&self, actual_nav: f64, mid_for: F) -> Option<EstimateTick>
    where
        F: Fn(&str) -> Option<f64>,
    {
        if !self.seeded {
            return None;
        }
        let mut estimated = self.cash;
        for (inst, pos) in self.positions.iter() {
            if pos.units == 0.0 {
                continue;
            }
            if let Some(mid) = mid_for(inst) {
                // For a long position, unrealized = units * (mid - avg_price).
                // OANDA's NAV uses bid/ask for the closing side, so a small
                // spread-driven drift is expected and visible in the chart.
                estimated += pos.units * (mid - pos.avg_price);
            }
        }
        let drift = estimated - actual_nav;
        let drift_bps = if actual_nav != 0.0 {
            drift / actual_nav * 10_000.0
        } else {
            0.0
        };
        Some(EstimateTick {
            time: chrono::Utc::now(),
            estimated_balance: estimated,
            actual_balance: actual_nav,
            drift,
            drift_bps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use proptest::prelude::*;

    fn fill(instrument: &str, units: f64, price: f64, pl: f64) -> TransactionEvent {
        TransactionEvent {
            id: "test".into(),
            time: Utc::now(),
            kind: "ORDER_FILL".into(),
            instrument: Some(instrument.into()),
            units: Some(units),
            price: Some(price),
            pl: Some(pl),
            reason: None,
            raw: serde_json::Value::Null,
        }
    }

    #[test]
    fn seeded_zero_position_estimate_equals_cash() {
        let mut b = PaperBook::new();
        b.seed_cash(100_000.0);
        let est = b.mark_to_market(100_000.0, |_| None).unwrap();
        assert!((est.estimated_balance - 100_000.0).abs() < 1e-9);
        assert!((est.drift).abs() < 1e-9);
    }

    #[test]
    fn unseeded_returns_none() {
        let b = PaperBook::new();
        assert!(b.mark_to_market(0.0, |_| None).is_none());
    }

    #[test]
    fn open_long_then_mark_at_higher_mid_increases_estimate() {
        let mut b = PaperBook::new();
        b.seed_cash(100_000.0);
        b.apply_oanda_fill(&fill("EUR_USD", 100.0, 1.10, 0.0));
        // Mid moves from 1.10 -> 1.11 = +1 bp on 100 units = +0.01 estimate.
        let est = b
            .mark_to_market(
                100_000.0,
                |i| {
                    if i == "EUR_USD" {
                        Some(1.11)
                    } else {
                        None
                    }
                },
            )
            .unwrap();
        assert!((est.estimated_balance - (100_000.0 + 100.0 * (1.11 - 1.10))).abs() < 1e-9);
    }

    #[test]
    fn closing_position_returns_to_seeded_cash() {
        let mut b = PaperBook::new();
        b.seed_cash(100_000.0);
        b.apply_oanda_fill(&fill("EUR_USD", 100.0, 1.10, 0.0));
        // Close at 1.11, so realized = 100 * (1.11 - 1.10) = 1.0
        b.apply_oanda_fill(&fill("EUR_USD", -100.0, 1.11, 1.0));
        // Position should be flat; cash should be 100_000 + 1.0
        assert_eq!(b.position("EUR_USD").map(|p| p.units), Some(0.0));
        assert!((b.cash() - 100_001.0).abs() < 1e-9);
        let est = b.mark_to_market(100_001.0, |_| None).unwrap();
        assert!((est.estimated_balance - 100_001.0).abs() < 1e-9);
        assert!((est.drift).abs() < 1e-9);
    }

    #[test]
    fn financing_event_adjusts_cash_directly() {
        let mut b = PaperBook::new();
        b.seed_cash(100_000.0);
        let fin = TransactionEvent {
            id: "1".into(),
            time: Utc::now(),
            kind: "DAILY_FINANCING".into(),
            instrument: None,
            units: None,
            price: None,
            pl: Some(-0.50),
            reason: None,
            raw: serde_json::Value::Null,
        };
        b.apply_oanda_fill(&fin);
        assert!((b.cash() - 99_999.50).abs() < 1e-9);
    }

    #[test]
    fn ignores_unrelated_transaction_kind() {
        let mut b = PaperBook::new();
        b.seed_cash(100_000.0);
        let other = TransactionEvent {
            id: "1".into(),
            time: Utc::now(),
            kind: "SOME_OTHER_KIND".into(),
            instrument: None,
            units: None,
            price: None,
            pl: Some(-99999.0), // would ruin us if applied
            reason: None,
            raw: serde_json::Value::Null,
        };
        b.apply_oanda_fill(&other);
        assert!((b.cash() - 100_000.0).abs() < 1e-9);
    }

    proptest! {
        /// PaperBook invariant: after applying a sequence of opens-then-equal-and-opposite-closes
        /// at the same price, cash returns to the seeded value (no drift).
        #[test]
        fn round_trip_at_same_price_preserves_cash(
            initial_cash in 1.0f64..1_000_000.0,
            units in 1i64..100_000,
            price_int in 1i64..1_000_000,
        ) {
            let price = price_int as f64 / 1000.0;
            let units_f = units as f64;
            let mut b = PaperBook::new();
            b.seed_cash(initial_cash);
            b.apply_oanda_fill(&fill("EUR_USD", units_f, price, 0.0));
            // Close at the same price -> realized = 0
            b.apply_oanda_fill(&fill("EUR_USD", -units_f, price, 0.0));
            prop_assert!((b.cash() - initial_cash).abs() < 1e-6);
            prop_assert_eq!(b.position("EUR_USD").map(|p| p.units), Some(0.0));
        }
    }
}

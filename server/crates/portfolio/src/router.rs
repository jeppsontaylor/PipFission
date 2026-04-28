//! `OrderRouter` trait + the in-process `PaperRouter`.
//!
//! `PaperRouter` synthesizes fills from current best bid/ask: a buy fills
//! at ask, a sell at bid. Slippage is modeled solely by the spread
//! (no extra slippage_bp in v1). Fees default to 0 (OANDA practice has
//! no commissions).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use dashmap::DashMap;
use thiserror::Error;

use market_domain::{OrderIntent, PaperFillEvent, PriceTick, RoutingMode};

#[derive(Debug, Error)]
pub enum RouterError {
    #[error("E_NO_PRICE: no current price for {0}")]
    NoPrice(String),
    #[error("E_ZERO_UNITS: order with zero units rejected")]
    ZeroUnits,
    #[error("E_OANDA: {0}")]
    Oanda(String),
}

#[async_trait]
pub trait OrderRouter: Send + Sync + std::fmt::Debug {
    fn mode(&self) -> RoutingMode;
    async fn submit(&self, intent: OrderIntent) -> Result<PaperFillEvent, RouterError>;
}

/// Looks up the most-recent best bid/ask per instrument. The api-server
/// mirrors AppState.latest_prices into this struct so the router doesn't
/// import api-server types.
#[derive(Debug, Default)]
pub struct PriceOracle {
    inner: DashMap<String, PriceTick>,
}

impl PriceOracle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&self, tick: PriceTick) {
        self.inner.insert(tick.instrument.clone(), tick);
    }

    pub fn get(&self, instrument: &str) -> Option<PriceTick> {
        self.inner.get(instrument).map(|v| v.clone())
    }
}

#[derive(Debug)]
pub struct PaperRouter {
    oracle: Arc<PriceOracle>,
    next_id: AtomicU64,
    fee_per_unit: f64,
}

impl PaperRouter {
    pub fn new(oracle: Arc<PriceOracle>) -> Self {
        Self {
            oracle,
            next_id: AtomicU64::new(1),
            fee_per_unit: 0.0,
        }
    }
}

#[async_trait]
impl OrderRouter for PaperRouter {
    fn mode(&self) -> RoutingMode {
        RoutingMode::Internal
    }

    async fn submit(&self, intent: OrderIntent) -> Result<PaperFillEvent, RouterError> {
        if intent.units == 0 {
            return Err(RouterError::ZeroUnits);
        }
        let tick = self
            .oracle
            .get(&intent.instrument)
            .ok_or_else(|| RouterError::NoPrice(intent.instrument.clone()))?;
        // Buys hit ask, sells hit bid.
        let price = if intent.units > 0 { tick.ask } else { tick.bid };
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let fee = (intent.units.unsigned_abs() as f64) * self.fee_per_unit;
        Ok(PaperFillEvent {
            mode: RoutingMode::Internal,
            order_id: format!("paper-{id}"),
            instrument: intent.instrument,
            units: intent.units,
            price,
            fee,
            time: Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use market_domain::PriceTick;

    fn make_tick(inst: &str, bid: f64, ask: f64) -> PriceTick {
        PriceTick {
            instrument: inst.into(),
            time: Utc::now(),
            bid,
            ask,
            mid: (bid + ask) / 2.0,
            spread: ask - bid,
            closeout_bid: None,
            closeout_ask: None,
            status: None,
        }
    }

    #[tokio::test]
    async fn paper_router_buys_at_ask() {
        let oracle = Arc::new(PriceOracle::new());
        oracle.update(make_tick("EUR_USD", 1.10, 1.10020));
        let r = PaperRouter::new(oracle);
        let f = r
            .submit(OrderIntent {
                instrument: "EUR_USD".into(),
                units: 100,
                time: Utc::now(),
                model_id: "m1".into(),
            })
            .await
            .unwrap();
        assert_eq!(f.units, 100);
        assert!((f.price - 1.10020).abs() < 1e-9);
    }

    #[tokio::test]
    async fn paper_router_sells_at_bid() {
        let oracle = Arc::new(PriceOracle::new());
        oracle.update(make_tick("EUR_USD", 1.10, 1.10020));
        let r = PaperRouter::new(oracle);
        let f = r
            .submit(OrderIntent {
                instrument: "EUR_USD".into(),
                units: -100,
                time: Utc::now(),
                model_id: "m1".into(),
            })
            .await
            .unwrap();
        assert!((f.price - 1.10).abs() < 1e-9);
    }

    #[tokio::test]
    async fn paper_router_rejects_zero_units() {
        let oracle = Arc::new(PriceOracle::new());
        oracle.update(make_tick("EUR_USD", 1.10, 1.10020));
        let r = PaperRouter::new(oracle);
        let res = r
            .submit(OrderIntent {
                instrument: "EUR_USD".into(),
                units: 0,
                time: Utc::now(),
                model_id: "m1".into(),
            })
            .await;
        assert!(matches!(res, Err(RouterError::ZeroUnits)));
    }

    #[tokio::test]
    async fn paper_router_no_price_errors() {
        let oracle = Arc::new(PriceOracle::new());
        let r = PaperRouter::new(oracle);
        let res = r
            .submit(OrderIntent {
                instrument: "EUR_USD".into(),
                units: 10,
                time: Utc::now(),
                model_id: "m1".into(),
            })
            .await;
        assert!(matches!(res, Err(RouterError::NoPrice(_))));
    }
}

//! OandaRouter — wraps the existing `Client::place_market_order` so that
//! `OrderIntent`s emitted by the strategy can hit the OANDA practice
//! account. Implements `portfolio::OrderRouter` so it's swap-compatible
//! with `PaperRouter` via the AppState `ArcSwap`.
//!
//! Sprint 2 milestone M6.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::Value;

use market_domain::{OrderIntent, PaperFillEvent, RoutingMode};
use portfolio::{OrderRouter, RouterError};

use crate::client::Client;

#[derive(Debug)]
pub struct OandaRouter {
    client: Client,
    account_id: String,
    /// Fallback price oracle used if OANDA's response doesn't include a
    /// fill price (rare but possible for partial fills / rejections).
    fallback: Arc<portfolio::PriceOracle>,
}

impl OandaRouter {
    pub fn new(client: Client, account_id: String, fallback: Arc<portfolio::PriceOracle>) -> Self {
        Self {
            client,
            account_id,
            fallback,
        }
    }
}

#[async_trait]
impl OrderRouter for OandaRouter {
    fn mode(&self) -> RoutingMode {
        RoutingMode::OandaPractice
    }

    async fn submit(&self, intent: OrderIntent) -> Result<PaperFillEvent, RouterError> {
        if intent.units == 0 {
            return Err(RouterError::ZeroUnits);
        }
        let resp: Value = self
            .client
            .place_market_order(&self.account_id, &intent.instrument, intent.units)
            .await
            .map_err(|e| RouterError::Oanda(format!("{e:#}")))?;

        // OANDA returns either `orderFillTransaction` (success) or
        // `orderCancelTransaction` (rejected). Pull a fill price out of the
        // success path; fall back to current best bid/ask on the response
        // path that didn't include a price.
        let (price, order_id) = if let Some(fill) = resp.get("orderFillTransaction") {
            let p = fill
                .get("price")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok());
            let id = fill
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("oanda-unknown-{}", Utc::now().timestamp_millis()));
            (p, id)
        } else if let Some(cxl) = resp.get("orderCancelTransaction") {
            let reason = cxl
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("UNKNOWN");
            return Err(RouterError::Oanda(format!("order rejected: {reason}")));
        } else {
            (
                None,
                format!("oanda-unknown-{}", Utc::now().timestamp_millis()),
            )
        };

        let price = match price {
            Some(p) => p,
            None => {
                let tick = self
                    .fallback
                    .get(&intent.instrument)
                    .ok_or_else(|| RouterError::NoPrice(intent.instrument.clone()))?;
                if intent.units > 0 {
                    tick.ask
                } else {
                    tick.bid
                }
            }
        };

        Ok(PaperFillEvent {
            mode: RoutingMode::OandaPractice,
            order_id,
            instrument: intent.instrument,
            units: intent.units,
            price,
            fee: 0.0,
            time: Utc::now(),
        })
    }
}

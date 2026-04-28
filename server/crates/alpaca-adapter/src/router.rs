//! Alpaca paper-trading order router.
//!
//! POSTs to `https://paper-api.alpaca.markets/v2/orders` with the
//! Alpaca header auth (no Bearer; uses `APCA-API-KEY-ID` +
//! `APCA-API-SECRET-KEY`). Implements `portfolio::OrderRouter` so it
//! can be slotted into the same `ArcSwap<dyn OrderRouter>` as
//! OandaRouter.
//!
//! Sprint 2 milestone A4. Symbol translation: data API uses `BTC/USD`,
//! trading API uses `BTCUSD` — we strip the slash at the boundary.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{json, Value};

use market_domain::{OrderIntent, PaperFillEvent, RoutingMode};
use portfolio::{OrderRouter, PriceOracle, RouterError};

use crate::config::AlpacaConfig;

#[derive(Debug)]
pub struct AlpacaRouter {
    cfg: AlpacaConfig,
    http: reqwest::Client,
    fallback_oracle: Arc<PriceOracle>,
}

impl AlpacaRouter {
    pub fn new(cfg: AlpacaConfig, fallback_oracle: Arc<PriceOracle>) -> anyhow::Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("APCA-API-KEY-ID", cfg.key.parse()?);
        headers.insert("APCA-API-SECRET-KEY", cfg.secret.parse()?);
        let http = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(10))
            .user_agent("oanda-live-server/0.1 alpaca")
            .build()?;
        Ok(Self { cfg, http, fallback_oracle })
    }

    /// `GET /v2/account` — small snapshot useful for the dashboard.
    pub async fn fetch_account(&self) -> anyhow::Result<Value> {
        let url = format!("{}/v2/account", self.cfg.trading_base_url);
        let resp = self.http.get(&url).send().await?;
        let status = resp.status();
        let body: Value = resp.json().await?;
        if !status.is_success() {
            return Err(anyhow::anyhow!("alpaca /v2/account {}: {}", status, body));
        }
        Ok(body)
    }
}

#[async_trait]
impl OrderRouter for AlpacaRouter {
    fn mode(&self) -> RoutingMode {
        // Alpaca sits behind the same "OandaPractice" enum value because we
        // share one `live` toggle. We extend the meaning: when mode is
        // OandaPractice, OANDA forex routes via OandaRouter and Alpaca
        // crypto routes via AlpacaRouter.
        RoutingMode::OandaPractice
    }

    async fn submit(&self, intent: OrderIntent) -> Result<PaperFillEvent, RouterError> {
        if intent.units == 0 {
            return Err(RouterError::ZeroUnits);
        }
        let trading_symbol = AlpacaConfig::data_symbol_to_trading(&intent.instrument);
        let side = if intent.units > 0 { "buy" } else { "sell" };

        // Alpaca crypto orders are best expressed as `notional` (dollar
        // amount) rather than `qty` (asset quantity) — easier to size
        // consistently across BTC ($77k/coin) and ETH ($2.3k/coin).
        //
        // Mapping: 1 intent unit → $0.10 of notional. Strategy default
        // units_per_signal=100 → $10 (Alpaca's minimum). Larger signals
        // scale linearly. Floor at $10 — Alpaca rejects below that with
        // code 40310000 ("cost basis must be >= minimal amount of order 10").
        const ALPACA_MIN_NOTIONAL: f64 = 10.0;
        let notional_dollars: f64 =
            ((intent.units.unsigned_abs() as f64) * 0.10).max(ALPACA_MIN_NOTIONAL);
        let notional = format!("{notional_dollars:.2}");

        let body = json!({
            "symbol": trading_symbol,
            "notional": notional,
            "side": side,
            "type": "market",
            "time_in_force": "gtc",
        });

        let url = format!("{}/v2/orders", self.cfg.trading_base_url);
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| RouterError::Oanda(format!("alpaca post: {e:#}")))?;
        let status = resp.status();
        let v: Value = resp
            .json()
            .await
            .map_err(|e| RouterError::Oanda(format!("alpaca parse: {e:#}")))?;
        if !status.is_success() {
            // Alpaca returns {"code":..., "message":"..."} on failure.
            let msg = v
                .get("message")
                .and_then(|x| x.as_str())
                .unwrap_or("(no message)");
            return Err(RouterError::Oanda(format!(
                "alpaca order rejected ({status}): {msg}"
            )));
        }

        let order_id = v
            .get("id")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("alpaca-unknown-{}", Utc::now().timestamp_millis()));

        // Alpaca's `POST /v2/orders` returns the order in `accepted` /
        // `pending_new` state — the fill price comes via streaming or a
        // follow-up GET. For Sprint 2 we estimate price from the local
        // oracle (last quote) which is good enough for display; the
        // periodic account-poll will reconcile against actual NAV.
        let est_price = self
            .fallback_oracle
            .get(&intent.instrument)
            .map(|t| if intent.units > 0 { t.ask } else { t.bid })
            .unwrap_or(0.0);

        Ok(PaperFillEvent {
            mode: RoutingMode::OandaPractice,
            order_id,
            instrument: intent.instrument,
            units: intent.units,
            price: est_price,
            fee: 0.0,
            time: Utc::now(),
        })
    }
}

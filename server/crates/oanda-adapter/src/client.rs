//! Thin async client for the OANDA v20 REST + streaming endpoints.
//!
//! The streaming endpoints (`/pricing/stream`, `/transactions/stream`) emit
//! newline-delimited JSON over a long-lived HTTP connection. We deliberately
//! do *not* deserialize each line into a strict struct here — we keep them as
//! `serde_json::Value` so individual stream consumers can pluck out the fields
//! they care about without the whole pipeline breaking when OANDA tacks on a
//! new optional field.

use std::pin::Pin;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use futures_util::Stream;
use reqwest::header::{ACCEPT_ENCODING, AUTHORIZATION, CONNECTION};
use reqwest::Client as HttpClient;
use serde_json::Value;

use market_domain::Config;

/// Pinned, boxed, Send-safe byte stream — what callers actually want to
/// `.next().await` on without fighting the borrow checker.
pub type ByteStream = Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>;

/// Wraps a `reqwest::Client` configured with OANDA auth.
#[derive(Clone, Debug)]
pub struct Client {
    cfg: Config,
    rest: HttpClient,
    stream: HttpClient,
}

impl Client {
    pub fn new(cfg: Config) -> Result<Self> {
        let auth = format!("Bearer {}", cfg.api_token);

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            auth.parse().context("invalid OANDA_API_TOKEN")?,
        );
        headers.insert("Accept-Datetime-Format", "RFC3339".parse().unwrap());

        // REST client: short-ish timeouts, gzip enabled.
        let rest = HttpClient::builder()
            .default_headers(headers.clone())
            .timeout(Duration::from_secs(10))
            .pool_idle_timeout(Some(Duration::from_secs(60)))
            .user_agent("oanda-live-server/0.1")
            .build()
            .context("building REST client")?;

        // Streaming client: NO global timeout (or it'll kill long-lived
        // connections). We rely on heartbeat absence to detect stalls.
        // Disable gzip for streaming so the chunked NDJSON arrives line-aligned.
        //
        // IMPORTANT: do NOT call `.timeout(Duration::from_secs(0))` — in
        // reqwest 0.12 that's a 0-second timeout (immediate timeout), not
        // "no timeout". Omit the call entirely; `ClientBuilder` defaults
        // to no per-request timeout.
        let mut stream_headers = headers;
        stream_headers.insert(CONNECTION, "keep-alive".parse().unwrap());
        stream_headers.insert(ACCEPT_ENCODING, "identity".parse().unwrap());
        let stream = HttpClient::builder()
            .default_headers(stream_headers)
            .connect_timeout(Duration::from_secs(15))
            .pool_idle_timeout(Some(Duration::from_secs(300)))
            .tcp_keepalive(Some(Duration::from_secs(20)))
            .user_agent("oanda-live-server/0.1")
            .build()
            .context("building streaming client")?;

        Ok(Self { cfg, rest, stream })
    }

    pub fn cfg(&self) -> &Config {
        &self.cfg
    }

    /// `GET /v3/accounts` and pick the first one. Used when OANDA_ACCOUNT_ID
    /// isn't set.
    pub async fn discover_account_id(&self) -> Result<String> {
        let url = format!("{}/v3/accounts", self.cfg.rest_url);
        let resp = self
            .rest
            .get(&url)
            .send()
            .await
            .context("GET /v3/accounts")?;
        let status = resp.status();
        let body: Value = resp.json().await.context("parsing /v3/accounts response")?;
        if !status.is_success() {
            return Err(anyhow!("/v3/accounts returned {}: {}", status, body));
        }
        let accounts = body
            .get("accounts")
            .and_then(|a| a.as_array())
            .ok_or_else(|| anyhow!("/v3/accounts missing 'accounts' array: {body}"))?;
        let first = accounts
            .first()
            .and_then(|a| a.get("id"))
            .and_then(|i| i.as_str())
            .ok_or_else(|| anyhow!("no accounts on this token"))?;
        Ok(first.to_string())
    }

    /// `GET /v3/accounts/{id}/summary` — small, fast, hot-pollable.
    pub async fn account_summary(&self, account_id: &str) -> Result<Value> {
        let url = format!("{}/v3/accounts/{}/summary", self.cfg.rest_url, account_id);
        let resp = self.rest.get(&url).send().await.context("GET /summary")?;
        let status = resp.status();
        let body: Value = resp.json().await.context("parsing /summary")?;
        if !status.is_success() {
            return Err(anyhow!("/summary {}: {}", status, body));
        }
        Ok(body.get("account").cloned().unwrap_or(body))
    }

    /// `GET /v3/accounts/{id}/pricing/stream` — NDJSON pricing stream.
    /// Each chunk is one JSON object; consumers should split on `\n`.
    pub async fn pricing_stream(
        &self,
        account_id: &str,
        instruments: &[String],
    ) -> Result<ByteStream> {
        let url = format!(
            "{}/v3/accounts/{}/pricing/stream",
            self.cfg.stream_url, account_id
        );
        let resp = self
            .stream
            .get(&url)
            .query(&[("instruments", instruments.join(","))])
            .send()
            .await
            .context("opening pricing/stream")?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("pricing/stream {}: {}", status, body));
        }
        Ok(Box::pin(resp.bytes_stream()))
    }

    /// `GET /v3/accounts/{id}/transactions/stream` — NDJSON transaction stream.
    pub async fn transactions_stream(&self, account_id: &str) -> Result<ByteStream> {
        let url = format!(
            "{}/v3/accounts/{}/transactions/stream",
            self.cfg.stream_url, account_id
        );
        let resp = self
            .stream
            .get(&url)
            .send()
            .await
            .context("opening transactions/stream")?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("transactions/stream {}: {}", status, body));
        }
        Ok(Box::pin(resp.bytes_stream()))
    }

    /// `POST /v3/accounts/{id}/orders` — place a market order. Wired up so
    /// the dashboard / a future strategy can submit fills, but not auto-called
    /// until M6 (OandaRouter).
    #[allow(dead_code)]
    pub async fn place_market_order(
        &self,
        account_id: &str,
        instrument: &str,
        units: i64,
    ) -> Result<Value> {
        let url = format!("{}/v3/accounts/{}/orders", self.cfg.rest_url, account_id);
        let body = serde_json::json!({
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": units.to_string(),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        });
        let resp = self
            .rest
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("POST /orders")?;
        let status = resp.status();
        let v: Value = resp.json().await.context("parsing /orders response")?;
        if !status.is_success() {
            return Err(anyhow!("/orders {}: {}", status, v));
        }
        Ok(v)
    }
}

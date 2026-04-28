//! Runtime configuration. Pulled from env vars with sensible defaults.

use std::env;

use thiserror::Error;

/// The 7 major FX pairs we track on the dashboard.
pub const DEFAULT_INSTRUMENTS: &[&str] = &[
    "EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD",
];

/// Errors building Config from env. Stable codes.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("E_CFG_MISSING_TOKEN: OANDA_API_TOKEN must be set (in .env or environment)")]
    MissingToken,
    #[error("E_CFG_INVALID_ENV: OANDA_ENV must be practice|live, got {0:?}")]
    InvalidEnv(String),
}

#[derive(Clone, Debug)]
pub struct Config {
    pub api_token: String,
    pub account_id: Option<String>,
    pub environment: String, // "practice" | "live"
    pub rest_url: String,
    pub stream_url: String,
    pub instruments: Vec<String>,
    pub bind_addr: String,
    /// How often to poll account summary, in milliseconds. OANDA rate-limits
    /// REST to 100 req/s per token globally; 250ms = 4Hz which is plenty.
    pub account_poll_ms: u64,
    /// How often the estimator emits a tick.
    pub estimator_tick_ms: u64,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        let api_token = env::var("OANDA_API_TOKEN").map_err(|_| ConfigError::MissingToken)?;
        let account_id = env::var("OANDA_ACCOUNT_ID").ok().filter(|s| !s.is_empty());
        let environment = env::var("OANDA_ENV").unwrap_or_else(|_| "practice".to_string());

        let (rest_url, stream_url) = match environment.as_str() {
            "live" => (
                "https://api-fxtrade.oanda.com".to_string(),
                "https://stream-fxtrade.oanda.com".to_string(),
            ),
            "practice" => (
                "https://api-fxpractice.oanda.com".to_string(),
                "https://stream-fxpractice.oanda.com".to_string(),
            ),
            other => return Err(ConfigError::InvalidEnv(other.to_string())),
        };

        let instruments: Vec<String> = match env::var("OANDA_INSTRUMENTS") {
            Ok(s) if !s.trim().is_empty() => s.split(',').map(|x| x.trim().to_string()).collect(),
            _ => DEFAULT_INSTRUMENTS.iter().map(|s| s.to_string()).collect(),
        };

        let bind_addr = env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8088".to_string());
        let account_poll_ms = env::var("ACCOUNT_POLL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(250);
        let estimator_tick_ms = env::var("ESTIMATOR_TICK_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(500);

        Ok(Self {
            api_token,
            account_id,
            environment,
            rest_url,
            stream_url,
            instruments,
            bind_addr,
            account_poll_ms,
            estimator_tick_ms,
        })
    }
}

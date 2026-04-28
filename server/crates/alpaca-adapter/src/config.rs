//! Alpaca config. Loaded from env. Both data + trading credentials are
//! the same key/secret pair (Alpaca uses the same auth for paper-trading
//! REST and crypto-data WS).

use std::env;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AlpacaConfigError {
    #[error("E_ALPACA_MISSING_KEY: ALPACA_KEY must be set in env")]
    MissingKey,
    #[error("E_ALPACA_MISSING_SECRET: ALPACA_SECRET must be set in env")]
    MissingSecret,
}

#[derive(Clone, Debug)]
pub struct AlpacaConfig {
    pub key: String,
    pub secret: String,
    /// Crypto data WS URL. Default: us exchange.
    pub data_ws_url: String,
    /// Paper trading REST base URL.
    pub trading_base_url: String,
    /// Symbols to subscribe to. Format used by the data API: "BTC/USD".
    pub symbols: Vec<String>,
    /// How often to poll the Alpaca account, in milliseconds.
    pub account_poll_ms: u64,
}

impl AlpacaConfig {
    pub fn from_env() -> Result<Self, AlpacaConfigError> {
        let key = env::var("ALPACA_KEY").map_err(|_| AlpacaConfigError::MissingKey)?;
        let secret = env::var("ALPACA_SECRET").map_err(|_| AlpacaConfigError::MissingSecret)?;
        let data_ws_url = env::var("ALPACA_DATA_WS_URL").unwrap_or_else(|_| {
            "wss://stream.data.alpaca.markets/v1beta3/crypto/us".to_string()
        });
        let trading_base_url = env::var("ALPACA_TRADING_URL")
            .unwrap_or_else(|_| "https://paper-api.alpaca.markets".to_string());
        let symbols: Vec<String> = match env::var("ALPACA_SYMBOLS") {
            Ok(s) if !s.trim().is_empty() => {
                s.split(',').map(|x| x.trim().to_string()).collect()
            }
            _ => vec!["BTC/USD".into(), "ETH/USD".into()],
        };
        let account_poll_ms = env::var("ALPACA_ACCOUNT_POLL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2_000);
        Ok(Self {
            key,
            secret,
            data_ws_url,
            trading_base_url,
            symbols,
            account_poll_ms,
        })
    }

    /// Translate a data-API symbol (BTC/USD) to a trading-API symbol
    /// (BTCUSD). Alpaca uses different conventions for the two.
    pub fn data_symbol_to_trading(symbol: &str) -> String {
        symbol.replace('/', "")
    }
}

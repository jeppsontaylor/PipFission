// Mirror of the Rust feature schema in
// `server/crates/market-domain/src/features.rs` and the strategy
// constants in `server/crates/strategy/src/runner.rs`. Single
// source of truth on the client side.

export const FEATURE_NAMES: readonly string[] = [
  "log_return_1tick",
  "log_return_10tick",
  "log_return_60tick",
  "realized_vol_60tick",
  "realized_vol_300tick",
  "spread_bp",
  "spread_zscore_300tick",
  "mid_minus_sma60_over_mid",
  "mid_minus_sma300_over_mid",
  "rsi_14_on_60tick_bars",
  "bollinger_pct_b_300tick",
  "atr_14_on_60tick_bars",
  "macd_line_60_300",
  "macd_signal_minus_line",
  "tick_velocity_60",
  "range_60tick_over_mid",
  "minute_of_hour_sin",
  "minute_of_hour_cos",
  "ob_top_imbalance",
  "ob_depth_weighted_mid_bp",
  "ob_bid_wall_ratio",
  "ob_ask_wall_ratio",
  "ob_cum_depth_10bp",
  "ob_cum_depth_50bp",
] as const;

export const FEATURE_COUNT = FEATURE_NAMES.length;

/// Indices 18..24 are crypto-only (depend on order book depth, which
/// the OANDA forex feed does not provide). Names match the Rust side.
export const CRYPTO_ONLY_FEATURE_START = 18;

export function isCryptoOnlyFeature(idx: number): boolean {
  return idx >= CRYPTO_ONLY_FEATURE_START;
}

// Strategy / feature-engine constants. Mirror of:
// - `feature-engine/src/lib.rs:20`        WARMUP_TICKS
// - `strategy/src/runner.rs:21-25`        TRAIN_AFTER, RETRAIN_EVERY
export const WARMUP_TICKS = 300;
export const TRAIN_AFTER = 1000;
export const RETRAIN_EVERY = 100;
export const FIRST_MODEL_TICK_TARGET = WARMUP_TICKS + TRAIN_AFTER;

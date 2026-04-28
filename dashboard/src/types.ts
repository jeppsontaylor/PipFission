// Mirror of the server's serde-tagged Event enum. Keep these names in sync
// with `server/src/state.rs` (rename_all = "snake_case").

export interface PriceTick {
  instrument: string;
  time: string;
  bid: number;
  ask: number;
  mid: number;
  spread: number;
  closeout_bid: number | null;
  closeout_ask: number | null;
  status: string | null;
}

export interface AccountSnapshot {
  time: string;
  nav: number;
  balance: number;
  unrealized_pl: number;
  realized_pl: number;
  margin_used: number;
  margin_available: number;
  open_position_count: number;
  open_trade_count: number;
  pending_order_count: number;
  leverage: number;
  currency: string;
}

export interface TransactionEvent {
  id: string;
  time: string;
  kind: string;
  instrument: string | null;
  units: number | null;
  price: number | null;
  pl: number | null;
  reason: string | null;
  raw: unknown;
}

export interface EstimateTick {
  time: string;
  estimated_balance: number;
  actual_balance: number;
  drift: number;
  drift_bps: number;
}

export interface StreamHealth {
  connected: boolean;
  last_message: string | null;
  messages_received: number;
  reconnects: number;
  last_error: string | null;
}

export interface ConnStatus {
  pricing_stream: StreamHealth;
  transaction_stream: StreamHealth;
  account_poll: StreamHealth;
}

export interface AlpacaAccountSnapshot {
  time: string;
  equity: number;
  cash: number;
  buying_power: number;
  portfolio_value: number;
  status: string;
  currency: string;
  long_market_value: number;
  short_market_value: number;
}

export interface Snapshot {
  server_time: string;
  prices: PriceTick[];
  account: AccountSnapshot | null;
  estimate: EstimateTick | null;
  recent_transactions: TransactionEvent[];
  connections: ConnStatus;
  alpaca?: AlpacaAccountSnapshot | null;
}

/** Venue inference from instrument symbol. Used for the dashboard
 *  venue filter — this matches the Rust side's classify_venue(). */
export type Venue = "oanda" | "alpaca";

export function instrumentVenue(instrument: string): Venue {
  if (instrument.includes("/")) return "alpaca";
  if (
    instrument === "BTC_USD" ||
    instrument === "ETH_USD" ||
    instrument === "SOL_USD" ||
    instrument === "DOGE_USD" ||
    instrument === "LTC_USD"
  ) {
    return "alpaca";
  }
  return "oanda";
}

export type RoutingMode = "internal" | "oanda_practice";

export type SignalDirection = "long" | "flat" | "short";

export interface FeatureVector {
  instrument: string;
  time: string;
  version: number;
  vector: number[];
}

export interface StrategySignal {
  instrument: string;
  time: string;
  direction: SignalDirection;
  confidence: number;
  probs: [number, number, number];
  model_id: string;
  model_version: number;
}

export interface FitnessMetrics {
  samples: number;
  accuracy: number;
  log_loss: number;
  sharpe: number;
  class_distribution: [number, number, number];
}

export interface ModelFitness {
  instrument: string;
  model_id: string;
  model_version: number;
  trained_at: string;
  train: FitnessMetrics;
  oos: FitnessMetrics;
  samples_seen: number;
  train_window: [number, number];
  oos_window: [number, number];
}

export interface PaperPosition {
  instrument: string;
  units: number;
  avg_price: number;
  realized: number;
}

export interface OrderBookLevel {
  price: number;
  size: number;
}

export interface OrderBookSide {
  levels: [number, number][]; // tuples of (price, size)
}

export interface OrderBookSnapshot {
  instrument: string;
  time: string;
  bids: OrderBookSide;
  asks: OrderBookSide;
}

// ----- ML pipeline event types -----------------------------------------
// These mirror the server's new event variants (server/crates/market-domain
// /src/{bar,champion}.rs). Keep in sync with the Rust source.

export interface Bar10s {
  instrument: string;
  ts_ms: number;
  open: number;
  high: number;
  low: number;
  close: number;
  n_ticks: number;
  spread_bp_avg: number;
}

export interface ChampionSignal {
  instrument: string;
  time: string;
  p_long: number;
  p_short: number;
  p_take: number;
  calibrated: number;
  model_id: string;
  /** "onnx" or "fallback" */
  kind: string;
}

export interface ChampionChanged {
  model_id: string;
  n_features: number;
  /** "onnx" or "fallback" */
  kind: string;
}

export interface ChampionLoadFailed {
  reason: string;
}

export interface TraderDecision {
  instrument: string;
  time: string;
  bar_idx: number;
  /** "open_long" | "open_short" | "close" | "skip" */
  action: string;
  /** "signal" | "stop_loss" | "take_profit" | "trailing_stop" |
   *  "max_hold" | "reverse" | "spread_too_wide" | "stale_data" |
   *  "daily_loss_kill" | "drawdown_pause" | "cooldown" |
   *  "below_threshold" | "min_hold" */
  reason: string;
  price: number;
  realized_r: number | null;
  params_id: string;
  model_id: string;
}

export interface PaperBookSnapshot {
  mode: RoutingMode;
  version: number;
  time: string;
  cash: number;
  equity: number;
  realized_pl: number;
  unrealized_pl: number;
  positions: PaperPosition[];
}

export interface PaperFillEvent {
  mode: RoutingMode;
  order_id: string;
  instrument: string;
  units: number;
  price: number;
  fee: number;
  time: string;
}

export interface Reconciliation {
  time: string;
  mode: RoutingMode;
  internal_paper_equity: number;
  oanda_actual_equity: number;
  oanda_minus_internal_bp: number;
}

export interface ModeAck {
  mode: RoutingMode;
  effective_at: string;
  error: string | null;
}

export type ServerEvent =
  | {
      type: "hello";
      account_id: string;
      instruments: string[];
      environment: string;
      snapshot: Snapshot;
    }
  | ({ type: "price" } & PriceTick)
  | ({ type: "account" } & AccountSnapshot)
  | ({ type: "transaction" } & TransactionEvent)
  | ({ type: "estimate" } & EstimateTick)
  | ({ type: "status" } & ConnStatus)
  | ({ type: "features" } & FeatureVector)
  | ({ type: "signal" } & StrategySignal)
  | ({ type: "fitness" } & ModelFitness)
  | ({ type: "paper_book" } & PaperBookSnapshot)
  | ({ type: "paper_fill" } & PaperFillEvent)
  | ({ type: "reconciliation" } & Reconciliation)
  | ({ type: "mode_ack" } & ModeAck)
  | ({ type: "order_book" } & OrderBookSnapshot)
  | ({ type: "bar10s" } & Bar10s)
  | ({ type: "champion_signal" } & ChampionSignal)
  | ({ type: "champion_changed" } & ChampionChanged)
  | ({ type: "champion_load_failed" } & ChampionLoadFailed)
  | ({ type: "trader_decision" } & TraderDecision);

/**
 * Outbound commands sent up the WebSocket to the server.
 *
 * Mirrors `market_domain::ClientCommand`. JSON-serialized via
 * `#[serde(tag = "type", rename_all = "snake_case")]`.
 */
export type ClientCommand =
  | { type: "set_mode"; mode: RoutingMode }
  | { type: "manual_order"; instrument: string; units: number };

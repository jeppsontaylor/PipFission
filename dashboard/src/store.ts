// Reducer + hook that turns the WebSocket event stream into a React-friendly
// store. Keeps the live state (latest prices per instrument, latest account,
// rolling histories for charts and sparklines, recent transactions, and
// connection health) out of the component tree.

import { useEffect, useMemo, useReducer, useRef } from "react";
import type {
  AccountSnapshot,
  AlpacaAccountSnapshot,
  ClientCommand,
  ConnStatus,
  EstimateTick,
  ModelFitness,
  OrderBookSnapshot,
  PaperBookSnapshot,
  PaperFillEvent,
  PriceTick,
  Reconciliation,
  RoutingMode,
  ServerEvent,
  SignalDirection,
  Snapshot,
  StrategySignal,
  TransactionEvent,
} from "./types";
import {
  DEFAULT_HISTORY_LIMIT,
  fetchFillHistory,
  fetchPriceHistory,
  fetchSignalHistory,
  type FillHistoryPoint,
  type PriceHistoryPoint,
  type SignalHistoryPoint,
} from "./historyApi";
import {
  applyChampionChanged,
  applyChampionLoadFailed,
  applyChampionSignal,
  applyTraderDecision,
  initialMLState,
  type MLState,
} from "./mlState";
import { LiveSocket, defaultWsUrl, type WsStatus } from "./ws";

/// Every plot buffer must be at least this big so charts can show a
/// rolling window of ≥1500 data points once filled. Until they hit
/// 1500 they show whatever is available. Single source of truth.
const PLOT_ROLLING_WINDOW = 1800;

const PRICE_HISTORY_LIMIT = PLOT_ROLLING_WINDOW;
const SIGNAL_HISTORY_LIMIT = PLOT_ROLLING_WINDOW; // confidence trace
const NAV_HISTORY_LIMIT = PLOT_ROLLING_WINDOW;
const EQUITY_HISTORY_LIMIT = PLOT_ROLLING_WINDOW;
const TX_LIMIT = 200;
const PAPER_FILLS_LIMIT = 500;
const FITNESS_HISTORY_LIMIT = 200;
const REJECTION_LIMIT = 100;
const REJECTION_RATE_WINDOW_MS = 60_000; // 1 minute
const PRICE_TIMES_RING = 60; // for tick-rate / ETA estimation

export interface PricePoint {
  t: number; // ms since epoch
  bid: number;
  ask: number;
  mid: number;
}

export interface NavPoint {
  t: number;
  estimated: number;
  actual: number;
  drift: number;
}

export interface EquityPoint {
  t: number;
  paper: number;
  oanda: number | null;
}

export interface FitnessPoint {
  t: number;
  instrument: string;
  version: number;
  oosAccuracy: number;
  oosLogLoss: number;
  oosSharpe: number;
  trainAccuracy: number;
}

export interface SignalPoint {
  t: number;
  direction: "long" | "flat" | "short";
  confidence: number;
  // Signed confidence: long → +c, short → -c, flat → 0. Easier to plot.
  signedConfidence: number;
}

export interface RejectionEntry {
  t: number;
  instrument: string | null;
  reason: string;
  kind: string;
}

export interface State {
  ready: boolean;
  accountId: string | null;
  environment: string | null;
  instruments: string[];
  prices: Record<string, PriceTick>;
  priceHistory: Record<string, PricePoint[]>;
  account: AccountSnapshot | null;
  estimate: EstimateTick | null;
  navHistory: NavPoint[];
  transactions: TransactionEvent[];
  connections: ConnStatus | null;
  wsStatus: WsStatus;

  // === Sprint 2 additions ===
  /// Currently active routing mode reported by the server (set on mode_ack).
  /// Defaults to "internal" before any ack lands.
  mode: RoutingMode;
  /// Pending mode change (set when we send set_mode, cleared on mode_ack).
  pendingMode: RoutingMode | null;
  /// Last mode ack — useful for surfacing rejection reasons.
  lastModeAck: { at: number; error: string | null } | null;
  /// Per-instrument latest signal.
  signals: Record<string, StrategySignal>;
  /// Per-instrument rolling history of signals — drives the live chart's
  /// confidence trace + threshold band.
  signalHistory: Record<string, SignalPoint[]>;
  /// Per-instrument latest fitness.
  fitness: Record<string, ModelFitness>;
  /// Rolling history of fitness updates (for the line chart).
  fitnessHistory: FitnessPoint[];
  /// Per-instrument number of feature samples observed (training progress).
  featureCounts: Record<string, number>;
  /// Per-instrument **uncapped** total tick count. Used by the data
  /// header panel — distinct from `priceHistory` which is rolling.
  priceCounts: Record<string, number>;
  /// Per-instrument total signals received.
  signalCounts: Record<string, number>;
  /// Per-instrument total paper fills.
  fillCounts: Record<string, number>;
  /// Last N price arrival times per instrument — for tick-rate / ETA.
  priceTimes: Record<string, number[]>;
  /// Latest internal paper book snapshot.
  paperBook: PaperBookSnapshot | null;
  /// History of paper-book equity (for the equity-curve chart).
  equityHistory: EquityPoint[];
  /// Internal paper fills.
  paperFills: PaperFillEvent[];
  /// Latest reconciliation event.
  reconciliation: Reconciliation | null;
  /// Recent OANDA order rejections (ORDER_CANCEL transactions). Used by
  /// the RejectionBanner to surface MARKET_HALTED + similar errors.
  recentRejections: RejectionEntry[];
  /// Latest Alpaca paper account snapshot.
  alpacaAccount: AlpacaAccountSnapshot | null;
  /// Latest order book snapshot per instrument (Alpaca crypto only).
  orderBooks: Record<string, OrderBookSnapshot>;
  /// ML pipeline state — see mlState.ts. Champion status, recent
  /// champion signals + trader decisions per instrument.
  ml: MLState;
}

const initial: State = {
  ready: false,
  accountId: null,
  environment: null,
  instruments: [],
  prices: {},
  priceHistory: {},
  account: null,
  estimate: null,
  navHistory: [],
  transactions: [],
  connections: null,
  wsStatus: "connecting",

  mode: "internal",
  pendingMode: null,
  lastModeAck: null,
  signals: {},
  signalHistory: {},
  fitness: {},
  priceCounts: {},
  signalCounts: {},
  fillCounts: {},
  priceTimes: {},
  fitnessHistory: [],
  featureCounts: {},
  paperBook: null,
  equityHistory: [],
  paperFills: [],
  reconciliation: null,
  recentRejections: [],
  alpacaAccount: null,
  orderBooks: {},
  ml: initialMLState,
};

type Action =
  | { type: "ws_status"; status: WsStatus }
  | { type: "event"; event: ServerEvent }
  | { type: "request_mode"; mode: RoutingMode }
  | {
      type: "prefill_price";
      instrument: string;
      points: PriceHistoryPoint[];
    }
  | {
      type: "prefill_signal";
      instrument: string;
      points: SignalHistoryPoint[];
    }
  | { type: "prefill_fills"; points: FillHistoryPoint[] };

function pushBounded<T>(arr: T[], item: T, limit: number): T[] {
  if (arr.length >= limit) {
    const out = arr.slice(arr.length - limit + 1);
    out.push(item);
    return out;
  }
  return [...arr, item];
}

function applySnapshot(s: State, snap: Snapshot): State {
  const alpacaAccount = snap.alpaca ?? s.alpacaAccount;
  return _applySnapshot(s, snap, alpacaAccount);
}

function _applySnapshot(
  s: State,
  snap: Snapshot,
  alpacaAccount: AlpacaAccountSnapshot | null,
): State {
  const prices: Record<string, PriceTick> = {};
  const priceHistory: Record<string, PricePoint[]> = { ...s.priceHistory };
  for (const p of snap.prices) {
    prices[p.instrument] = p;
    if (!priceHistory[p.instrument]) priceHistory[p.instrument] = [];
    priceHistory[p.instrument] = pushBounded(
      priceHistory[p.instrument],
      { t: Date.parse(p.time), bid: p.bid, ask: p.ask, mid: p.mid },
      PRICE_HISTORY_LIMIT,
    );
  }
  const navHistory: NavPoint[] = snap.estimate
    ? [
        {
          t: Date.parse(snap.estimate.time),
          estimated: snap.estimate.estimated_balance,
          actual: snap.estimate.actual_balance,
          drift: snap.estimate.drift,
        },
      ]
    : [];
  return {
    ...s,
    prices,
    priceHistory,
    account: snap.account,
    estimate: snap.estimate,
    navHistory,
    transactions: snap.recent_transactions.slice(-TX_LIMIT),
    connections: snap.connections,
    alpacaAccount,
  };
}

function reduce(s: State, a: Action): State {
  switch (a.type) {
    case "ws_status":
      return { ...s, wsStatus: a.status };
    case "event": {
      const e = a.event;
      switch (e.type) {
        case "hello":
          return applySnapshot(
            {
              ...s,
              ready: true,
              accountId: e.account_id,
              environment: e.environment,
              instruments: e.instruments,
            },
            e.snapshot,
          );
        case "price": {
          const tick: PriceTick = {
            instrument: e.instrument,
            time: e.time,
            bid: e.bid,
            ask: e.ask,
            mid: e.mid,
            spread: e.spread,
            closeout_bid: e.closeout_bid,
            closeout_ask: e.closeout_ask,
            status: e.status,
          };
          const tMs = Date.parse(tick.time);
          const nextHist = pushBounded(
            s.priceHistory[tick.instrument] ?? [],
            { t: tMs, bid: tick.bid, ask: tick.ask, mid: tick.mid },
            PRICE_HISTORY_LIMIT,
          );
          const nextTimes = pushBounded(
            s.priceTimes[tick.instrument] ?? [],
            tMs,
            PRICE_TIMES_RING,
          );
          const nextCount = (s.priceCounts[tick.instrument] ?? 0) + 1;
          return {
            ...s,
            prices: { ...s.prices, [tick.instrument]: tick },
            priceHistory: { ...s.priceHistory, [tick.instrument]: nextHist },
            priceCounts: { ...s.priceCounts, [tick.instrument]: nextCount },
            priceTimes: { ...s.priceTimes, [tick.instrument]: nextTimes },
          };
        }
        case "account": {
          const acct: AccountSnapshot = {
            time: e.time,
            nav: e.nav,
            balance: e.balance,
            unrealized_pl: e.unrealized_pl,
            realized_pl: e.realized_pl,
            margin_used: e.margin_used,
            margin_available: e.margin_available,
            open_position_count: e.open_position_count,
            open_trade_count: e.open_trade_count,
            pending_order_count: e.pending_order_count,
            leverage: e.leverage,
            currency: e.currency,
          };
          return { ...s, account: acct };
        }
        case "transaction": {
          const tx: TransactionEvent = {
            id: e.id,
            time: e.time,
            kind: e.kind,
            instrument: e.instrument,
            units: e.units,
            price: e.price,
            pl: e.pl,
            reason: e.reason,
            raw: e.raw,
          };
          // Surface order-cancel / reject transactions in the rejection
          // panel so MARKET_HALTED and similar are unmissable.
          let recentRejections = s.recentRejections;
          if (
            tx.kind === "ORDER_CANCEL" ||
            tx.kind === "MARKET_ORDER_REJECT" ||
            tx.kind === "ORDER_REJECT"
          ) {
            const reason = tx.reason ?? "UNKNOWN";
            recentRejections = pushBounded(
              recentRejections,
              { t: Date.parse(tx.time), instrument: tx.instrument, reason, kind: tx.kind },
              REJECTION_LIMIT,
            );
          }
          return {
            ...s,
            transactions: pushBounded(s.transactions, tx, TX_LIMIT),
            recentRejections,
          };
        }
        case "estimate": {
          const est: EstimateTick = {
            time: e.time,
            estimated_balance: e.estimated_balance,
            actual_balance: e.actual_balance,
            drift: e.drift,
            drift_bps: e.drift_bps,
          };
          const point: NavPoint = {
            t: Date.parse(est.time),
            estimated: est.estimated_balance,
            actual: est.actual_balance,
            drift: est.drift,
          };
          return {
            ...s,
            estimate: est,
            navHistory: pushBounded(s.navHistory, point, NAV_HISTORY_LIMIT),
          };
        }
        case "status": {
          const conn: ConnStatus = {
            pricing_stream: e.pricing_stream,
            transaction_stream: e.transaction_stream,
            account_poll: e.account_poll,
          };
          return { ...s, connections: conn };
        }
        case "features": {
          // We don't render every feature vector; just count them per
          // instrument so the ML status panel can show training progress.
          const next = { ...s.featureCounts };
          next[e.instrument] = (next[e.instrument] ?? 0) + 1;
          return { ...s, featureCounts: next };
        }
        case "signal": {
          const sig: StrategySignal = {
            instrument: e.instrument,
            time: e.time,
            direction: e.direction,
            confidence: e.confidence,
            probs: e.probs,
            model_id: e.model_id,
            model_version: e.model_version,
          };
          const sign = sig.direction === "long" ? 1 : sig.direction === "short" ? -1 : 0;
          const point: SignalPoint = {
            t: Date.parse(sig.time),
            direction: sig.direction,
            confidence: sig.confidence,
            signedConfidence: sign * sig.confidence,
          };
          const nextHist = pushBounded(
            s.signalHistory[sig.instrument] ?? [],
            point,
            SIGNAL_HISTORY_LIMIT,
          );
          const nextSigCount = (s.signalCounts[sig.instrument] ?? 0) + 1;
          return {
            ...s,
            signals: { ...s.signals, [sig.instrument]: sig },
            signalHistory: { ...s.signalHistory, [sig.instrument]: nextHist },
            signalCounts: { ...s.signalCounts, [sig.instrument]: nextSigCount },
          };
        }
        case "fitness": {
          const f: ModelFitness = {
            instrument: e.instrument,
            model_id: e.model_id,
            model_version: e.model_version,
            trained_at: e.trained_at,
            train: e.train,
            oos: e.oos,
            samples_seen: e.samples_seen,
            train_window: e.train_window,
            oos_window: e.oos_window,
          };
          const point: FitnessPoint = {
            t: Date.parse(f.trained_at),
            instrument: f.instrument,
            version: f.model_version,
            oosAccuracy: f.oos.accuracy,
            oosLogLoss: f.oos.log_loss,
            oosSharpe: f.oos.sharpe,
            trainAccuracy: f.train.accuracy,
          };
          return {
            ...s,
            fitness: { ...s.fitness, [f.instrument]: f },
            fitnessHistory: pushBounded(s.fitnessHistory, point, FITNESS_HISTORY_LIMIT),
          };
        }
        case "paper_book": {
          const pb: PaperBookSnapshot = {
            mode: e.mode,
            version: e.version,
            time: e.time,
            cash: e.cash,
            equity: e.equity,
            realized_pl: e.realized_pl,
            unrealized_pl: e.unrealized_pl,
            positions: e.positions,
          };
          const point: EquityPoint = {
            t: Date.parse(pb.time),
            paper: pb.equity,
            oanda: s.estimate?.actual_balance ?? null,
          };
          return {
            ...s,
            paperBook: pb,
            equityHistory: pushBounded(s.equityHistory, point, EQUITY_HISTORY_LIMIT),
          };
        }
        case "paper_fill": {
          const f: PaperFillEvent = {
            mode: e.mode,
            order_id: e.order_id,
            instrument: e.instrument,
            units: e.units,
            price: e.price,
            fee: e.fee,
            time: e.time,
          };
          const nextFillCount = (s.fillCounts[f.instrument] ?? 0) + 1;
          return {
            ...s,
            paperFills: pushBounded(s.paperFills, f, PAPER_FILLS_LIMIT),
            fillCounts: { ...s.fillCounts, [f.instrument]: nextFillCount },
          };
        }
        case "reconciliation": {
          const r: Reconciliation = {
            time: e.time,
            mode: e.mode,
            internal_paper_equity: e.internal_paper_equity,
            oanda_actual_equity: e.oanda_actual_equity,
            oanda_minus_internal_bp: e.oanda_minus_internal_bp,
          };
          return { ...s, reconciliation: r };
        }
        case "mode_ack": {
          return {
            ...s,
            mode: e.mode,
            pendingMode: null,
            lastModeAck: { at: Date.parse(e.effective_at), error: e.error },
          };
        }
        case "order_book": {
          const ob: OrderBookSnapshot = {
            instrument: e.instrument,
            time: e.time,
            bids: e.bids,
            asks: e.asks,
          };
          return { ...s, orderBooks: { ...s.orderBooks, [ob.instrument]: ob } };
        }
        case "bar10s":
          // 10s bars are persisted server-side and consumed by the
          // live-inference + live-trader runners. The dashboard
          // currently doesn't render them directly (the price chart
          // already shows tick-level data), but the variant is
          // recognised here so it doesn't fall through to "unknown".
          return s;
        case "champion_signal": {
          const ev = {
            instrument: e.instrument,
            time: e.time,
            p_long: e.p_long,
            p_short: e.p_short,
            p_take: e.p_take,
            calibrated: e.calibrated,
            model_id: e.model_id,
            kind: e.kind,
          };
          return { ...s, ml: applyChampionSignal(s.ml, ev) };
        }
        case "champion_changed":
          return {
            ...s,
            ml: applyChampionChanged(s.ml, {
              model_id: e.model_id,
              kind: e.kind,
              n_features: e.n_features,
            }),
          };
        case "champion_load_failed":
          return {
            ...s,
            ml: applyChampionLoadFailed(s.ml, { reason: e.reason }),
          };
        case "trader_decision": {
          const ev = {
            instrument: e.instrument,
            time: e.time,
            bar_idx: e.bar_idx,
            action: e.action,
            reason: e.reason,
            price: e.price,
            realized_r: e.realized_r,
            params_id: e.params_id,
            model_id: e.model_id,
          };
          return { ...s, ml: applyTraderDecision(s.ml, ev) };
        }
      }
      return s;
    }
    case "request_mode":
      return { ...s, pendingMode: a.mode };
    case "prefill_price": {
      // Merge historical points with whatever live ticks already
      // arrived during the fetch. Dedup by time_ms and keep the most
      // recent PLOT_ROLLING_WINDOW after sort.
      const existing = s.priceHistory[a.instrument] ?? [];
      const merged = mergePricePoints(a.points, existing);
      return {
        ...s,
        priceHistory: { ...s.priceHistory, [a.instrument]: merged },
      };
    }
    case "prefill_signal": {
      const existing = s.signalHistory[a.instrument] ?? [];
      const merged = mergeSignalPoints(a.points, existing);
      return {
        ...s,
        signalHistory: { ...s.signalHistory, [a.instrument]: merged },
      };
    }
    case "prefill_fills": {
      const merged = mergeFills(a.points, s.paperFills);
      // Re-derive per-instrument fill counts from the merged set so
      // they reflect history (the live counter started at 0 on connect).
      const fillCounts: Record<string, number> = { ...s.fillCounts };
      for (const f of a.points) {
        fillCounts[f.instrument] = Math.max(
          fillCounts[f.instrument] ?? 0,
          (fillCounts[f.instrument] ?? 0) + 1,
        );
      }
      return { ...s, paperFills: merged, fillCounts };
    }
  }
}

// ---- prefill merge helpers -------------------------------------------

function mergePricePoints(
  hist: PriceHistoryPoint[],
  live: PricePoint[],
): PricePoint[] {
  const seen = new Set<number>();
  const out: PricePoint[] = [];
  // Historical first (oldest first), then any live tick whose t isn't
  // already covered. Stable insertion preserves arrival order for
  // ties — the live array drives the leading edge.
  for (const h of hist) {
    if (seen.has(h.time_ms)) continue;
    seen.add(h.time_ms);
    out.push({ t: h.time_ms, bid: h.bid, ask: h.ask, mid: h.mid });
  }
  for (const p of live) {
    if (seen.has(p.t)) continue;
    seen.add(p.t);
    out.push(p);
  }
  out.sort((a, b) => a.t - b.t);
  return out.length > PRICE_HISTORY_LIMIT
    ? out.slice(out.length - PRICE_HISTORY_LIMIT)
    : out;
}

function mergeSignalPoints(
  hist: SignalHistoryPoint[],
  live: SignalPoint[],
): SignalPoint[] {
  const seen = new Set<number>();
  const out: SignalPoint[] = [];
  for (const h of hist) {
    if (seen.has(h.time_ms)) continue;
    seen.add(h.time_ms);
    const dir = h.direction as SignalDirection;
    const sign = dir === "long" ? 1 : dir === "short" ? -1 : 0;
    out.push({
      t: h.time_ms,
      direction: dir,
      confidence: h.confidence,
      signedConfidence: sign * h.confidence,
    });
  }
  for (const p of live) {
    if (seen.has(p.t)) continue;
    seen.add(p.t);
    out.push(p);
  }
  out.sort((a, b) => a.t - b.t);
  return out.length > SIGNAL_HISTORY_LIMIT
    ? out.slice(out.length - SIGNAL_HISTORY_LIMIT)
    : out;
}

type Dispatch = (a: Action) => void;

/**
 * Fire one parallel /api/history fetch per instrument for prices and
 * (for already-trained instruments) signals, plus one global fill
 * fetch. Each fetch dispatches its result via `prefill_*` actions which
 * the reducer merges with whatever live data has already arrived.
 *
 * We dedupe on the `prefilledRef` set so a WS reconnect doesn't
 * stampede the API.
 */
function kickOffPrefill(
  instruments: string[],
  alreadyDone: Set<string>,
  dispatch: Dispatch,
): void {
  const todo = instruments.filter((i) => !alreadyDone.has(i));
  for (const inst of todo) alreadyDone.add(inst);

  // Fills: a single global fetch covers every instrument.
  if (!alreadyDone.has("__fills__")) {
    alreadyDone.add("__fills__");
    fetchFillHistory(DEFAULT_HISTORY_LIMIT)
      .then((points) => dispatch({ type: "prefill_fills", points }))
      .catch((err) => console.warn("history(fill) failed", err));
  }

  for (const inst of todo) {
    fetchPriceHistory(inst, DEFAULT_HISTORY_LIMIT)
      .then((points) => {
        if (points.length > 0) dispatch({ type: "prefill_price", instrument: inst, points });
      })
      .catch((err) => console.warn(`history(price) ${inst} failed`, err));

    // Signals: harmless to attempt for untrained instruments — server
    // returns an empty array. Saves a "do they have a model?" round trip.
    fetchSignalHistory(inst, DEFAULT_HISTORY_LIMIT)
      .then((points) => {
        if (points.length > 0) dispatch({ type: "prefill_signal", instrument: inst, points });
      })
      .catch((err) => console.warn(`history(signal) ${inst} failed`, err));
  }
}

function mergeFills(
  hist: FillHistoryPoint[],
  live: PaperFillEvent[],
): PaperFillEvent[] {
  // Dedup by order_id (server's primary key for fills).
  const byId = new Map<string, PaperFillEvent>();
  for (const h of hist) {
    byId.set(h.order_id, {
      mode: h.mode === "oanda_practice" ? "oanda_practice" : "internal",
      order_id: h.order_id,
      instrument: h.instrument,
      units: h.units,
      price: h.price,
      fee: h.fee,
      time: new Date(h.time_ms).toISOString(),
    });
  }
  for (const f of live) byId.set(f.order_id, f);
  const out = [...byId.values()].sort(
    (a, b) => Date.parse(a.time) - Date.parse(b.time),
  );
  return out.length > PAPER_FILLS_LIMIT
    ? out.slice(out.length - PAPER_FILLS_LIMIT)
    : out;
}

export interface LiveStore {
  state: State;
  /** Send a `set_mode` command to the server. The store flips
   *  `pendingMode` immediately and clears it when `mode_ack` arrives. */
  setMode(mode: RoutingMode): void;
  /** Fire a manual order via the WS. `units` is signed (positive = buy,
   *  negative = sell). Used by the test panel — same dispatch path as
   *  strategy-emitted intents. */
  manualOrder(instrument: string, units: number): void;
}

export function useLiveStore(): LiveStore {
  const [state, dispatch] = useReducer(reduce, initial);
  const sockRef = useRef<LiveSocket | null>(null);

  // Track which instruments we've already prefilled so a WS reconnect
  // (which re-fires `hello`) doesn't trigger a redundant fetch storm.
  const prefilledRef = useRef<Set<string>>(new Set());
  // Cancel in-flight fetches if the component unmounts.
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const sock = new LiveSocket(defaultWsUrl());
    sockRef.current = sock;
    abortRef.current = new AbortController();
    const offEvent = sock.on((event) => {
      dispatch({ type: "event", event });
      if (event.type === "hello") {
        kickOffPrefill(event.instruments, prefilledRef.current, dispatch);
      }
    });
    const offStatus = sock.onStatus((status) => dispatch({ type: "ws_status", status }));
    sock.start();
    return () => {
      offEvent();
      offStatus();
      sock.stop();
      abortRef.current?.abort();
    };
  }, []);

  return useMemo<LiveStore>(
    () => ({
      state,
      setMode(mode: RoutingMode) {
        const cmd: ClientCommand = { type: "set_mode", mode };
        dispatch({ type: "request_mode", mode });
        sockRef.current?.send(cmd);
      },
      manualOrder(instrument: string, units: number) {
        if (units === 0) return;
        const cmd: ClientCommand = { type: "manual_order", instrument, units };
        sockRef.current?.send(cmd);
      },
    }),
    [state],
  );
}

export function fmtCcy(n: number | null | undefined, currency = "USD"): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(n);
}

export function fmtPx(n: number | null | undefined, decimals = 5): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(decimals);
}

/**
 * Estimate the per-instrument tick rate in Hz from the recent
 * arrival timestamps stored in `state.priceTimes[instrument]`.
 * Returns 0 when there are too few samples to estimate (we don't
 * extrapolate from a single tick).
 */
export function rateHz(times: number[] | undefined): number {
  if (!times || times.length < 2) return 0;
  const span_ms = times[times.length - 1] - times[0];
  if (span_ms <= 0) return 0;
  return ((times.length - 1) * 1000) / span_ms;
}

/**
 * Format a number of seconds as a short human duration —
 * "12s" / "4m 32s" / "1h 07m". `null` when input isn't finite.
 */
export function fmtDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) return "—";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const sec = s % 60;
  if (m < 60) return `${m}m ${sec.toString().padStart(2, "0")}s`;
  const h = Math.floor(m / 60);
  const mr = m % 60;
  return `${h}h ${mr.toString().padStart(2, "0")}m`;
}

export function instrumentDecimals(inst: string): number {
  // Crypto pairs use 2 decimals (BTC ~$77k, ETH ~$3.5k, level of precision
  // beyond cents adds noise to the UI without information).
  if (inst.includes("/")) return 2;
  // JPY forex pairs: 3 decimals; everything else forex: 5 decimals.
  return inst.endsWith("JPY") ? 3 : 5;
}

/** Convenience: a stable list of instruments based on what's been seen. */
export function useInstrumentList(state: State): string[] {
  return useMemo(() => {
    if (state.instruments.length > 0) return state.instruments;
    return Object.keys(state.prices).sort();
  }, [state.instruments, state.prices]);
}

/** Number of rejections in the last 60 seconds + the most-frequent reason. */
export function useRejectionRate(state: State): {
  count: number;
  topReason: string | null;
  topReasonCount: number;
} {
  return useMemo(() => {
    const cutoff = Date.now() - REJECTION_RATE_WINDOW_MS;
    const recent = state.recentRejections.filter((r) => r.t >= cutoff);
    const byReason = new Map<string, number>();
    for (const r of recent) {
      byReason.set(r.reason, (byReason.get(r.reason) ?? 0) + 1);
    }
    let topReason: string | null = null;
    let topReasonCount = 0;
    for (const [reason, n] of byReason.entries()) {
      if (n > topReasonCount) {
        topReason = reason;
        topReasonCount = n;
      }
    }
    return { count: recent.length, topReason, topReasonCount };
  }, [state.recentRejections]);
}

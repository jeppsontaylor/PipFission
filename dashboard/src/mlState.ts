// Pure-function state slice for the ML pipeline (champion + trader).
// Lives in its own file so the reducer logic can be unit-tested without
// pulling in the WebSocket / hook plumbing in `store.ts`.
//
// The store exposes one merged `MLState` object and a single
// `applyMlEvent(state, event)` reducer. New WS event variants flow
// through here.

import type {
  ChampionChanged,
  ChampionLoadFailed,
  ChampionSignal,
  TraderDecision,
} from "./types";

/** Bounded rolling window for both champion-signal and trader-decision
 *  histories. Keeps the dashboard chart cheap. */
export const ML_HISTORY_LIMIT = 600;

export interface ChampionPoint {
  /** Bar-close timestamp in ms. */
  t: number;
  p_long: number;
  p_short: number;
  p_take: number;
  /** Lower-cased predictor kind ("onnx" or "fallback"). */
  kind: string;
}

export interface DecisionPoint {
  t: number;
  action: string;
  reason: string;
  price: number;
  realized_r: number | null;
  bar_idx: number;
}

export interface ChampionStatus {
  model_id: string;
  /** "onnx" | "fallback" | "uninitialised" */
  kind: string;
  /** ms-since-epoch when the last `champion_changed` arrived. */
  last_changed_at: number | null;
  /** Most-recent `champion_load_failed.reason`, or null. */
  last_error: string | null;
}

export interface MLState {
  championStatus: ChampionStatus;
  /** Latest champion signal per instrument. */
  championSignals: Record<string, ChampionSignal>;
  /** Bounded rolling history of champion signals per instrument. */
  championHistory: Record<string, ChampionPoint[]>;
  /** Latest trader decision per instrument (any action). */
  traderDecisions: Record<string, TraderDecision>;
  /** Bounded rolling history of trader decisions per instrument. */
  traderDecisionHistory: Record<string, DecisionPoint[]>;
}

export const initialMLState: MLState = {
  championStatus: {
    model_id: "uninitialised",
    kind: "uninitialised",
    last_changed_at: null,
    last_error: null,
  },
  championSignals: {},
  championHistory: {},
  traderDecisions: {},
  traderDecisionHistory: {},
};

/** Append `pt` to the rolling list at `key`, capped at ML_HISTORY_LIMIT. */
function pushRolling<T>(map: Record<string, T[]>, key: string, pt: T): Record<string, T[]> {
  const cur = map[key] ?? [];
  const next = cur.length >= ML_HISTORY_LIMIT ? cur.slice(1) : cur.slice();
  next.push(pt);
  return { ...map, [key]: next };
}

export function applyChampionSignal(s: MLState, ev: ChampionSignal): MLState {
  const t = Date.parse(ev.time);
  if (Number.isNaN(t)) return s;
  const point: ChampionPoint = {
    t,
    p_long: ev.p_long,
    p_short: ev.p_short,
    p_take: ev.p_take,
    kind: ev.kind,
  };
  return {
    ...s,
    championSignals: { ...s.championSignals, [ev.instrument]: ev },
    championHistory: pushRolling(s.championHistory, ev.instrument, point),
  };
}

export function applyChampionChanged(s: MLState, ev: ChampionChanged): MLState {
  return {
    ...s,
    championStatus: {
      model_id: ev.model_id,
      kind: ev.kind,
      last_changed_at: Date.now(),
      // Successful swap clears any prior error.
      last_error: null,
    },
  };
}

export function applyChampionLoadFailed(s: MLState, ev: ChampionLoadFailed): MLState {
  return {
    ...s,
    championStatus: {
      ...s.championStatus,
      last_error: ev.reason,
    },
  };
}

export function applyTraderDecision(s: MLState, ev: TraderDecision): MLState {
  const t = Date.parse(ev.time);
  if (Number.isNaN(t)) return s;
  const point: DecisionPoint = {
    t,
    action: ev.action,
    reason: ev.reason,
    price: ev.price,
    realized_r: ev.realized_r,
    bar_idx: ev.bar_idx,
  };
  return {
    ...s,
    traderDecisions: { ...s.traderDecisions, [ev.instrument]: ev },
    traderDecisionHistory: pushRolling(s.traderDecisionHistory, ev.instrument, point),
  };
}

/** Pretty-printed action label for the dashboard. */
export function actionLabel(action: string): string {
  switch (action) {
    case "open_long":
      return "Open Long";
    case "open_short":
      return "Open Short";
    case "close":
      return "Close";
    case "skip":
      return "Skip";
    default:
      return action;
  }
}

/** Reason → short human-readable label. */
export function reasonLabel(reason: string): string {
  const map: Record<string, string> = {
    signal: "Signal",
    stop_loss: "Stop loss",
    take_profit: "Take profit",
    trailing_stop: "Trailing stop",
    max_hold: "Max hold",
    reverse: "Reverse signal",
    spread_too_wide: "Spread too wide",
    stale_data: "Stale data",
    daily_loss_kill: "Daily loss kill",
    drawdown_pause: "Drawdown pause",
    cooldown: "Cooldown",
    below_threshold: "Below threshold",
    min_hold: "Min hold",
  };
  return map[reason] ?? reason;
}

/** True iff `action` corresponds to a position-change. */
export function isOrderingAction(action: string): boolean {
  return action === "open_long" || action === "open_short" || action === "close";
}

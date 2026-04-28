// Pure helpers for the live-trade tape: aggregate stats over the
// `trade_ledger` rows and a cumulative-R curve for the sparkline. Kept
// separate from the React component so the math is unit testable.

import type { TradeLedgerRow } from "./mlApi";

export interface LedgerSummary {
  count: number;
  wins: number;
  losses: number;
  hitRate: number;
  /** Sum of realized_r across every closed trade. */
  cumulativeR: number;
  /** Net dollar P/L = sum(qty * (exit-entry) for long, -qty*(exit-entry) for short)
   *  minus fees and slippage. Aggregates raw price units, not fractional R. */
  netPL: number;
  /** Σ(positive returns) / |Σ(negative returns)|. Infinity when no
   *  losers, zero when no winners. Capped at 999 for display. */
  profitFactor: number;
  /** Largest single positive realized_r. */
  bestTradeR: number | null;
  /** Largest negative realized_r. */
  worstTradeR: number | null;
  /** Bucket counts for the exit-reason histogram. */
  reasons: Record<string, number>;
}

export function summarizeLedger(rows: TradeLedgerRow[]): LedgerSummary {
  if (rows.length === 0) {
    return {
      count: 0,
      wins: 0,
      losses: 0,
      hitRate: 0,
      cumulativeR: 0,
      netPL: 0,
      profitFactor: 0,
      bestTradeR: null,
      worstTradeR: null,
      reasons: {},
    };
  }
  let wins = 0;
  let losses = 0;
  let cumR = 0;
  let netPL = 0;
  let posR = 0;
  let negR = 0;
  let best = -Infinity;
  let worst = Infinity;
  const reasons: Record<string, number> = {};
  for (const r of rows) {
    cumR += r.realized_r;
    const sideMul = r.side >= 0 ? 1 : -1;
    netPL += sideMul * r.qty * (r.exit_px - r.entry_px) - r.fee - r.slip;
    if (r.realized_r > 0) {
      wins += 1;
      posR += r.realized_r;
    } else if (r.realized_r < 0) {
      losses += 1;
      negR += -r.realized_r;
    }
    if (r.realized_r > best) best = r.realized_r;
    if (r.realized_r < worst) worst = r.realized_r;
    reasons[r.exit_reason] = (reasons[r.exit_reason] ?? 0) + 1;
  }
  let pf: number;
  if (negR === 0) pf = posR > 0 ? 999 : 0;
  else pf = Math.min(999, posR / negR);
  const decided = wins + losses;
  return {
    count: rows.length,
    wins,
    losses,
    hitRate: decided > 0 ? wins / decided : 0,
    cumulativeR: cumR,
    netPL,
    profitFactor: pf,
    bestTradeR: Number.isFinite(best) ? best : null,
    worstTradeR: Number.isFinite(worst) ? worst : null,
    reasons,
  };
}

export interface CumPoint {
  t: number;
  cum_r: number;
}

/**
 * Cumulative-R curve in chronological order. Caller passes rows that
 * may already be ascending (the REST endpoint returns ascending) but we
 * sort defensively to keep the chart well-formed if a caller hands us
 * a different ordering.
 */
export function cumulativeRCurve(rows: TradeLedgerRow[]): CumPoint[] {
  const sorted = [...rows].sort((a, b) => a.ts_out_ms - b.ts_out_ms);
  const out: CumPoint[] = [];
  let acc = 0;
  for (const r of sorted) {
    acc += r.realized_r;
    out.push({ t: r.ts_out_ms, cum_r: acc });
  }
  return out;
}

export function sideLabel(side: number): "Long" | "Short" | "Flat" {
  if (side > 0) return "Long";
  if (side < 0) return "Short";
  return "Flat";
}

export function sideTone(side: number): string {
  if (side > 0) return "text-emerald-300";
  if (side < 0) return "text-rose-300";
  return "text-slate-400";
}

const EXIT_REASON_LABELS: Record<string, string> = {
  take_profit: "Take profit",
  stop_loss: "Stop loss",
  max_hold: "Max hold",
  trailing_stop: "Trailing stop",
  signal_flip: "Signal flip",
  manual: "Manual",
  shutdown: "Shutdown",
};

export function exitReasonLabel(reason: string): string {
  return EXIT_REASON_LABELS[reason] ?? reason;
}

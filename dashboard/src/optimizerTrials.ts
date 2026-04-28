// Pure presentation helpers for the optimizer-trial leaderboard. Kept
// out of the React component so the formatting + ranking logic is unit
// testable without rendering. The component (OptimizerTrialsTable) is a
// thin shell over these.
//
// The shape of `params_json` is whatever the Python trader optimizer
// serialised — a flat dict of TraderParams fields (long_threshold,
// short_threshold, stop_loss_atr, …). We don't enforce the schema
// here; the table just shows whatever a trial saved.

import type { OptimizerTrialRow } from "./mlApi";

export interface ParsedParams {
  /** Raw key/value pairs after JSON.parse. */
  raw: Record<string, unknown>;
  /** Compact one-line summary for the table — `lt=0.62 st=0.58 sl=1.4`. */
  snippet: string;
}

/** Keys we lift into the snippet, in display order. Other keys still
 *  appear in the title-attribute tooltip via `raw`. */
const SNIPPET_KEYS: ReadonlyArray<[string, string]> = [
  ["long_threshold", "lt"],
  ["short_threshold", "st"],
  ["take_threshold", "tt"],
  ["stop_loss_atr", "sl"],
  ["take_profit_atr", "tp"],
  ["min_hold_bars", "minH"],
  ["max_hold_bars", "maxH"],
  ["cooldown_bars", "cd"],
];

/**
 * Safe JSON.parse — never throws. Returns `null` for malformed inputs
 * so the table can render `—` instead of crashing the panel.
 */
export function parseParamsJson(json: string | null | undefined): ParsedParams | null {
  if (!json) return null;
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch {
    return null;
  }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const dict = raw as Record<string, unknown>;
  const parts: string[] = [];
  for (const [key, short] of SNIPPET_KEYS) {
    const v = dict[key];
    if (typeof v === "number" && Number.isFinite(v)) {
      parts.push(`${short}=${formatParamValue(v)}`);
    }
  }
  return { raw: dict, snippet: parts.join(" ") };
}

function formatParamValue(v: number): string {
  if (Number.isInteger(v)) return v.toString();
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

/**
 * Sort trials for the leaderboard. Pareto-optimal first (rank 1 ↑),
 * then by score desc within each rank. `pareto_rank=0` is the
 * "unranked" sentinel the Python optimizer writes before the study
 * completes — those drop to the bottom so finished Pareto hits surface.
 */
export function sortTrialsForLeaderboard(rows: OptimizerTrialRow[]): OptimizerTrialRow[] {
  // Don't mutate the caller's array.
  return [...rows].sort((a, b) => {
    const ra = a.pareto_rank > 0 ? a.pareto_rank : Number.POSITIVE_INFINITY;
    const rb = b.pareto_rank > 0 ? b.pareto_rank : Number.POSITIVE_INFINITY;
    if (ra !== rb) return ra - rb;
    return b.score - a.score;
  });
}

export function topNByScore(rows: OptimizerTrialRow[], n: number): OptimizerTrialRow[] {
  return sortTrialsForLeaderboard(rows).slice(0, Math.max(0, n));
}

export interface TrialsSummary {
  count: number;
  paretoCount: number;
  bestScore: number | null;
  bestSortino: number | null;
  bestDrawdownBp: number | null;
  latestStudy: string | null;
  latestTsMs: number | null;
}

export function summarizeTrials(rows: OptimizerTrialRow[]): TrialsSummary {
  if (rows.length === 0) {
    return {
      count: 0,
      paretoCount: 0,
      bestScore: null,
      bestSortino: null,
      bestDrawdownBp: null,
      latestStudy: null,
      latestTsMs: null,
    };
  }
  let bestScore = -Infinity;
  let bestSortino = -Infinity;
  let bestDrawdownBp = Infinity;
  let paretoCount = 0;
  let latestTsMs = -Infinity;
  let latestStudy: string | null = null;
  for (const r of rows) {
    if (r.score > bestScore) bestScore = r.score;
    if (r.sortino > bestSortino) bestSortino = r.sortino;
    if (r.max_dd_bp < bestDrawdownBp) bestDrawdownBp = r.max_dd_bp;
    if (r.pareto_rank === 1) paretoCount += 1;
    if (r.ts_ms > latestTsMs) {
      latestTsMs = r.ts_ms;
      latestStudy = r.study_id;
    }
  }
  return {
    count: rows.length,
    paretoCount,
    bestScore: Number.isFinite(bestScore) ? bestScore : null,
    bestSortino: Number.isFinite(bestSortino) ? bestSortino : null,
    bestDrawdownBp: Number.isFinite(bestDrawdownBp) ? bestDrawdownBp : null,
    latestStudy,
    latestTsMs: Number.isFinite(latestTsMs) ? latestTsMs : null,
  };
}

/** Tailwind classes for the rank badge. */
export function paretoTone(rank: number): string {
  if (rank === 1) return "bg-emerald-900/50 border-emerald-700/60 text-emerald-200";
  if (rank === 2) return "bg-sky-900/40 border-sky-700/50 text-sky-200";
  if (rank === 3) return "bg-violet-900/40 border-violet-700/50 text-violet-200";
  if (rank > 3) return "bg-slate-800/60 border-slate-700 text-slate-300";
  return "bg-slate-800/40 border-slate-700/60 text-slate-400";
}

export function paretoLabel(rank: number): string {
  if (rank <= 0) return "—";
  return `P${rank}`;
}

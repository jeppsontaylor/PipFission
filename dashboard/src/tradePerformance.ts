// Pure helpers for the TradePerformanceBreakdown panel. Computes
// deeper performance stats than `tradeLedger.ts` — Sharpe/Sortino
// on realised-R series, max drawdown, per-instrument and per-reason
// breakdowns. All functions are deterministic on their inputs so
// they're trivial to unit test without rendering.

import type { TradeLedgerRow } from "./mlApi";

export interface PerfStats {
  count: number;
  wins: number;
  losses: number;
  scratched: number;
  hitRate: number;
  /** Sum of realized_r — fractional return, net of costs. */
  cumulativeR: number;
  avgWinR: number;
  avgLossR: number;
  /** Σwin / |Σloss|. ∞ when no losses; 0 when no wins. */
  profitFactor: number;
  /** Win-vs-loss expectancy: hitRate * avgWin + (1-hitRate) * avgLoss. */
  expectancyR: number;
  /** Sample-stdev annualised Sharpe-like ratio assuming each trade is
   *  one observation. Use comparatively, not absolutely — the unit is
   *  "per trade." */
  sharpePerTrade: number;
  sortinoPerTrade: number;
  /** Largest peak-to-trough drawdown on the cumulative-R series. */
  maxDrawdownR: number;
}

export interface ReasonBucket {
  reason: string;
  count: number;
  cumulativeR: number;
  avgR: number;
}

export interface InstrumentPerf {
  instrument: string;
  count: number;
  cumulativeR: number;
  hitRate: number;
}

/** Aggregate stats across the entire row set. */
export function computePerfStats(rows: TradeLedgerRow[]): PerfStats {
  const empty: PerfStats = {
    count: 0,
    wins: 0,
    losses: 0,
    scratched: 0,
    hitRate: 0,
    cumulativeR: 0,
    avgWinR: 0,
    avgLossR: 0,
    profitFactor: 0,
    expectancyR: 0,
    sharpePerTrade: 0,
    sortinoPerTrade: 0,
    maxDrawdownR: 0,
  };
  if (rows.length === 0) return empty;

  let wins = 0;
  let losses = 0;
  let scratched = 0;
  let sumWin = 0;
  let sumLoss = 0;
  let cumR = 0;
  for (const r of rows) {
    cumR += r.realized_r;
    if (r.realized_r > 0) {
      wins += 1;
      sumWin += r.realized_r;
    } else if (r.realized_r < 0) {
      losses += 1;
      sumLoss += -r.realized_r;
    } else {
      scratched += 1;
    }
  }

  const decided = wins + losses;
  const hitRate = decided > 0 ? wins / decided : 0;
  const avgWinR = wins > 0 ? sumWin / wins : 0;
  const avgLossR = losses > 0 ? -sumLoss / losses : 0;
  const profitFactor =
    sumLoss === 0
      ? wins > 0
        ? Number.POSITIVE_INFINITY
        : 0
      : sumWin / sumLoss;
  const expectancyR = hitRate * avgWinR + (1 - hitRate) * avgLossR;

  // Per-trade Sharpe / Sortino. Assumes each trade is one IID
  // observation. Use as a relative comparison metric, not an absolute
  // (annualising would require a horizon assumption we don't have).
  const mean = cumR / rows.length;
  let varianceSum = 0;
  let downsideSum = 0;
  let downsideN = 0;
  for (const r of rows) {
    const dev = r.realized_r - mean;
    varianceSum += dev * dev;
    if (r.realized_r < 0) {
      downsideSum += r.realized_r * r.realized_r;
      downsideN += 1;
    }
  }
  const std = rows.length > 1 ? Math.sqrt(varianceSum / (rows.length - 1)) : 0;
  const downsideStd = downsideN > 0 ? Math.sqrt(downsideSum / downsideN) : 0;
  const sharpePerTrade = std > 0 ? mean / std : 0;
  const sortinoPerTrade = downsideStd > 0 ? mean / downsideStd : 0;

  // Cumulative-R drawdown (in chronological order — caller passes
  // ascending, we sort defensively to match the chart's behaviour).
  const sorted = [...rows].sort((a, b) => a.ts_out_ms - b.ts_out_ms);
  let peak = 0;
  let cumWalk = 0;
  let maxDd = 0;
  for (const r of sorted) {
    cumWalk += r.realized_r;
    if (cumWalk > peak) peak = cumWalk;
    const dd = peak - cumWalk;
    if (dd > maxDd) maxDd = dd;
  }

  return {
    count: rows.length,
    wins,
    losses,
    scratched,
    hitRate,
    cumulativeR: cumR,
    avgWinR,
    avgLossR,
    profitFactor: Math.min(999, profitFactor),
    expectancyR,
    sharpePerTrade,
    sortinoPerTrade,
    maxDrawdownR: maxDd,
  };
}

/** Breakdown of trades by exit reason. Ordered by count descending. */
export function reasonBreakdown(rows: TradeLedgerRow[]): ReasonBucket[] {
  const map = new Map<string, { count: number; cumR: number }>();
  for (const r of rows) {
    const e = map.get(r.exit_reason) ?? { count: 0, cumR: 0 };
    e.count += 1;
    e.cumR += r.realized_r;
    map.set(r.exit_reason, e);
  }
  const out: ReasonBucket[] = [];
  for (const [reason, v] of map) {
    out.push({
      reason,
      count: v.count,
      cumulativeR: v.cumR,
      avgR: v.count > 0 ? v.cumR / v.count : 0,
    });
  }
  out.sort((a, b) => b.count - a.count);
  return out;
}

/** Per-instrument cumulative R + hit rate. Sorted by cumulative R desc. */
export function instrumentBreakdown(rows: TradeLedgerRow[]): InstrumentPerf[] {
  const map = new Map<
    string,
    { count: number; wins: number; losses: number; cumR: number }
  >();
  for (const r of rows) {
    const e = map.get(r.instrument) ?? { count: 0, wins: 0, losses: 0, cumR: 0 };
    e.count += 1;
    e.cumR += r.realized_r;
    if (r.realized_r > 0) e.wins += 1;
    else if (r.realized_r < 0) e.losses += 1;
    map.set(r.instrument, e);
  }
  const out: InstrumentPerf[] = [];
  for (const [instrument, v] of map) {
    const decided = v.wins + v.losses;
    out.push({
      instrument,
      count: v.count,
      cumulativeR: v.cumR,
      hitRate: decided > 0 ? v.wins / decided : 0,
    });
  }
  out.sort((a, b) => b.cumulativeR - a.cumulativeR);
  return out;
}

/** Format a fractional-R value as a ±N.NN% string. */
export function fmtPctR(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${(v * 100).toFixed(digits)}%`;
}

/** Format a number with sign control. */
export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

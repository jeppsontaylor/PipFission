// Pure helpers for the price-chart label overlay. Kept out of the
// React component so the picking + partitioning logic is unit testable.
//
// `LabelRow` shape comes from the server's `/api/labels/recent` —
// each row is one ideal entry the label optimizer chose on the trailing
// 1000-bar window.

import type { LabelRow } from "./mlApi";

export interface LabelMarker {
  /** ms since epoch — the bar's open time, mapped to the price chart's t axis. */
  t: number;
  /** Mid price at entry; we render the marker at this y. Caller supplies. */
  price: number;
  /** -1 short / 0 flat / +1 long. */
  side: number;
  /** Fraction of R the optimizer realized for this entry. */
  realized_r: number;
  /** Cosmetic — `tp` | `sl` | `vert`. */
  barrier_hit: string;
  /** Which run produced this label. */
  label_run_id: string;
}

/**
 * Returns the label_run_id of the most-recent labels in `rows`. The
 * server responds in ascending ts_ms, so the last row is the freshest.
 * Used to filter to "the active run only" — older runs would clutter
 * the overlay with stale entries the optimizer no longer thinks are
 * ideal.
 */
export function pickActiveLabelRun(rows: LabelRow[]): string | null {
  if (rows.length === 0) return null;
  // Be defensive about ordering — pick the run_id of the row with the
  // largest ts_ms.
  let best = rows[0];
  for (const r of rows) {
    if (r.ts_ms > best.ts_ms) best = r;
  }
  return best.label_run_id;
}

/**
 * Filter `rows` to a single label run + a chart x-axis window, and join
 * each row to its bar's mid-price via `priceAt`. Rows whose bar isn't
 * in the price history (`priceAt` returns null) are dropped — drawing
 * a marker without a price would land it on the y-axis.
 */
export function buildLabelMarkers(
  rows: LabelRow[],
  domain: [number, number],
  priceAt: (tsMs: number) => number | null,
  activeRunId?: string | null,
): LabelMarker[] {
  const runId = activeRunId ?? pickActiveLabelRun(rows);
  if (runId === null) return [];
  const out: LabelMarker[] = [];
  for (const r of rows) {
    if (r.label_run_id !== runId) continue;
    if (r.ts_ms < domain[0] || r.ts_ms > domain[1]) continue;
    if (r.side === 0) continue; // flat labels aren't entries
    const px = priceAt(r.ts_ms);
    if (px === null || !Number.isFinite(px)) continue;
    out.push({
      t: r.ts_ms,
      price: px,
      side: r.side,
      realized_r: r.realized_r,
      barrier_hit: r.barrier_hit,
      label_run_id: r.label_run_id,
    });
  }
  return out;
}

/**
 * Splits markers into long (`side > 0`) and short (`side < 0`).
 * Convenience for recharts <Scatter> which takes one data array per
 * shape/color.
 */
export function partitionMarkers(
  markers: LabelMarker[],
): { longs: LabelMarker[]; shorts: LabelMarker[] } {
  const longs: LabelMarker[] = [];
  const shorts: LabelMarker[] = [];
  for (const m of markers) {
    if (m.side > 0) longs.push(m);
    else if (m.side < 0) shorts.push(m);
  }
  return { longs, shorts };
}

/**
 * Build a `priceAt(tsMs)` lookup from a sorted-by-time price series.
 * Uses binary search — O(log N) per query. Returns the mid of the bar
 * whose timestamp is closest to (and not after) `tsMs`. Returns null
 * if the requested time is before the series starts.
 */
export function makePriceLookup<T extends { t: number; mid: number }>(
  series: T[],
): (tsMs: number) => number | null {
  if (series.length === 0) return () => null;
  return (tsMs: number) => {
    if (tsMs < series[0].t) return null;
    let lo = 0;
    let hi = series.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >>> 1;
      if (series[mid].t <= tsMs) lo = mid;
      else hi = mid - 1;
    }
    return series[lo].mid;
  };
}

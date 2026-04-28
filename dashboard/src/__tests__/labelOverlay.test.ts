import { describe, expect, it } from "vitest";

import {
  buildLabelMarkers,
  makePriceLookup,
  partitionMarkers,
  pickActiveLabelRun,
} from "../labelOverlay";
import type { LabelRow } from "../mlApi";

function lbl(overrides: Partial<LabelRow>): LabelRow {
  return {
    instrument: "EUR_USD",
    ts_ms: 1_700_000_000_000,
    t1_ms: 1_700_000_010_000,
    side: 1,
    meta_y: 1,
    realized_r: 0.001,
    barrier_hit: "tp",
    oracle_score: 0.8,
    label_run_id: "run-1",
    ...overrides,
  };
}

describe("pickActiveLabelRun", () => {
  it("returns null on empty input", () => {
    expect(pickActiveLabelRun([])).toBeNull();
  });

  it("picks the run_id of the latest ts_ms even when ordering is mixed", () => {
    const rows = [
      lbl({ ts_ms: 100, label_run_id: "old" }),
      lbl({ ts_ms: 300, label_run_id: "new" }),
      lbl({ ts_ms: 200, label_run_id: "mid" }),
    ];
    expect(pickActiveLabelRun(rows)).toBe("new");
  });
});

describe("buildLabelMarkers", () => {
  const domain: [number, number] = [100, 300];
  const priceAt = (t: number): number | null => 1.1 + (t - 100) * 0.0001;

  it("returns empty when no rows", () => {
    expect(buildLabelMarkers([], domain, priceAt)).toEqual([]);
  });

  it("filters to the active run, drops out-of-window + flat rows, joins price", () => {
    const rows: LabelRow[] = [
      lbl({ ts_ms: 150, side: 1, label_run_id: "run-A" }),
      lbl({ ts_ms: 200, side: -1, label_run_id: "run-A" }),
      lbl({ ts_ms: 175, side: 0, label_run_id: "run-A" }), // flat → drop
      lbl({ ts_ms: 50, side: 1, label_run_id: "run-A" }), // out of window → drop
      lbl({ ts_ms: 250, side: 1, label_run_id: "run-OLD" }), // stale run → drop (run-A is freshest)
    ];
    // run-A has the freshest ts_ms (200), so pickActiveLabelRun picks "run-A".
    // Wait — run-OLD's row ts_ms is 250 which is > 200. To make this clear,
    // pin the active run explicitly:
    const markers = buildLabelMarkers(rows, domain, priceAt, "run-A");
    expect(markers.map((m) => m.t)).toEqual([150, 200]);
    expect(markers[0].side).toBe(1);
    expect(markers[1].side).toBe(-1);
    expect(markers[0].price).toBeCloseTo(1.1 + 50 * 0.0001, 5);
  });

  it("drops markers whose price lookup returns null or non-finite", () => {
    const sparsePriceAt = (t: number): number | null => (t === 200 ? null : 1.1);
    const rows = [lbl({ ts_ms: 150, side: 1 }), lbl({ ts_ms: 200, side: -1 })];
    const markers = buildLabelMarkers(rows, domain, sparsePriceAt);
    expect(markers.map((m) => m.t)).toEqual([150]);
  });

  it("uses the freshest ts_ms to pick the active run when one isn't passed", () => {
    const rows: LabelRow[] = [
      lbl({ ts_ms: 100, side: 1, label_run_id: "old" }),
      lbl({ ts_ms: 200, side: -1, label_run_id: "new" }),
      lbl({ ts_ms: 150, side: 1, label_run_id: "old" }),
    ];
    const markers = buildLabelMarkers(rows, domain, priceAt);
    expect(markers.length).toBe(1);
    expect(markers[0].label_run_id).toBe("new");
  });
});

describe("partitionMarkers", () => {
  it("splits long vs short", () => {
    const markers = [
      { t: 1, price: 1, side: 1, realized_r: 0, barrier_hit: "tp", label_run_id: "x" },
      { t: 2, price: 1, side: -1, realized_r: 0, barrier_hit: "sl", label_run_id: "x" },
      { t: 3, price: 1, side: 1, realized_r: 0, barrier_hit: "vert", label_run_id: "x" },
    ];
    const { longs, shorts } = partitionMarkers(markers);
    expect(longs.map((m) => m.t)).toEqual([1, 3]);
    expect(shorts.map((m) => m.t)).toEqual([2]);
  });
});

describe("makePriceLookup", () => {
  it("returns null for the empty series", () => {
    expect(makePriceLookup([])(123)).toBeNull();
  });

  it("returns null when ts is before the series starts", () => {
    const lookup = makePriceLookup([{ t: 100, mid: 1.0 }]);
    expect(lookup(50)).toBeNull();
  });

  it("returns the most recent bar's mid (≤ tsMs)", () => {
    const series = [
      { t: 100, mid: 1.0 },
      { t: 200, mid: 1.1 },
      { t: 300, mid: 1.2 },
    ];
    const lookup = makePriceLookup(series);
    expect(lookup(100)).toBe(1.0);
    expect(lookup(150)).toBe(1.0);
    expect(lookup(200)).toBe(1.1);
    expect(lookup(250)).toBe(1.1);
    expect(lookup(300)).toBe(1.2);
    expect(lookup(9999)).toBe(1.2);
  });

  it("works on a single-element series", () => {
    const lookup = makePriceLookup([{ t: 100, mid: 1.0 }]);
    expect(lookup(99)).toBeNull();
    expect(lookup(100)).toBe(1.0);
    expect(lookup(101)).toBe(1.0);
  });
});

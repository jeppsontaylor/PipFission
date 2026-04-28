import { describe, expect, it } from "vitest";

import {
  computePerfStats,
  fmtPctR,
  instrumentBreakdown,
  reasonBreakdown,
} from "../tradePerformance";
import type { TradeLedgerRow } from "../mlApi";

function row(overrides: Partial<TradeLedgerRow>): TradeLedgerRow {
  return {
    run_id: "r1",
    instrument: "EUR_USD",
    ts_in_ms: 1_000,
    ts_out_ms: 2_000,
    side: 1,
    qty: 1,
    entry_px: 1.1,
    exit_px: 1.1,
    fee: 0,
    slip: 0,
    realized_r: 0,
    exit_reason: "take_profit",
    model_id: null,
    params_id: null,
    entry_p_long: null,
    entry_p_short: null,
    entry_calibrated: null,
    entry_spread_bp: null,
    entry_atr_14: null,
    exit_p_long: null,
    exit_p_short: null,
    decision_chain: null,
    snapshot_path: null,
    ...overrides,
  };
}

describe("computePerfStats", () => {
  it("zero state on empty input", () => {
    const s = computePerfStats([]);
    expect(s).toMatchObject({
      count: 0,
      wins: 0,
      losses: 0,
      hitRate: 0,
      cumulativeR: 0,
      profitFactor: 0,
      expectancyR: 0,
      sharpePerTrade: 0,
      sortinoPerTrade: 0,
      maxDrawdownR: 0,
    });
  });

  it("aggregates wins/losses/scratched", () => {
    const rows = [
      row({ realized_r: 0.002 }),
      row({ realized_r: -0.001 }),
      row({ realized_r: 0 }),
    ];
    const s = computePerfStats(rows);
    expect(s.count).toBe(3);
    expect(s.wins).toBe(1);
    expect(s.losses).toBe(1);
    expect(s.scratched).toBe(1);
    expect(s.hitRate).toBeCloseTo(0.5);
  });

  it("computes profit factor and expectancy", () => {
    const rows = [
      row({ realized_r: 0.003 }),
      row({ realized_r: 0.001 }),
      row({ realized_r: -0.002 }),
    ];
    const s = computePerfStats(rows);
    expect(s.profitFactor).toBeCloseTo(0.004 / 0.002);
    // hitRate 2/3, avgWin 0.002, avgLoss -0.002
    expect(s.expectancyR).toBeCloseTo((2 / 3) * 0.002 + (1 / 3) * -0.002, 5);
  });

  it("computes max drawdown on the cum-R series", () => {
    // ts_out_ms forces chronological ordering; try a sequence where
    // we go +5 → -10 → +3 (peak 5, trough -5, recovery -2).
    const rows = [
      row({ ts_out_ms: 100, realized_r: 0.005 }),
      row({ ts_out_ms: 200, realized_r: -0.010 }),
      row({ ts_out_ms: 300, realized_r: 0.003 }),
    ];
    const s = computePerfStats(rows);
    // Peak after step 1 = 0.005; trough after step 2 = -0.005 → DD = 0.010.
    expect(s.maxDrawdownR).toBeCloseTo(0.010, 5);
  });

  it("sortino uses downside-only volatility (positive-mean case: |Sortino| > |Sharpe|)", () => {
    // Mean must be positive for the |Sortino| > |Sharpe| relationship to
    // hold cleanly. With one downside obs and several upside obs, Sharpe
    // penalises the upside vol while Sortino doesn't.
    const rows = [
      row({ realized_r: 0.003 }),
      row({ realized_r: 0.004 }),
      row({ realized_r: 0.005 }),
      row({ realized_r: -0.001 }),
    ];
    const s = computePerfStats(rows);
    expect(s.sharpePerTrade).toBeGreaterThan(0);
    expect(s.sortinoPerTrade).toBeGreaterThan(0);
    expect(s.sortinoPerTrade).toBeGreaterThan(s.sharpePerTrade);
  });

  it("with no downside observations, sortino is 0 (downside stdev is 0)", () => {
    const rows = [
      row({ realized_r: 0.001 }),
      row({ realized_r: 0.002 }),
    ];
    const s = computePerfStats(rows);
    expect(s.sortinoPerTrade).toBe(0);
  });

  it("clamps profit factor at 999", () => {
    const rows = [row({ realized_r: 0.001 }), row({ realized_r: 0.002 })];
    expect(computePerfStats(rows).profitFactor).toBe(999);
  });
});

describe("reasonBreakdown", () => {
  it("groups by exit_reason and orders by count desc", () => {
    const rows = [
      row({ exit_reason: "take_profit", realized_r: 0.002 }),
      row({ exit_reason: "stop_loss", realized_r: -0.001 }),
      row({ exit_reason: "take_profit", realized_r: 0.001 }),
      row({ exit_reason: "max_hold", realized_r: 0.0005 }),
    ];
    const out = reasonBreakdown(rows);
    expect(out.map((b) => b.reason)).toEqual(["take_profit", "stop_loss", "max_hold"]);
    expect(out[0].count).toBe(2);
    expect(out[0].cumulativeR).toBeCloseTo(0.003);
    expect(out[0].avgR).toBeCloseTo(0.0015);
  });
});

describe("instrumentBreakdown", () => {
  it("groups by instrument, sorts by cum R desc", () => {
    const rows = [
      row({ instrument: "EUR_USD", realized_r: 0.001 }),
      row({ instrument: "USD_JPY", realized_r: -0.002 }),
      row({ instrument: "EUR_USD", realized_r: 0.002 }),
      row({ instrument: "USD_JPY", realized_r: 0.001 }),
    ];
    const out = instrumentBreakdown(rows);
    expect(out[0].instrument).toBe("EUR_USD");
    expect(out[0].cumulativeR).toBeCloseTo(0.003);
    expect(out[0].count).toBe(2);
    expect(out[0].hitRate).toBeCloseTo(1.0);
    expect(out[1].instrument).toBe("USD_JPY");
    expect(out[1].cumulativeR).toBeCloseTo(-0.001);
  });
});

describe("fmtPctR", () => {
  it("adds + sign for non-negative", () => {
    expect(fmtPctR(0.012)).toBe("+1.20%");
    expect(fmtPctR(0)).toBe("+0.00%");
  });
  it("preserves - sign for negative", () => {
    expect(fmtPctR(-0.005)).toBe("-0.50%");
  });
  it("handles non-finite", () => {
    expect(fmtPctR(Number.NaN)).toBe("—");
    expect(fmtPctR(Number.POSITIVE_INFINITY)).toBe("—");
  });
});

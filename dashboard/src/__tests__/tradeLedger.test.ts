import { describe, expect, it } from "vitest";

import {
  cumulativeRCurve,
  exitReasonLabel,
  sideLabel,
  sideTone,
  summarizeLedger,
} from "../tradeLedger";
import type { TradeLedgerRow } from "../mlApi";

function row(overrides: Partial<TradeLedgerRow>): TradeLedgerRow {
  return {
    run_id: "run-1",
    instrument: "EUR_USD",
    ts_in_ms: 1_700_000_000_000,
    ts_out_ms: 1_700_000_010_000,
    side: 1,
    qty: 1,
    entry_px: 1.1,
    exit_px: 1.1,
    fee: 0,
    slip: 0,
    realized_r: 0,
    exit_reason: "take_profit",
    // Phase D enrichment columns — null in fixtures.
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

describe("summarizeLedger", () => {
  it("returns zero-state on empty", () => {
    const s = summarizeLedger([]);
    expect(s.count).toBe(0);
    expect(s.wins).toBe(0);
    expect(s.losses).toBe(0);
    expect(s.hitRate).toBe(0);
    expect(s.cumulativeR).toBe(0);
    expect(s.profitFactor).toBe(0);
    expect(s.bestTradeR).toBeNull();
    expect(s.worstTradeR).toBeNull();
  });

  it("aggregates wins/losses/profit_factor correctly", () => {
    const rows = [
      row({ realized_r: 0.002, exit_reason: "take_profit" }),
      row({ realized_r: -0.001, exit_reason: "stop_loss" }),
      row({ realized_r: 0.003, exit_reason: "take_profit" }),
      row({ realized_r: -0.002, exit_reason: "max_hold" }),
      row({ realized_r: 0, exit_reason: "manual" }), // scratched
    ];
    const s = summarizeLedger(rows);
    expect(s.count).toBe(5);
    expect(s.wins).toBe(2);
    expect(s.losses).toBe(2);
    expect(s.hitRate).toBeCloseTo(0.5, 5);
    expect(s.cumulativeR).toBeCloseTo(0.002 - 0.001 + 0.003 - 0.002 + 0, 5);
    // posR=0.005, negR=0.003 → 5/3 ≈ 1.6667
    expect(s.profitFactor).toBeCloseTo(0.005 / 0.003, 4);
    expect(s.bestTradeR).toBeCloseTo(0.003);
    expect(s.worstTradeR).toBeCloseTo(-0.002);
    expect(s.reasons.take_profit).toBe(2);
    expect(s.reasons.stop_loss).toBe(1);
    expect(s.reasons.max_hold).toBe(1);
    expect(s.reasons.manual).toBe(1);
  });

  it("returns 999 for profit factor when no losses but at least one win", () => {
    const rows = [row({ realized_r: 0.001 }), row({ realized_r: 0.002 })];
    expect(summarizeLedger(rows).profitFactor).toBe(999);
  });

  it("returns 0 for profit factor when no wins and no losses", () => {
    const rows = [row({ realized_r: 0 }), row({ realized_r: 0 })];
    expect(summarizeLedger(rows).profitFactor).toBe(0);
  });

  it("computes net P/L with side sign and subtracts fee+slip", () => {
    const rows = [
      // long: qty=2 entry=1.0 exit=1.05 → +0.10, minus 0.01 fee, 0.005 slip = 0.085
      row({ side: 1, qty: 2, entry_px: 1.0, exit_px: 1.05, fee: 0.01, slip: 0.005 }),
      // short: qty=1 entry=1.0 exit=0.99 → +0.01 (sideMul=-1 * (0.99-1.0)=0.01)
      row({ side: -1, qty: 1, entry_px: 1.0, exit_px: 0.99, fee: 0, slip: 0 }),
    ];
    const s = summarizeLedger(rows);
    expect(s.netPL).toBeCloseTo(0.085 + 0.01, 5);
  });
});

describe("cumulativeRCurve", () => {
  it("yields a monotonic time index and running sum", () => {
    const rows = [
      row({ ts_out_ms: 30, realized_r: 0.003 }),
      row({ ts_out_ms: 10, realized_r: 0.001 }),
      row({ ts_out_ms: 20, realized_r: -0.002 }),
    ];
    const curve = cumulativeRCurve(rows);
    expect(curve.map((p) => p.t)).toEqual([10, 20, 30]);
    expect(curve.map((p) => p.cum_r.toFixed(3))).toEqual(["0.001", "-0.001", "0.002"]);
  });

  it("returns empty for no rows", () => {
    expect(cumulativeRCurve([])).toEqual([]);
  });
});

describe("sideLabel / sideTone", () => {
  it("maps signs to labels", () => {
    expect(sideLabel(1)).toBe("Long");
    expect(sideLabel(-1)).toBe("Short");
    expect(sideLabel(0)).toBe("Flat");
  });
  it("returns distinct tone classes", () => {
    expect(sideTone(1)).not.toBe(sideTone(-1));
    expect(sideTone(0)).not.toBe(sideTone(1));
  });
});

describe("exitReasonLabel", () => {
  it("maps known reasons; passes through unknown", () => {
    expect(exitReasonLabel("take_profit")).toBe("Take profit");
    expect(exitReasonLabel("stop_loss")).toBe("Stop loss");
    expect(exitReasonLabel("custom_reason_xyz")).toBe("custom_reason_xyz");
  });
});

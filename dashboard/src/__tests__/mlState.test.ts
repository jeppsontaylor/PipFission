import { describe, expect, it } from "vitest";

import {
  ML_HISTORY_LIMIT,
  actionLabel,
  applyChampionChanged,
  applyChampionLoadFailed,
  applyChampionSignal,
  applyTraderDecision,
  initialMLState,
  isOrderingAction,
  reasonLabel,
} from "../mlState";
import type {
  ChampionChanged,
  ChampionLoadFailed,
  ChampionSignal,
  TraderDecision,
} from "../types";

const SIGNAL_BASE: Omit<ChampionSignal, "time"> = {
  instrument: "EUR_USD",
  p_long: 0.7,
  p_short: 0.3,
  p_take: 0.7,
  calibrated: 0.7,
  model_id: "side_lgbm_xyz",
  kind: "onnx",
};

function mkSignal(time: string, overrides: Partial<ChampionSignal> = {}): ChampionSignal {
  return { ...SIGNAL_BASE, time, ...overrides } as ChampionSignal;
}

const DECISION_BASE: Omit<TraderDecision, "time"> = {
  instrument: "EUR_USD",
  bar_idx: 0,
  action: "skip",
  reason: "below_threshold",
  price: 1.1,
  realized_r: null,
  params_id: "params_x",
  model_id: "side_lgbm_xyz",
};

function mkDecision(time: string, overrides: Partial<TraderDecision> = {}): TraderDecision {
  return { ...DECISION_BASE, time, ...overrides } as TraderDecision;
}

describe("applyChampionSignal", () => {
  it("records the latest signal and appends a history point", () => {
    let s = initialMLState;
    s = applyChampionSignal(s, mkSignal("2026-04-26T12:00:00Z"));
    expect(s.championSignals.EUR_USD?.p_long).toBe(0.7);
    expect(s.championHistory.EUR_USD?.length).toBe(1);
  });

  it("ignores signals with unparseable timestamps (no NaN points)", () => {
    let s = initialMLState;
    s = applyChampionSignal(s, mkSignal("not-a-date"));
    expect(s.championHistory.EUR_USD).toBeUndefined();
  });

  it("caps the rolling history at ML_HISTORY_LIMIT", () => {
    let s = initialMLState;
    const overflow = ML_HISTORY_LIMIT + 25;
    for (let i = 0; i < overflow; i += 1) {
      const ts = new Date(1_700_000_000_000 + i * 10_000).toISOString();
      s = applyChampionSignal(s, mkSignal(ts));
    }
    expect(s.championHistory.EUR_USD?.length).toBe(ML_HISTORY_LIMIT);
    // Newest point survived; oldest got dropped.
    const last = s.championHistory.EUR_USD![ML_HISTORY_LIMIT - 1];
    expect(last.t).toBe(1_700_000_000_000 + (overflow - 1) * 10_000);
  });

  it("buckets per-instrument independently", () => {
    let s = initialMLState;
    s = applyChampionSignal(s, mkSignal("2026-04-26T12:00:00Z", { instrument: "EUR_USD" }));
    s = applyChampionSignal(s, mkSignal("2026-04-26T12:00:00Z", { instrument: "USD_JPY" }));
    expect(Object.keys(s.championHistory).sort()).toEqual(["EUR_USD", "USD_JPY"]);
  });
});

describe("applyChampionChanged", () => {
  it("captures the new model_id and clears prior error", () => {
    let s = applyChampionLoadFailed(initialMLState, { reason: "bad sha" } as ChampionLoadFailed);
    expect(s.championStatus.last_error).toBe("bad sha");
    s = applyChampionChanged(s, {
      model_id: "champ-1",
      kind: "onnx",
      n_features: 24,
    } as ChampionChanged);
    expect(s.championStatus.model_id).toBe("champ-1");
    expect(s.championStatus.last_error).toBeNull();
    expect(s.championStatus.last_changed_at).not.toBeNull();
  });
});

describe("applyTraderDecision", () => {
  it("records latest and appends to history", () => {
    let s = initialMLState;
    s = applyTraderDecision(s, mkDecision("2026-04-26T12:00:00Z", { action: "open_long" }));
    expect(s.traderDecisions.EUR_USD?.action).toBe("open_long");
    expect(s.traderDecisionHistory.EUR_USD?.length).toBe(1);
  });

  it("preserves chronological order in history", () => {
    let s = initialMLState;
    const ts = (i: number) => new Date(1_700_000_000_000 + i * 10_000).toISOString();
    s = applyTraderDecision(s, mkDecision(ts(0), { action: "open_long" }));
    s = applyTraderDecision(s, mkDecision(ts(1), { action: "skip", reason: "min_hold" }));
    s = applyTraderDecision(
      s,
      mkDecision(ts(2), { action: "close", reason: "take_profit", realized_r: 0.002 }),
    );
    const hist = s.traderDecisionHistory.EUR_USD!;
    expect(hist.map((p) => p.action)).toEqual(["open_long", "skip", "close"]);
    expect(hist[2].realized_r).toBe(0.002);
  });
});

describe("label helpers", () => {
  it("actionLabel maps known actions", () => {
    expect(actionLabel("open_long")).toBe("Open Long");
    expect(actionLabel("close")).toBe("Close");
    expect(actionLabel("custom_action")).toBe("custom_action");
  });

  it("reasonLabel maps known reasons", () => {
    expect(reasonLabel("stop_loss")).toBe("Stop loss");
    expect(reasonLabel("custom_reason")).toBe("custom_reason");
  });

  it("isOrderingAction recognises position-changing actions", () => {
    expect(isOrderingAction("open_long")).toBe(true);
    expect(isOrderingAction("open_short")).toBe(true);
    expect(isOrderingAction("close")).toBe(true);
    expect(isOrderingAction("skip")).toBe(false);
    expect(isOrderingAction("anything_else")).toBe(false);
  });
});

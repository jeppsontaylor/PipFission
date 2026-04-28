import { describe, expect, it } from "vitest";

import {
  pickSelectedInstrument,
  pickTabTone,
} from "../mlPanelTabs";
import type { ChampionSignal } from "../types";

function sig(overrides: Partial<ChampionSignal>): ChampionSignal {
  return {
    type: "champion_signal" as never,
    instrument: "EUR_USD",
    time: "2026-04-26T12:00:00Z",
    p_long: 0.5,
    p_short: 0.5,
    p_take: 0.5,
    calibrated: 0.5,
    model_id: "m1",
    kind: "onnx",
    ...overrides,
  } as ChampionSignal;
}

describe("pickTabTone", () => {
  it("idle when no signal", () => {
    expect(pickTabTone(undefined)).toBe("idle");
  });
  it("long when p_long crosses threshold and dominates", () => {
    expect(pickTabTone(sig({ p_long: 0.7, p_short: 0.3 }))).toBe("long");
  });
  it("short when p_short crosses threshold and dominates", () => {
    expect(pickTabTone(sig({ p_long: 0.2, p_short: 0.8 }))).toBe("short");
  });
  it("flat when neither side crosses threshold", () => {
    expect(pickTabTone(sig({ p_long: 0.5, p_short: 0.5 }))).toBe("flat");
  });
  it("flat when long crosses threshold but short is too close", () => {
    // 0.56 vs 0.55 — within 0.02 → not decisive → flat
    expect(pickTabTone(sig({ p_long: 0.56, p_short: 0.55 }))).toBe("flat");
  });
  it("custom threshold respected", () => {
    expect(pickTabTone(sig({ p_long: 0.4, p_short: 0.3 }), 0.35)).toBe("long");
  });
});

describe("pickSelectedInstrument", () => {
  it("empty list returns empty string", () => {
    expect(pickSelectedInstrument([], "EUR_USD")).toBe("");
  });
  it("keeps the current pick when still present", () => {
    expect(pickSelectedInstrument(["EUR_USD", "USD_JPY"], "USD_JPY")).toBe("USD_JPY");
  });
  it("falls back to first when current is gone", () => {
    expect(pickSelectedInstrument(["EUR_USD", "USD_JPY"], "BTC_USD")).toBe("EUR_USD");
  });
  it("falls back to first on empty current", () => {
    expect(pickSelectedInstrument(["EUR_USD"], "")).toBe("EUR_USD");
  });
});

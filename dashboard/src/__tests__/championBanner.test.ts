import { describe, expect, it } from "vitest";

import {
  DEFAULT_SWAP_BANNER_MS,
  pickBannerState,
  shortModelId,
} from "../championBanner";
import type { ChampionStatus } from "../mlState";

function status(overrides: Partial<ChampionStatus>): ChampionStatus {
  return {
    model_id: "uninitialised",
    kind: "uninitialised",
    last_changed_at: null,
    last_error: null,
    ...overrides,
  };
}

describe("pickBannerState", () => {
  it("idle when nothing has happened", () => {
    expect(pickBannerState(status({}), 0).kind).toBe("idle");
  });

  it("error wins over a fresh swap (so a load-failed surfaces even if a swap landed earlier)", () => {
    const s = status({
      model_id: "side_lgbm_xyz",
      kind: "onnx",
      last_changed_at: 100,
      last_error: "bad sha",
    });
    const out = pickBannerState(s, 110);
    expect(out.kind).toBe("error");
    expect(out.errorReason).toBe("bad sha");
  });

  it("swapped within the window", () => {
    const s = status({
      model_id: "side_lgbm_xyz",
      kind: "onnx",
      last_changed_at: 1000,
    });
    const out = pickBannerState(s, 1500);
    expect(out.kind).toBe("swapped");
    expect(out.modelId).toBe("side_lgbm_xyz");
    expect(out.ageMs).toBe(500);
  });

  it("idle when swap is older than the window", () => {
    const s = status({
      model_id: "side_lgbm_xyz",
      kind: "onnx",
      last_changed_at: 1_000_000,
    });
    const out = pickBannerState(s, 1_000_000 + DEFAULT_SWAP_BANNER_MS + 1);
    expect(out.kind).toBe("idle");
  });

  it("idle when last_changed_at is in the future (clock skew)", () => {
    const s = status({
      model_id: "side_lgbm_xyz",
      kind: "onnx",
      last_changed_at: 5_000,
    });
    expect(pickBannerState(s, 1_000).kind).toBe("idle");
  });

  it("never shows swapped for the uninitialised sentinel", () => {
    const s = status({
      model_id: "uninitialised",
      last_changed_at: 100,
    });
    expect(pickBannerState(s, 110).kind).toBe("idle");
  });

  it("custom window respected", () => {
    const s = status({
      model_id: "m",
      kind: "onnx",
      last_changed_at: 0,
    });
    expect(pickBannerState(s, 100, 50).kind).toBe("idle");
    expect(pickBannerState(s, 100, 200).kind).toBe("swapped");
  });
});

describe("shortModelId", () => {
  it("returns short ids unchanged", () => {
    expect(shortModelId("abc")).toBe("abc");
  });
  it("truncates with ellipsis", () => {
    expect(shortModelId("0123456789abcdef0123456789")).toBe("01234567…23456789");
  });
  it("respects custom head/tail lengths", () => {
    expect(shortModelId("abcdefghij", 2, 2)).toBe("ab…ij");
  });
});

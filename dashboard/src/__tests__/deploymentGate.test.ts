import { describe, expect, it } from "vitest";

import {
  evaluateFloors,
  parseThresholds,
  splitBlockedReasons,
} from "../deploymentGate";
import type { ModelDeploymentGateRow } from "../mlApi";

function row(overrides: Partial<ModelDeploymentGateRow>): ModelDeploymentGateRow {
  return {
    model_id: "m",
    instrument: "EUR_USD",
    ts_ms: 1_700_000_000_000,
    oos_auc: 0.6,
    oos_log_loss: 0.5,
    oos_brier: 0.2,
    oos_balanced_acc: 0.55,
    fine_tune_sortino: 0.5,
    fine_tune_max_dd_bp: 500,
    passed_gate: true,
    blocked_reasons: "",
    gate_thresholds_json: JSON.stringify({
      min_oos_auc: 0.55,
      max_oos_log_loss: 0.7,
      min_oos_balanced_acc: 0.52,
      min_fine_tune_sortino: 0.3,
      max_fine_tune_dd_bp: 1500,
      require_lockbox_pass: true,
    }),
    ...overrides,
  };
}

describe("parseThresholds", () => {
  it("returns defaults for malformed input", () => {
    const t = parseThresholds("not json");
    expect(t.min_oos_auc).toBe(0.55);
    expect(t.require_lockbox_pass).toBe(true);
  });
  it("parses valid JSON", () => {
    const t = parseThresholds(JSON.stringify({ min_oos_auc: 0.7 }));
    expect(t.min_oos_auc).toBe(0.7);
    // Other fields fall back to defaults.
    expect(t.max_oos_log_loss).toBe(0.7);
  });
});

describe("evaluateFloors", () => {
  it("passes everything for a comfortably good candidate", () => {
    const floors = evaluateFloors(row({}));
    expect(floors.length).toBe(5);
    expect(floors.every((f) => f.passed)).toBe(true);
  });

  it("detects the failing floor on an underperformer", () => {
    const floors = evaluateFloors(row({ oos_auc: 0.40 }));
    const auc = floors.find((f) => f.label === "OOS AUC")!;
    expect(auc.passed).toBe(false);
    expect(auc.fractionOver).toBeLessThan(0);
  });

  it("max-style thresholds (log loss, drawdown) pass when actual is below", () => {
    const floors = evaluateFloors(row({ oos_log_loss: 0.4, fine_tune_max_dd_bp: 200 }));
    const ll = floors.find((f) => f.label === "OOS log loss")!;
    const dd = floors.find((f) => f.label === "Fine-tune max DD (bp)")!;
    expect(ll.passed).toBe(true);
    expect(dd.passed).toBe(true);
    expect(ll.direction).toBe("le");
    expect(dd.direction).toBe("le");
  });

  it("max-style thresholds fail when actual exceeds floor", () => {
    const floors = evaluateFloors(row({ fine_tune_max_dd_bp: 2000 }));
    const dd = floors.find((f) => f.label === "Fine-tune max DD (bp)")!;
    expect(dd.passed).toBe(false);
  });
});

describe("splitBlockedReasons", () => {
  it("splits semicolon-delimited list", () => {
    expect(splitBlockedReasons("a; b; c")).toEqual(["a", "b", "c"]);
  });
  it("filters empty entries", () => {
    expect(splitBlockedReasons("a;;b; ")).toEqual(["a", "b"]);
  });
  it("returns empty array on empty input", () => {
    expect(splitBlockedReasons("")).toEqual([]);
  });
});

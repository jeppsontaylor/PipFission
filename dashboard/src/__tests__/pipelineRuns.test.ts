import { describe, expect, it } from "vitest";

import {
  autoRetrainProgress,
  commandLabel,
  formatElapsed,
  liveElapsed,
  parseArgs,
  pipelineButtonState,
  statusLabel,
  statusTone,
  summarizePipelineRuns,
} from "../pipelineRuns";
import type { PipelineRunRow } from "../mlApi";

function row(overrides: Partial<PipelineRunRow>): PipelineRunRow {
  return {
    run_id: "run-1",
    command: "label",
    instrument: "EUR_USD",
    args_json: "{}",
    ts_started_ms: 1_700_000_000_000,
    ts_finished_ms: 1_700_000_001_000,
    status: "success",
    elapsed_ms: 1000,
    error_msg: null,
    ...overrides,
  };
}

describe("summarizePipelineRuns", () => {
  it("returns zeroes for empty input", () => {
    const s = summarizePipelineRuns([]);
    expect(s).toEqual({
      total: 0,
      success: 0,
      failed: 0,
      running: 0,
      inflight: null,
      lastSuccessByCommand: {},
    });
  });

  it("counts statuses, picks newest inflight, tracks per-command success ts", () => {
    const rows = [
      row({ run_id: "1", status: "success", command: "label", ts_started_ms: 100 }),
      row({ run_id: "2", status: "success", command: "label", ts_started_ms: 200 }),
      row({ run_id: "3", status: "failed", command: "train.side", ts_started_ms: 300 }),
      row({ run_id: "4", status: "running", command: "finetune", ts_started_ms: 400, ts_finished_ms: null, elapsed_ms: null }),
      row({ run_id: "5", status: "running", command: "lockbox", ts_started_ms: 500, ts_finished_ms: null, elapsed_ms: null }),
    ];
    const s = summarizePipelineRuns(rows);
    expect(s.total).toBe(5);
    expect(s.success).toBe(2);
    expect(s.failed).toBe(1);
    expect(s.running).toBe(2);
    expect(s.inflight?.run_id).toBe("5"); // most recent
    expect(s.lastSuccessByCommand).toEqual({ label: 200 });
  });
});

describe("formatElapsed", () => {
  it("handles null", () => {
    expect(formatElapsed(null)).toBe("—");
  });
  it("formats sub-second in ms", () => {
    expect(formatElapsed(43)).toBe("43ms");
  });
  it("formats sub-minute in seconds with one decimal", () => {
    expect(formatElapsed(1400)).toBe("1.4s");
    expect(formatElapsed(59_000)).toBe("59.0s");
  });
  it("formats minutes and seconds", () => {
    expect(formatElapsed(60_000)).toBe("1m 00s");
    expect(formatElapsed(723_000)).toBe("12m 03s");
  });
});

describe("liveElapsed", () => {
  it("returns elapsed_ms when set", () => {
    expect(liveElapsed(row({ elapsed_ms: 1234 }))).toBe(1234);
  });
  it("computes against `now` for in-flight rows", () => {
    const r = row({ status: "running", ts_started_ms: 100, ts_finished_ms: null, elapsed_ms: null });
    expect(liveElapsed(r, 5_000)).toBe(4_900);
  });
  it("clamps negative to 0", () => {
    const r = row({ status: "running", ts_started_ms: 1000, ts_finished_ms: null, elapsed_ms: null });
    expect(liveElapsed(r, 500)).toBe(0);
  });
});

describe("statusLabel / statusTone", () => {
  it("uses three-letter tags", () => {
    expect(statusLabel("success")).toBe("OK");
    expect(statusLabel("running")).toBe("RUN");
    expect(statusLabel("failed")).toBe("FAIL");
    expect(statusLabel("custom")).toBe("CUSTOM");
  });
  it("returns distinct tone strings", () => {
    const tones = new Set([
      statusTone("success"),
      statusTone("running"),
      statusTone("failed"),
      statusTone("other"),
    ]);
    expect(tones.size).toBe(4);
  });
});

describe("commandLabel", () => {
  it("maps known commands", () => {
    expect(commandLabel("label")).toBe("Labels");
    expect(commandLabel("train.side")).toBe("Train side");
    expect(commandLabel("finetune")).toBe("Fine-tune");
    expect(commandLabel("lockbox")).toBe("Lockbox");
    expect(commandLabel("export.champion")).toBe("Export ONNX");
  });
  it("falls through for unknown commands", () => {
    expect(commandLabel("custom_thing")).toBe("custom_thing");
  });
});

describe("pipelineButtonState", () => {
  const base = {
    enabled: true,
    inflightInstrument: null as string | null,
    selectedInstrument: "EUR_USD",
    submitting: false,
  };

  it("submitting takes priority over everything", () => {
    const s = pipelineButtonState({ ...base, submitting: true, enabled: false });
    expect(s.disabled).toBe(true);
    expect(s.label).toBe("Submitting…");
  });

  it("disables when server gate is off", () => {
    const s = pipelineButtonState({ ...base, enabled: false });
    expect(s.disabled).toBe(true);
    expect(s.label).toBe("Trigger disabled");
    expect(s.reason).toContain("PIPELINE_TRIGGER_ENABLED");
  });

  it("disables when something else is in flight", () => {
    const s = pipelineButtonState({ ...base, inflightInstrument: "BTC_USD" });
    expect(s.disabled).toBe(true);
    expect(s.label).toContain("BTC_USD");
  });

  it("disables when no instrument is selected", () => {
    const s = pipelineButtonState({ ...base, selectedInstrument: "" });
    expect(s.disabled).toBe(true);
    expect(s.label).toBe("Run pipeline");
  });

  it("enables when everything is ready", () => {
    const s = pipelineButtonState({ ...base, selectedInstrument: "EUR_USD" });
    expect(s.disabled).toBe(false);
    expect(s.label).toBe("Run pipeline for EUR_USD");
  });
});

describe("autoRetrainProgress", () => {
  it("ratio shows clamped denominator (≥1)", () => {
    expect(autoRetrainProgress(10, 0).ratio).toBe("10/1");
    expect(autoRetrainProgress(10, 100).ratio).toBe("10/100");
  });
  it("fraction clamps high values for bar width purposes", () => {
    const overdue = autoRetrainProgress(500, 100);
    expect(overdue.fraction).toBeLessThanOrEqual(1.5);
    expect(overdue.tone).toBe("text-rose-200");
  });
  it("amber band kicks in at 80%", () => {
    expect(autoRetrainProgress(80, 100).tone).toBe("text-amber-200");
    expect(autoRetrainProgress(79, 100).tone).toBe("text-emerald-200");
    expect(autoRetrainProgress(99, 100).tone).toBe("text-amber-200");
    expect(autoRetrainProgress(100, 100).tone).toBe("text-rose-200");
  });
  it("0 bars is healthy emerald", () => {
    expect(autoRetrainProgress(0, 100).tone).toBe("text-emerald-200");
  });
});

describe("parseArgs", () => {
  it("returns {} on malformed input", () => {
    expect(parseArgs("not json")).toEqual({});
    expect(parseArgs("")).toEqual({});
    expect(parseArgs("[]")).toEqual({});
    expect(parseArgs("null")).toEqual({});
  });
  it("returns the dict on valid json object", () => {
    expect(parseArgs(JSON.stringify({ a: 1, b: "two" }))).toEqual({ a: 1, b: "two" });
  });
});

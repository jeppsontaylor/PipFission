// Pure helpers for the pipeline-runs panel. Aggregate view, formatting,
// and status colouring — split out of the React component so it's
// straightforward to unit test.

import type { PipelineRunRow } from "./mlApi";

export interface PipelineRunsSummary {
  total: number;
  success: number;
  failed: number;
  running: number;
  /** Newest in-flight run, if any. */
  inflight: PipelineRunRow | null;
  /** ts_started_ms of the most-recent successful run, per command. */
  lastSuccessByCommand: Record<string, number>;
}

export function summarizePipelineRuns(rows: PipelineRunRow[]): PipelineRunsSummary {
  let success = 0;
  let failed = 0;
  let running = 0;
  let inflight: PipelineRunRow | null = null;
  const lastSuccessByCommand: Record<string, number> = {};
  for (const r of rows) {
    if (r.status === "success") success += 1;
    else if (r.status === "failed") failed += 1;
    else if (r.status === "running") {
      running += 1;
      // Newest in-flight wins.
      if (inflight === null || r.ts_started_ms > inflight.ts_started_ms) {
        inflight = r;
      }
    }
    if (r.status === "success") {
      const prev = lastSuccessByCommand[r.command] ?? -Infinity;
      if (r.ts_started_ms > prev) {
        lastSuccessByCommand[r.command] = r.ts_started_ms;
      }
    }
  }
  return { total: rows.length, success, failed, running, inflight, lastSuccessByCommand };
}

export function statusTone(status: string): string {
  switch (status) {
    case "success":
      return "bg-emerald-900/40 border-emerald-700/60 text-emerald-200";
    case "failed":
      return "bg-rose-900/40 border-rose-700/60 text-rose-200";
    case "running":
      return "bg-sky-900/40 border-sky-700/60 text-sky-200";
    default:
      return "bg-slate-800/60 border-slate-700 text-slate-300";
  }
}

export function statusLabel(status: string): string {
  switch (status) {
    case "running":
      return "RUN";
    case "success":
      return "OK";
    case "failed":
      return "FAIL";
    default:
      return status.toUpperCase();
  }
}

/**
 * Format an elapsed-ms value compactly. Returns "—" for null;
 * "  43ms" for sub-second; "1.4s" for sub-minute; "12m 03s" for longer.
 */
export function formatElapsed(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) return "—";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.floor((ms % 60_000) / 1000);
  return `${m}m ${s.toString().padStart(2, "0")}s`;
}

/**
 * If a run is still running, compute the "live" elapsed against `now`
 * (defaults to Date.now()). Lets the dashboard show "running for 47s".
 */
export function liveElapsed(row: PipelineRunRow, now: number = Date.now()): number {
  if (row.elapsed_ms !== null) return row.elapsed_ms;
  return Math.max(0, now - row.ts_started_ms);
}

export function commandLabel(cmd: string): string {
  switch (cmd) {
    case "label":
      return "Labels";
    case "train.side":
      return "Train side";
    case "finetune":
      return "Fine-tune";
    case "lockbox":
      return "Lockbox";
    case "export.champion":
      return "Export ONNX";
    default:
      return cmd;
  }
}

export interface AutoRetrainProgress {
  /** 0..1 fraction of threshold reached. Clamped. */
  fraction: number;
  /** "47/100" style display string. */
  ratio: string;
  /** Color tone: green (idle), amber (≥80%), rose (overdue). */
  tone: string;
}

/** Pure progress descriptor for one tracked instrument. */
export function autoRetrainProgress(
  bars: number,
  threshold: number,
): AutoRetrainProgress {
  const t = Math.max(1, threshold);
  const frac = Math.max(0, bars / t);
  const display = `${bars}/${t}`;
  let tone: string;
  if (frac >= 1) tone = "text-rose-200";
  else if (frac >= 0.8) tone = "text-amber-200";
  else tone = "text-emerald-200";
  return { fraction: Math.min(1.5, frac), ratio: display, tone };
}

export interface TriggerButtonState {
  disabled: boolean;
  /** Tooltip / aria-label explaining why the button is in this state. */
  reason: string;
  /** Visible label on the button. */
  label: string;
}

/**
 * Decide whether the "Run pipeline" button should be active. Pure,
 * easily unit-tested. Inputs:
 *   * `enabled` — whether the server has `PIPELINE_TRIGGER_ENABLED=true`.
 *   * `inflightInstrument` — non-null while a run is in flight.
 *   * `selectedInstrument` — what the user picked in the dropdown.
 *   * `submitting` — true while the POST is in flight on the client.
 */
export function pipelineButtonState(opts: {
  enabled: boolean;
  inflightInstrument: string | null;
  selectedInstrument: string;
  submitting: boolean;
}): TriggerButtonState {
  if (opts.submitting) {
    return { disabled: true, reason: "submitting…", label: "Submitting…" };
  }
  if (!opts.enabled) {
    return {
      disabled: true,
      reason: "PIPELINE_TRIGGER_ENABLED is not set on the server",
      label: "Trigger disabled",
    };
  }
  if (opts.inflightInstrument !== null) {
    return {
      disabled: true,
      reason: `pipeline already running for ${opts.inflightInstrument}`,
      label: `Running ${opts.inflightInstrument}…`,
    };
  }
  if (opts.selectedInstrument.trim() === "") {
    return {
      disabled: true,
      reason: "pick an instrument first",
      label: "Run pipeline",
    };
  }
  return {
    disabled: false,
    reason: `Run full retrain for ${opts.selectedInstrument}`,
    label: `Run pipeline for ${opts.selectedInstrument}`,
  };
}

/** Parse args_json defensively. Returns `{}` for malformed JSON. */
export function parseArgs(json: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(json);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    /* fall through */
  }
  return {};
}

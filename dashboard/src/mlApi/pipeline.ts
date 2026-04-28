// /api/pipeline/* — runs history, in-flight status, log tail, manual
// trigger, auto-retrain counters. Everything the dashboard's
// PipelineRunsCard reads from.

import { useEffect, useState } from "react";

import { BASE, fetchJson } from "./internal";

// === pipeline_runs (Python-side per-stage rows) ===========================

export interface PipelineRunRow {
  run_id: string;
  command: string;
  instrument: string | null;
  args_json: string;
  ts_started_ms: number;
  ts_finished_ms: number | null;
  /** "running" | "success" | "failed" */
  status: string;
  elapsed_ms: number | null;
  error_msg: string | null;
}

export function fetchPipelineRuns(
  limit = 50,
  signal?: AbortSignal,
): Promise<PipelineRunRow[]> {
  return fetchJson<PipelineRunRow[]>(
    `${BASE}/api/pipeline/runs?limit=${limit}`,
    signal,
  );
}

export interface PipelineRunsState {
  rows: PipelineRunRow[];
  fetchedAt: number | null;
  error: string | null;
  loading: boolean;
}

export interface PipelineRunsHookOptions {
  limit?: number;
  pollIntervalMs?: number;
}

/**
 * Polls `/api/pipeline/runs`. Default 5s — runs can finish quickly
 * (label) or take minutes (finetune), and we want the dashboard to
 * pick up the status flip from `running` to `success`/`failed`
 * promptly so the operator sees what happened.
 */
export function usePipelineRuns(
  opts: PipelineRunsHookOptions = {},
): PipelineRunsState {
  const [state, setState] = useState<PipelineRunsState>({
    rows: [],
    fetchedAt: null,
    error: null,
    loading: true,
  });
  useEffect(() => {
    let stopped = false;
    const ctrl = new AbortController();
    const interval = Math.max(2000, opts.pollIntervalMs ?? 5_000);
    const limit = opts.limit ?? 50;

    async function step() {
      try {
        const rows = await fetchPipelineRuns(limit, ctrl.signal);
        if (stopped) return;
        setState({ rows, fetchedAt: Date.now(), error: null, loading: false });
      } catch (err) {
        if (stopped) return;
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("aborted")) return;
        setState((prev) => ({ ...prev, error: msg, loading: false }));
      }
    }
    void step();
    const id = setInterval(step, interval);
    return () => {
      stopped = true;
      ctrl.abort();
      clearInterval(id);
    };
  }, [opts.limit, opts.pollIntervalMs]);
  return state;
}

// === pipeline trigger + flight status ====================================

export interface PipelineFlight {
  run_id: string;
  instrument: string;
  started_ms: number;
  child_pid: number | null;
  log_path: string | null;
  finished_ms: number | null;
  exit_status: string | null;
}

export interface PipelineStatusResponse {
  enabled: boolean;
  current: PipelineFlight | null;
}

export function fetchPipelineStatus(
  signal?: AbortSignal,
): Promise<PipelineStatusResponse> {
  return fetchJson<PipelineStatusResponse>(`${BASE}/api/pipeline/status`, signal);
}

export function fetchLastCompletedPipeline(
  signal?: AbortSignal,
): Promise<PipelineFlight | null> {
  return fetchJson<PipelineFlight | null>(`${BASE}/api/pipeline/last-completed`, signal);
}

export interface PipelineRunRequest {
  instrument: string;
  n_bars?: number;
  n_optuna_trials?: number;
  n_trader_trials?: number;
  seed?: number;
}

export interface PipelineRunResponse {
  run_id: string;
  instrument: string;
  started_ms: number;
  log_path: string;
}

/**
 * POST /api/pipeline/run. Returns the spawned run's metadata. The
 * server returns 202 on accept; this client surfaces non-2xx as an
 * Error with the server-provided message so the UI can display it.
 */
export async function triggerPipeline(
  body: PipelineRunRequest,
  signal?: AbortSignal,
): Promise<PipelineRunResponse> {
  const r = await fetch(`${BASE}/api/pipeline/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) {
    const text = await r.text().catch(() => `${r.status} ${r.statusText}`);
    throw new Error(`pipeline trigger ${r.status}: ${text}`);
  }
  return (await r.json()) as PipelineRunResponse;
}

export interface PipelineStatusState {
  enabled: boolean;
  current: PipelineFlight | null;
  fetchedAt: number | null;
  error: string | null;
}

/**
 * Polls /api/pipeline/status. Default 3s — short enough that the
 * "Run pipeline" button flips back to enabled within a couple of
 * frames after a run finishes.
 */
export function usePipelineStatus(
  pollIntervalMs?: number,
): PipelineStatusState {
  const [state, setState] = useState<PipelineStatusState>({
    enabled: false,
    current: null,
    fetchedAt: null,
    error: null,
  });
  useEffect(() => {
    let stopped = false;
    const ctrl = new AbortController();
    const interval = Math.max(1000, pollIntervalMs ?? 3_000);

    async function step() {
      try {
        const r = await fetchPipelineStatus(ctrl.signal);
        if (stopped) return;
        setState({
          enabled: r.enabled,
          current: r.current,
          fetchedAt: Date.now(),
          error: null,
        });
      } catch (err) {
        if (stopped) return;
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("aborted")) return;
        setState((prev) => ({ ...prev, error: msg }));
      }
    }
    void step();
    const id = setInterval(step, interval);
    return () => {
      stopped = true;
      ctrl.abort();
      clearInterval(id);
    };
  }, [pollIntervalMs]);
  return state;
}

// === log tail ============================================================

/**
 * Fetch a tail of the captured stdout/stderr log for `run_id`. Throws
 * a structured error on 404 (`log not found`) so the caller can show a
 * "no log yet" state instead of bubbling up a generic message.
 */
export async function fetchPipelineLog(
  run_id: string,
  tail = 65_536,
  signal?: AbortSignal,
): Promise<string> {
  const r = await fetch(
    `${BASE}/api/pipeline/log?run_id=${encodeURIComponent(run_id)}&tail=${tail}`,
    { headers: { Accept: "text/plain" }, signal },
  );
  if (r.status === 404) {
    throw new Error("log not found");
  }
  if (!r.ok) {
    const t = await r.text().catch(() => `${r.status}`);
    throw new Error(`log fetch ${r.status}: ${t}`);
  }
  return await r.text();
}

export interface PipelineLogState {
  text: string;
  fetchedAt: number | null;
  error: string | null;
  loading: boolean;
}

export interface PipelineLogHookOptions {
  /** Run id to fetch. Pass null to disable polling. */
  runId: string | null;
  /** Whether to keep polling. Set false once the run finishes. */
  live: boolean;
  tail?: number;
  pollIntervalMs?: number;
}

/**
 * Live-tail hook. Polls /api/pipeline/log every `pollIntervalMs`
 * while `live` is true; otherwise does a single fetch when `runId`
 * changes. Cancellation-safe.
 */
export function usePipelineLog(opts: PipelineLogHookOptions): PipelineLogState {
  const [state, setState] = useState<PipelineLogState>({
    text: "",
    fetchedAt: null,
    error: null,
    loading: false,
  });
  useEffect(() => {
    if (!opts.runId) {
      setState({ text: "", fetchedAt: null, error: null, loading: false });
      return;
    }
    let stopped = false;
    const ctrl = new AbortController();
    const interval = Math.max(1000, opts.pollIntervalMs ?? 2_000);
    const tail = opts.tail ?? 65_536;
    setState((s) => ({ ...s, loading: true }));

    async function step() {
      try {
        const text = await fetchPipelineLog(opts.runId!, tail, ctrl.signal);
        if (stopped) return;
        setState({ text, fetchedAt: Date.now(), error: null, loading: false });
      } catch (err) {
        if (stopped) return;
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("aborted")) return;
        setState((prev) => ({
          ...prev,
          error: msg,
          loading: false,
        }));
      }
    }
    void step();
    let id: ReturnType<typeof setInterval> | null = null;
    if (opts.live) {
      id = setInterval(step, interval);
    }
    return () => {
      stopped = true;
      ctrl.abort();
      if (id !== null) clearInterval(id);
    };
  }, [opts.runId, opts.live, opts.tail, opts.pollIntervalMs]);
  return state;
}

// === auto-retrain counters ===============================================

export interface AutoRetrainInstrumentStatus {
  instrument: string;
  bars_since_last_fire: number;
  last_fired_ms: number | null;
  last_skip_reason: string | null;
}

export interface AutoRetrainStatus {
  enabled: boolean;
  bars_threshold: number;
  instruments: AutoRetrainInstrumentStatus[];
}

export function fetchAutoRetrainStatus(
  signal?: AbortSignal,
): Promise<AutoRetrainStatus> {
  return fetchJson<AutoRetrainStatus>(`${BASE}/api/pipeline/auto-retrain`, signal);
}

export interface AutoRetrainState {
  status: AutoRetrainStatus | null;
  fetchedAt: number | null;
  error: string | null;
}

/**
 * Polls /api/pipeline/auto-retrain. Default 5s — counters tick every
 * 10 seconds at most (one per closed bar) so faster polling is wasted.
 */
export function useAutoRetrain(pollIntervalMs?: number): AutoRetrainState {
  const [state, setState] = useState<AutoRetrainState>({
    status: null,
    fetchedAt: null,
    error: null,
  });
  useEffect(() => {
    let stopped = false;
    const ctrl = new AbortController();
    const interval = Math.max(2000, pollIntervalMs ?? 5_000);

    async function step() {
      try {
        const status = await fetchAutoRetrainStatus(ctrl.signal);
        if (stopped) return;
        setState({ status, fetchedAt: Date.now(), error: null });
      } catch (err) {
        if (stopped) return;
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("aborted")) return;
        setState((prev) => ({ ...prev, error: msg }));
      }
    }
    void step();
    const id = setInterval(step, interval);
    return () => {
      stopped = true;
      ctrl.abort();
      clearInterval(id);
    };
  }, [pollIntervalMs]);
  return state;
}

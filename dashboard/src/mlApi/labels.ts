// /api/labels/recent — ideal long/short entry points produced by the
// label optimizer on the trailing 1000-bar window.

import { useEffect, useState } from "react";

import { BASE, fetchJson } from "./internal";

export interface LabelRow {
  instrument: string;
  ts_ms: number;
  t1_ms: number;
  side: number;
  meta_y: number;
  realized_r: number;
  barrier_hit: string;
  oracle_score: number;
  label_run_id: string;
}

export function fetchLabels(
  instrument: string,
  limit = 200,
  signal?: AbortSignal,
): Promise<LabelRow[]> {
  const params = new URLSearchParams({
    instrument,
    limit: String(limit),
  });
  return fetchJson<LabelRow[]>(`${BASE}/api/labels/recent?${params}`, signal);
}

export interface LabelsState {
  rows: LabelRow[];
  fetchedAt: number | null;
  error: string | null;
  loading: boolean;
}

export interface LabelsHookOptions {
  instrument: string;
  limit?: number;
  pollIntervalMs?: number;
}

/**
 * Polls `/api/labels/recent`. The label optimizer only re-runs every
 * 100 closed bars (~17 minutes at 10s bars) so 60s is plenty.
 */
export function useLabels(opts: LabelsHookOptions): LabelsState {
  const [state, setState] = useState<LabelsState>({
    rows: [],
    fetchedAt: null,
    error: null,
    loading: true,
  });
  useEffect(() => {
    let stopped = false;
    const ctrl = new AbortController();
    const interval = Math.max(2000, opts.pollIntervalMs ?? 60_000);
    const limit = opts.limit ?? 500;

    async function step() {
      try {
        const rows = await fetchLabels(opts.instrument, limit, ctrl.signal);
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
  }, [opts.instrument, opts.limit, opts.pollIntervalMs]);
  return state;
}

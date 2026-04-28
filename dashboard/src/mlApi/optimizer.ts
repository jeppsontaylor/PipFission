// /api/optimizer/trials — Optuna NSGA-II trial leaderboard.

import { useEffect, useState } from "react";

import { BASE, fetchJson } from "./internal";

export interface OptimizerTrialRow {
  study_id: string;
  trial_id: number;
  ts_ms: number;
  params_json: string;
  score: number;
  sortino: number;
  max_dd_bp: number;
  turnover: number;
  pareto_rank: number;
}

export function fetchOptimizerTrials(
  limit = 50,
  study?: string,
  signal?: AbortSignal,
): Promise<OptimizerTrialRow[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (study) params.set("study", study);
  return fetchJson<OptimizerTrialRow[]>(
    `${BASE}/api/optimizer/trials?${params}`,
    signal,
  );
}

export interface OptimizerTrialsState {
  rows: OptimizerTrialRow[];
  fetchedAt: number | null;
  error: string | null;
  loading: boolean;
}

export interface OptimizerTrialsHookOptions {
  limit?: number;
  study?: string;
  pollIntervalMs?: number;
}

/**
 * Polls `/api/optimizer/trials`. Returns a snapshot updated every
 * `pollIntervalMs` (default 30s — trials drip in slowly during a
 * retrain so faster polling burns bandwidth for nothing).
 */
export function useOptimizerTrials(
  opts: OptimizerTrialsHookOptions = {},
): OptimizerTrialsState {
  const [state, setState] = useState<OptimizerTrialsState>({
    rows: [],
    fetchedAt: null,
    error: null,
    loading: true,
  });
  useEffect(() => {
    let stopped = false;
    const ctrl = new AbortController();
    const interval = Math.max(2000, opts.pollIntervalMs ?? 30_000);
    const limit = opts.limit ?? 50;

    async function step() {
      try {
        const rows = await fetchOptimizerTrials(limit, opts.study, ctrl.signal);
        if (stopped) return;
        setState({
          rows,
          fetchedAt: Date.now(),
          error: null,
          loading: false,
        });
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
  }, [opts.limit, opts.study, opts.pollIntervalMs]);
  return state;
}

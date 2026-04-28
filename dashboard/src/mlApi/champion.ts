// Champion + model metrics + trader metrics + lockbox endpoints, plus
// the combined `useMlPipeline` hook that powers MLPipelinePanel.

import { useEffect, useState } from "react";

import { BASE, fetchJson } from "./internal";

export interface ChampionResponse {
  model_id: string;
  n_features: number;
  /** "onnx" | "fallback" | "none" | "uninitialised" */
  kind: string;
}

export interface ModelMetricsRow {
  model_id: string;
  instrument: string;
  ts_ms: number;
  oos_auc: number;
  oos_log_loss: number;
  oos_brier: number;
  oos_balanced_acc: number;
  train_sharpe: number;
  train_sortino: number;
  max_train_sortino: number;
  max_train_sharpe: number;
  n_train: number;
  n_oof: number;
}

export interface TraderMetricsRow {
  params_id: string;
  model_id: string;
  ts_ms: number;
  in_sample_sharpe: number;
  in_sample_sortino: number;
  fine_tune_sharpe: number;
  fine_tune_sortino: number;
  max_dd_bp: number;
  turnover_per_day: number;
  hit_rate: number;
  profit_factor: number;
  n_trades: number;
  params_json: string;
}

export interface LockboxRow {
  run_id: string;
  ts_ms: number;
  model_id: string;
  params_id: string;
  /** Server stores the payload as a JSON string in DuckDB. */
  summary_json: string;
  sealed: boolean;
}

export function fetchChampion(signal?: AbortSignal): Promise<ChampionResponse> {
  return fetchJson<ChampionResponse>(`${BASE}/api/strategy/champion`, signal);
}

export function fetchModelMetrics(
  instrument: string,
  signal?: AbortSignal,
): Promise<ModelMetricsRow | null> {
  const url = `${BASE}/api/model/metrics?instrument=${encodeURIComponent(instrument)}`;
  return fetchJson<ModelMetricsRow | null>(url, signal);
}

export function fetchTraderMetrics(signal?: AbortSignal): Promise<TraderMetricsRow | null> {
  return fetchJson<TraderMetricsRow | null>(`${BASE}/api/trader/metrics`, signal);
}

export function fetchLockboxResult(signal?: AbortSignal): Promise<LockboxRow | null> {
  return fetchJson<LockboxRow | null>(`${BASE}/api/lockbox/result`, signal);
}

export interface ModelDeploymentGateRow {
  model_id: string;
  instrument: string;
  ts_ms: number;
  oos_auc: number;
  oos_log_loss: number;
  oos_brier: number;
  oos_balanced_acc: number;
  fine_tune_sortino: number;
  fine_tune_max_dd_bp: number;
  passed_gate: boolean;
  /** Semicolon-separated reasons. Empty when passed. */
  blocked_reasons: string;
  /** JSON string with the threshold values that were in effect. */
  gate_thresholds_json: string;
}

export function fetchDeploymentGate(
  instrument: string,
  signal?: AbortSignal,
): Promise<ModelDeploymentGateRow | null> {
  const url = `${BASE}/api/model/deployment-gate?instrument=${encodeURIComponent(instrument)}`;
  return fetchJson<ModelDeploymentGateRow | null>(url, signal);
}

export interface DeploymentGateState {
  row: ModelDeploymentGateRow | null;
  fetchedAt: number | null;
  error: string | null;
  loading: boolean;
}

/** Polls the most-recent deployment-gate decision for `instrument`.
 *  Default 30s — the table only changes when a retrain completes. */
export function useDeploymentGate(
  instrument: string,
  pollIntervalMs?: number,
): DeploymentGateState {
  const [state, setState] = useState<DeploymentGateState>({
    row: null,
    fetchedAt: null,
    error: null,
    loading: true,
  });
  useEffect(() => {
    let stopped = false;
    const ctrl = new AbortController();
    const interval = Math.max(2000, pollIntervalMs ?? 30_000);
    if (!instrument) {
      setState({ row: null, fetchedAt: null, error: null, loading: false });
      return;
    }
    async function step() {
      try {
        const row = await fetchDeploymentGate(instrument, ctrl.signal);
        if (stopped) return;
        setState({ row, fetchedAt: Date.now(), error: null, loading: false });
      } catch (err) {
        if (stopped) return;
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("aborted")) return;
        setState((p) => ({ ...p, error: msg, loading: false }));
      }
    }
    void step();
    const id = setInterval(step, interval);
    return () => {
      stopped = true;
      ctrl.abort();
      clearInterval(id);
    };
  }, [instrument, pollIntervalMs]);
  return state;
}

export interface ModelCandidateRow {
  run_id: string;
  spec_name: string;
  model_id: string;
  instrument: string;
  ts_ms: number;
  oos_auc: number;
  oos_log_loss: number;
  oos_brier: number;
  oos_balanced_acc: number;
  n_train: number;
  n_oof: number;
  is_winner: boolean;
}

/**
 * Fetch all candidates from the most recent training run for
 * `instrument`. Sorted server-side by OOS log loss (winner first).
 */
export function fetchModelCandidates(
  instrument: string,
  signal?: AbortSignal,
): Promise<ModelCandidateRow[]> {
  const url = `${BASE}/api/model/candidates?instrument=${encodeURIComponent(instrument)}`;
  return fetchJson<ModelCandidateRow[]>(url, signal);
}

export interface ModelCandidatesState {
  rows: ModelCandidateRow[];
  fetchedAt: number | null;
  error: string | null;
  loading: boolean;
}

export function useModelCandidates(
  instrument: string,
  pollIntervalMs?: number,
): ModelCandidatesState {
  const [state, setState] = useState<ModelCandidatesState>({
    rows: [],
    fetchedAt: null,
    error: null,
    loading: true,
  });
  useEffect(() => {
    let stopped = false;
    const ctrl = new AbortController();
    const interval = Math.max(2000, pollIntervalMs ?? 30_000);
    if (!instrument) {
      setState({ rows: [], fetchedAt: null, error: null, loading: false });
      return;
    }
    async function step() {
      try {
        const rows = await fetchModelCandidates(instrument, ctrl.signal);
        if (stopped) return;
        setState({ rows, fetchedAt: Date.now(), error: null, loading: false });
      } catch (err) {
        if (stopped) return;
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("aborted")) return;
        setState((p) => ({ ...p, error: msg, loading: false }));
      }
    }
    void step();
    const id = setInterval(step, interval);
    return () => {
      stopped = true;
      ctrl.abort();
      clearInterval(id);
    };
  }, [instrument, pollIntervalMs]);
  return state;
}

export interface MLPipelineSummary {
  champion: ChampionResponse | null;
  modelMetrics: ModelMetricsRow | null;
  traderMetrics: TraderMetricsRow | null;
  lockbox: LockboxRow | null;
  /** Last successful refresh ts in ms; null while loading. */
  fetchedAt: number | null;
  /** Most recent error string, if any. Cleared on successful refresh. */
  error: string | null;
  loading: boolean;
}

export interface MLPipelineHookOptions {
  instrument: string;
  pollIntervalMs?: number;
}

/**
 * Polls the ML pipeline REST endpoints. Returns a summary that updates
 * on `pollIntervalMs` (default 10s). Each poll runs the four GETs in
 * parallel; if any one fails, the others still update and `error` is
 * set to the last failure. Cancellation-safe via AbortController.
 */
export function useMlPipeline(opts: MLPipelineHookOptions): MLPipelineSummary {
  const [state, setState] = useState<MLPipelineSummary>({
    champion: null,
    modelMetrics: null,
    traderMetrics: null,
    lockbox: null,
    fetchedAt: null,
    error: null,
    loading: true,
  });

  useEffect(() => {
    let stopped = false;
    const ctrl = new AbortController();
    const interval = Math.max(2000, opts.pollIntervalMs ?? 10_000);

    async function step() {
      const results = await Promise.allSettled([
        fetchChampion(ctrl.signal),
        fetchModelMetrics(opts.instrument, ctrl.signal),
        fetchTraderMetrics(ctrl.signal),
        fetchLockboxResult(ctrl.signal),
      ]);
      if (stopped) return;
      const errors: string[] = [];
      const next: MLPipelineSummary = {
        champion: state.champion,
        modelMetrics: state.modelMetrics,
        traderMetrics: state.traderMetrics,
        lockbox: state.lockbox,
        fetchedAt: Date.now(),
        error: null,
        loading: false,
      };
      const [c, m, t, l] = results;
      if (c.status === "fulfilled") next.champion = c.value;
      else errors.push(String(c.reason));
      if (m.status === "fulfilled") next.modelMetrics = m.value;
      else errors.push(String(m.reason));
      if (t.status === "fulfilled") next.traderMetrics = t.value;
      else errors.push(String(t.reason));
      if (l.status === "fulfilled") next.lockbox = l.value;
      else errors.push(String(l.reason));
      next.error = errors.length ? errors.join(" · ") : null;
      setState(next);
    }
    void step();
    const id = setInterval(step, interval);
    return () => {
      stopped = true;
      ctrl.abort();
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [opts.instrument, opts.pollIntervalMs]);

  return state;
}

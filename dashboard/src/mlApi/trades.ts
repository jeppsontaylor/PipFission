// /api/trade/ledger — closed round-trip trades from the live trader.

import { useEffect, useState } from "react";

import { BASE, fetchJson } from "./internal";

export interface TradeLedgerRow {
  run_id: string;
  instrument: string;
  ts_in_ms: number;
  ts_out_ms: number;
  /** -1 short, +1 long. */
  side: number;
  qty: number;
  entry_px: number;
  exit_px: number;
  fee: number;
  slip: number;
  /** Net realized return (after fees + slippage), in fractional units. */
  realized_r: number;
  exit_reason: string;
  // Phase D enrichment (nullable: pre-Phase-D rows return null).
  model_id: string | null;
  params_id: string | null;
  entry_p_long: number | null;
  entry_p_short: number | null;
  entry_calibrated: number | null;
  entry_spread_bp: number | null;
  entry_atr_14: number | null;
  exit_p_long: number | null;
  exit_p_short: number | null;
  decision_chain: string | null;
  snapshot_path: string | null;
}

/**
 * Phase D /api/trade/context payload. Field set varies based on
 * whether the per-trade JSON snapshot file is available — when it
 * is, `source === "snapshot_file"` and the `*_bars` arrays are
 * populated from the writer. When not, `source === "trade_ledger_only"`
 * and only the trade_ledger row's denormalised fields are returned.
 */
export interface TradeContextBar {
  ts_ms: number;
  open: number;
  high: number;
  low: number;
  close: number;
  n_ticks: number;
  spread_bp_avg: number;
}

export interface TradeContext {
  source: "snapshot_file" | "trade_ledger_only";
  schema_version?: number;
  run_id: string;
  instrument: string;
  side: number;
  qty?: number;
  entry_ts_ms: number;
  entry_price: number;
  exit_ts_ms: number;
  exit_price: number;
  realized_r: number;
  exit_reason: string;
  entry_iso?: string;
  exit_iso?: string;
  model_id: string | null;
  params_id: string | null;
  entry_p_long: number | null;
  entry_p_short: number | null;
  entry_calibrated: number | null;
  entry_spread_bp: number | null;
  entry_atr_14: number | null;
  exit_p_long: number | null;
  exit_p_short: number | null;
  decision_chain: string | string[] | null;
  snapshot_path: string | null;
  pre_trade_bars: TradeContextBar[];
  in_trade_bars: TradeContextBar[];
  post_trade_bars: TradeContextBar[];
}

export async function fetchTradeContext(
  run_id: string,
  ts_in_ms: number,
  signal?: AbortSignal,
): Promise<TradeContext> {
  const url = `${BASE}/api/trade/context?run_id=${encodeURIComponent(run_id)}&ts_in_ms=${ts_in_ms}`;
  return fetchJson<TradeContext>(url, signal);
}

export function fetchTradeLedger(
  limit = 200,
  instrument?: string,
  signal?: AbortSignal,
): Promise<TradeLedgerRow[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (instrument) params.set("instrument", instrument);
  return fetchJson<TradeLedgerRow[]>(
    `${BASE}/api/trade/ledger?${params}`,
    signal,
  );
}

export interface TradeLedgerState {
  rows: TradeLedgerRow[];
  fetchedAt: number | null;
  error: string | null;
  loading: boolean;
}

export interface TradeLedgerHookOptions {
  /** Optional. When omitted, returns trades across every instrument. */
  instrument?: string;
  limit?: number;
  pollIntervalMs?: number;
}

/**
 * Polls `/api/trade/ledger`. Default 15s — closed trades only land
 * after a position closes, so faster than that is wasted bandwidth.
 */
export function useTradeLedger(opts: TradeLedgerHookOptions = {}): TradeLedgerState {
  const [state, setState] = useState<TradeLedgerState>({
    rows: [],
    fetchedAt: null,
    error: null,
    loading: true,
  });
  useEffect(() => {
    let stopped = false;
    const ctrl = new AbortController();
    const interval = Math.max(2000, opts.pollIntervalMs ?? 15_000);
    const limit = opts.limit ?? 200;

    async function step() {
      try {
        const rows = await fetchTradeLedger(limit, opts.instrument, ctrl.signal);
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

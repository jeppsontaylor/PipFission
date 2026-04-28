// Tiny client for the server's `/api/history` endpoint. Used by the
// store on first connect to prefill the chart buffers with the last
// ~1000 points per instrument so the dashboard renders a populated
// chart immediately on reload — even on a quiet weekend feed.

export interface PriceHistoryPoint {
  time_ms: number;
  bid: number;
  ask: number;
  mid: number;
  spread: number;
  status: string | null;
}

export interface SignalHistoryPoint {
  time_ms: number;
  direction: "long" | "flat" | "short";
  confidence: number;
  prob_long: number;
  prob_flat: number;
  prob_short: number;
  model_id: string;
  model_version: number;
}

export interface FillHistoryPoint {
  instrument: string;
  time_ms: number;
  units: number;
  price: number;
  fee: number;
  mode: string;
  order_id: string;
}

interface PriceResp {
  kind: "price";
  instrument: string;
  points: PriceHistoryPoint[];
}
interface SignalResp {
  kind: "signal";
  instrument: string;
  points: SignalHistoryPoint[];
}
interface FillResp {
  kind: "fill";
  instrument: string | null;
  points: FillHistoryPoint[];
}

/**
 * The server clamps `limit` server-side at 10 000; passing higher is
 * a no-op. 1000 is the dashboard default — enough for a meaningful
 * chart without melting an idle CPU.
 */
export const DEFAULT_HISTORY_LIMIT = 1000;

const BASE = ""; // same-origin via Vite proxy / served-from-server

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url, { headers: { Accept: "application/json" } });
  if (!r.ok) throw new Error(`history fetch ${url}: ${r.status} ${r.statusText}`);
  return (await r.json()) as T;
}

export async function fetchPriceHistory(
  instrument: string,
  limit = DEFAULT_HISTORY_LIMIT,
): Promise<PriceHistoryPoint[]> {
  const url = `${BASE}/api/history?kind=price&instrument=${encodeURIComponent(instrument)}&limit=${limit}`;
  const r = await fetchJson<PriceResp>(url);
  return r.points;
}

export async function fetchSignalHistory(
  instrument: string,
  limit = DEFAULT_HISTORY_LIMIT,
): Promise<SignalHistoryPoint[]> {
  const url = `${BASE}/api/history?kind=signal&instrument=${encodeURIComponent(instrument)}&limit=${limit}`;
  const r = await fetchJson<SignalResp>(url);
  return r.points;
}

export async function fetchFillHistory(
  limit = DEFAULT_HISTORY_LIMIT,
  instrument?: string,
): Promise<FillHistoryPoint[]> {
  const params = new URLSearchParams({ kind: "fill", limit: String(limit) });
  if (instrument) params.set("instrument", instrument);
  const r = await fetchJson<FillResp>(`${BASE}/api/history?${params}`);
  return r.points;
}

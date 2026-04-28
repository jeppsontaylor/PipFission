// Shared low-level helpers used by every per-resource module in this
// directory. Not re-exported from index.ts — consumers should never
// import directly from here.

/** Same-origin base URL. Vite dev proxies `/api/*` to the Rust server;
 *  the production bundle is served from the same host. */
export const BASE = "";

export async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(url, {
    headers: { Accept: "application/json" },
    signal,
  });
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status} ${r.statusText}`);
  return (await r.json()) as T;
}

// Pure banner-state derivation from `championStatus`. Kept out of the
// React component so the priority logic (error > swap > idle) is
// straightforward to unit test.

import type { ChampionStatus } from "./mlState";

export type BannerKind = "swapped" | "error" | "idle";

export interface BannerState {
  kind: BannerKind;
  /** The current champion model id; surfaces in either banner. */
  modelId: string;
  /** Predictor kind ("onnx" / "fallback") for the swapped banner. */
  modelKind: string;
  /** Reason string from `champion_load_failed`. Set only when kind=error. */
  errorReason: string | null;
  /** ms since the last successful swap. Set only when kind=swapped. */
  ageMs: number | null;
}

/** Default window for the "swapped" banner — 30 seconds. After that we
 *  return to idle so the toast doesn't linger. Configurable for tests. */
export const DEFAULT_SWAP_BANNER_MS = 30_000;

/**
 * Decide what (if anything) the banner should show.
 *
 * Priority:
 *   1. `error` — `last_error` is set. Persistent until cleared by a
 *      successful swap (which sets `last_error = null`).
 *   2. `swapped` — `last_changed_at` was within `windowMs`, and the
 *      model id is real (not the bootstrap sentinel).
 *   3. `idle` — show nothing.
 *
 * Pure: same inputs → same output, no `Date.now()`.
 */
export function pickBannerState(
  status: ChampionStatus,
  now: number,
  windowMs: number = DEFAULT_SWAP_BANNER_MS,
): BannerState {
  // Error wins.
  if (status.last_error) {
    return {
      kind: "error",
      modelId: status.model_id,
      modelKind: status.kind,
      errorReason: status.last_error,
      ageMs: null,
    };
  }
  // Swap within window — only if a real swap actually happened.
  if (
    status.last_changed_at !== null &&
    status.model_id !== "uninitialised"
  ) {
    const age = now - status.last_changed_at;
    if (age >= 0 && age <= windowMs) {
      return {
        kind: "swapped",
        modelId: status.model_id,
        modelKind: status.kind,
        errorReason: null,
        ageMs: age,
      };
    }
  }
  return {
    kind: "idle",
    modelId: status.model_id,
    modelKind: status.kind,
    errorReason: null,
    ageMs: null,
  };
}

/** Short, human-readable model-id for banners — keep the head + tail
 *  so the operator can still recognise the version. */
export function shortModelId(id: string, head = 8, tail = 8): string {
  if (id.length <= head + tail + 1) return id;
  return `${id.slice(0, head)}…${id.slice(-tail)}`;
}

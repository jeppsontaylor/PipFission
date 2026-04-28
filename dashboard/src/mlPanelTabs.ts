// Pure helpers for the MLPipelinePanel's instrument tab strip. Kept
// separate so the "what colour is this tab's dot" + "if the selected
// instrument disappeared, what should we pick" logic is unit testable
// without mounting any React.

import type { ChampionSignal } from "./types";

export type TabTone = "long" | "short" | "flat" | "idle";

/**
 * Decide a tab's status dot from the latest champion signal:
 *   * long  — `p_long >= threshold` (default 0.55) and clearly above p_short
 *   * short — `p_short >= threshold` and clearly above p_long
 *   * flat  — signal exists but neither side is decisive
 *   * idle  — no champion signal received yet for that instrument
 */
export function pickTabTone(
  signal: ChampionSignal | undefined,
  threshold: number = 0.55,
): TabTone {
  if (!signal) return "idle";
  const longD = signal.p_long;
  const shortD = signal.p_short;
  if (longD >= threshold && longD > shortD + 0.02) return "long";
  if (shortD >= threshold && shortD > longD + 0.02) return "short";
  return "flat";
}

/**
 * Pick the next selected tab when the available `instruments` list
 * changes. Mirrors the existing convention in LiveChartView: prefer
 * the user's current pick, otherwise fall back to the first
 * available, otherwise empty string.
 */
export function pickSelectedInstrument(
  instruments: ReadonlyArray<string>,
  current: string,
): string {
  if (instruments.length === 0) return "";
  if (instruments.includes(current)) return current;
  return instruments[0];
}

export function tabToneClasses(tone: TabTone, selected: boolean): string {
  const dot = (() => {
    switch (tone) {
      case "long":
        return "bg-emerald-400";
      case "short":
        return "bg-rose-400";
      case "flat":
        return "bg-slate-400";
      case "idle":
      default:
        return "bg-slate-700";
    }
  })();
  return selected
    ? `border-ink-500 bg-ink-700/80 text-slate-100 ${dot}`
    : `border-ink-700/60 bg-ink-800/40 text-slate-400 hover:text-slate-200 hover:bg-ink-700/60 ${dot}`;
}

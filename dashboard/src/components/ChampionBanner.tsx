// Transient + persistent champion banner.
//
// Two states, picked by `pickBannerState`:
//
//   * "swapped" (~30s after a hot-swap): green strip — "New champion
//     deployed: <id>" — so the operator notices a model went live.
//
//   * "error" (persistent until cleared by a successful swap): rose
//     strip — "Champion load failed: <reason>" — surfaces a botched
//     ONNX / shape-mismatch event.
//
// Polls `Date.now()` once a second so the banner self-dismisses
// without external coordination.

import { useEffect, useState } from "react";

import { pickBannerState, shortModelId } from "../championBanner";
import type { ChampionStatus } from "../mlState";

interface Props {
  status: ChampionStatus;
}

export function ChampionBanner({ status }: Props) {
  // 1 Hz tick so the "swapped" banner self-dismisses after the window.
  // For "error" the value of `now` doesn't matter — the helper only
  // looks at `last_error` — but ticking is cheap.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const banner = pickBannerState(status, now);
  if (banner.kind === "idle") return null;

  if (banner.kind === "error") {
    return (
      <div
        role="alert"
        className="rounded-md border border-rose-700/60 bg-rose-950/40 px-3 py-2 text-sm text-rose-100 flex items-start gap-2"
      >
        <span className="font-semibold">Champion load failed</span>
        <span className="text-rose-200/90 break-all">{banner.errorReason}</span>
        <span
          className="ml-auto font-mono text-[11px] text-rose-200/60 shrink-0"
          title={banner.modelId}
        >
          current: {shortModelId(banner.modelId)}
        </span>
      </div>
    );
  }

  // banner.kind === "swapped"
  const ageSec = banner.ageMs !== null ? Math.floor(banner.ageMs / 1000) : 0;
  return (
    <div
      role="status"
      className="rounded-md border border-emerald-700/60 bg-emerald-950/40 px-3 py-2 text-sm text-emerald-100 flex items-center gap-2"
    >
      <span aria-hidden="true">✨</span>
      <span className="font-semibold">New champion deployed</span>
      <span
        className="font-mono text-emerald-200/90 break-all"
        title={banner.modelId}
      >
        {shortModelId(banner.modelId)}
      </span>
      <span className="text-[11px] text-emerald-200/70 shrink-0">
        ({banner.modelKind} · {ageSec}s ago)
      </span>
    </div>
  );
}

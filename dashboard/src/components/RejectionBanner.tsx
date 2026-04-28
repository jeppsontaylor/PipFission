import type { RoutingMode } from "../types";

export interface RejectionBannerProps {
  mode: RoutingMode;
  count: number;
  topReason: string | null;
  topReasonCount: number;
}

/**
 * Loud banner that surfaces OANDA order rejections.
 *
 * The most common cause is `MARKET_HALTED` — forex closes Sat 22:00 UTC
 * → Sun 22:00 UTC, and during that window OANDA cancels every market
 * order with that reason. The user needs to see this immediately rather
 * than wonder why their account isn't moving.
 */
export function RejectionBanner({ mode, count, topReason, topReasonCount }: RejectionBannerProps) {
  if (mode !== "oanda_practice" || count === 0) return null;

  const isMarketHalted = topReason === "MARKET_HALTED";
  const headline = isMarketHalted
    ? "OANDA is rejecting every order — MARKET_HALTED"
    : `OANDA rejected ${count} order${count === 1 ? "" : "s"} in the last minute`;

  const subheadline = isMarketHalted ? (
    <>
      Forex markets are <strong>closed</strong> (Sat 22:00 UTC → Sun 22:00 UTC). The strategy is
      sending real orders to your OANDA practice account, but OANDA cancels every one of them with{" "}
      <code className="font-mono text-amber-200">MARKET_HALTED</code>. Your $100k account will not
      change until markets reopen Sunday 22:00 UTC. The internal paper book keeps marking-to-market
      against the latest known prices.
    </>
  ) : (
    <>
      Most common reason: <code className="font-mono text-amber-200">{topReason ?? "UNKNOWN"}</code>{" "}
      ({topReasonCount}× in the last minute). Check the OANDA fills log below for details.
    </>
  );

  return (
    <div className="rounded-xl border border-amber-500/60 bg-amber-500/10 px-4 py-3 flex gap-3 items-start">
      <div className="text-amber-300 text-xl mt-0.5" aria-hidden="true">
        ⚠
      </div>
      <div className="flex-1">
        <div className="text-sm font-semibold text-amber-200">{headline}</div>
        <div className="text-xs text-amber-100/80 mt-1 leading-relaxed">{subheadline}</div>
      </div>
      <div className="text-right">
        <div className="text-2xl font-mono text-amber-200 leading-none">{count}</div>
        <div className="text-[10px] uppercase tracking-wider text-amber-300/80">last 60s</div>
      </div>
    </div>
  );
}

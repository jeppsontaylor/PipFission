import { useEffect, useRef, useState } from "react";
import type { PriceTick } from "../types";
import { instrumentVenue } from "../types";
import type { PricePoint } from "../store";
import { fmtPx, instrumentDecimals } from "../store";
import { Card } from "./Card";
import { Sparkline } from "./Sparkline";

interface Props {
  prices: Record<string, PriceTick>;
  history: Record<string, PricePoint[]>;
  instruments: string[];
}

export function PriceGrid({ prices, history, instruments }: Props) {
  return (
    <Card title="Live prices" right={<span className="text-xs text-slate-400">{instruments.length} pairs · streaming</span>}>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {instruments.map((inst) => (
          <PriceCell
            key={inst}
            instrument={inst}
            tick={prices[inst]}
            hist={history[inst] ?? []}
          />
        ))}
        {instruments.length === 0 && (
          <div className="col-span-full text-xs text-slate-500 italic px-2 py-3">
            No instruments match this filter.
          </div>
        )}
      </div>
    </Card>
  );
}

function PriceCell({
  instrument,
  tick,
  hist,
}: {
  instrument: string;
  tick: PriceTick | undefined;
  hist: PricePoint[];
}) {
  const decimals = instrumentDecimals(instrument);
  const lastMidRef = useRef<number | undefined>(undefined);
  const [flash, setFlash] = useState<"up" | "down" | null>(null);

  useEffect(() => {
    if (tick === undefined) return;
    const prev = lastMidRef.current;
    if (prev !== undefined && prev !== tick.mid) {
      setFlash(tick.mid > prev ? "up" : "down");
      const t = setTimeout(() => setFlash(null), 600);
      return () => clearTimeout(t);
    }
    lastMidRef.current = tick.mid;
  }, [tick?.mid, tick]);

  const spreadBps = tick && tick.mid > 0 ? (tick.spread / tick.mid) * 10_000 : 0;

  // Derive sparkline trend (positive if last > first).
  let trendPositive: boolean | undefined = undefined;
  if (hist.length > 1) {
    trendPositive = hist[hist.length - 1].mid >= hist[0].mid;
  }

  const flashClass =
    flash === "up" ? "flash-up" : flash === "down" ? "flash-down" : "";
  const venue = instrumentVenue(instrument);
  const venueBadge =
    venue === "alpaca"
      ? "bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-500/40"
      : "bg-accent-500/15 text-accent-300 border-accent-500/40";
  const venueLabel = venue === "alpaca" ? "ALPACA" : "OANDA";

  return (
    <div
      className={`rounded-lg border border-ink-700 bg-ink-800/60 px-3 py-2 transition-colors ${flashClass}`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <div className="text-sm font-semibold tracking-wide truncate">
            {instrument.replace("_", "/")}
          </div>
          <span
            className={`inline-block px-1.5 py-0.5 text-[9px] tracking-widest rounded border font-mono ${venueBadge}`}
          >
            {venueLabel}
          </span>
        </div>
        <Sparkline
          values={hist.map((p) => p.mid)}
          width={70}
          height={20}
          positive={trendPositive}
        />
      </div>
      <div className="mt-1 grid grid-cols-2 gap-2 font-mono text-sm">
        <div>
          <div className="text-[10px] uppercase text-slate-500">Bid</div>
          <div className="text-bear">{fmtPx(tick?.bid, decimals)}</div>
        </div>
        <div className="text-right">
          <div className="text-[10px] uppercase text-slate-500">Ask</div>
          <div className="text-bull">{fmtPx(tick?.ask, decimals)}</div>
        </div>
      </div>
      <div className="mt-1 flex items-center justify-between text-[11px] text-slate-400 font-mono">
        <span>mid {fmtPx(tick?.mid, decimals)}</span>
        <span>spread {spreadBps.toFixed(2)}bps</span>
      </div>
    </div>
  );
}

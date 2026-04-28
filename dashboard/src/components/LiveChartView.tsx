import { useEffect, useMemo, useState } from "react";

import type {
  ModelFitness,
  PaperFillEvent,
  PriceTick,
  StrategySignal,
} from "../types";
import { instrumentVenue } from "../types";
import type { PricePoint, SignalPoint } from "../store";
import { useLabels } from "../mlApi";
import { LiveChart } from "./LiveChart";

interface Props {
  instruments: string[];
  prices: Record<string, PriceTick>;
  priceHistory: Record<string, PricePoint[]>;
  fills: PaperFillEvent[];
  signals: Record<string, StrategySignal>;
  signalHistory: Record<string, SignalPoint[]>;
  fitness: Record<string, ModelFitness>;
}

/**
 * Picks an instrument and renders a full-width LiveChart for it.
 * Defaults to the first crypto symbol seen (BTC) so the live demo opens
 * on a moving market. If only forex symbols are available, picks the
 * first of those.
 */
export function LiveChartView({
  instruments,
  prices,
  priceHistory,
  fills,
  signals,
  signalHistory,
  fitness,
}: Props) {
  // Order: crypto pairs first (always-on, easier demo), forex after.
  const ordered = useMemo(() => {
    const crypto = instruments.filter((i) => instrumentVenue(i) === "alpaca");
    const fx = instruments.filter((i) => instrumentVenue(i) === "oanda");
    return [...crypto, ...fx];
  }, [instruments]);

  const [selected, setSelected] = useState<string>(ordered[0] ?? "");

  // If the selected instrument disappears (filter changed elsewhere),
  // fall back to the first available one.
  useEffect(() => {
    if (ordered.length === 0) return;
    if (!ordered.includes(selected)) {
      setSelected(ordered[0]);
    }
  }, [ordered, selected]);

  // Polled overlay of the optimizer's ideal long/short entries for the
  // currently-selected instrument. Hook is hoisted here (rather than
  // inside LiveChart) so the chart stays a pure presentation component.
  const labels = useLabels({ instrument: selected, limit: 500 });

  if (ordered.length === 0) {
    return (
      <div className="rounded-xl border border-ink-700 bg-ink-900/60 p-8 text-center text-slate-500">
        No instruments yet — waiting for the WebSocket hello payload…
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 overflow-x-auto pb-1">
        {ordered.map((inst) => {
          const isSel = inst === selected;
          const venue = instrumentVenue(inst);
          const venueAccent =
            venue === "alpaca"
              ? "border-fuchsia-500/60 text-fuchsia-200"
              : "border-blue-500/60 text-blue-200";
          const baseCls = isSel
            ? `bg-ink-700/80 ${venueAccent}`
            : "bg-ink-800/40 border-ink-700/60 text-slate-400 hover:bg-ink-700/60 hover:text-slate-200";
          const tick = prices[inst];
          return (
            <button
              key={inst}
              type="button"
              onClick={() => setSelected(inst)}
              className={`px-3 py-1.5 rounded-md border font-mono text-xs whitespace-nowrap transition-colors ${baseCls}`}
            >
              <span className="font-semibold">{inst.replace("_", "/")}</span>
              {tick && (
                <span className="ml-2 text-[10px] text-slate-500">
                  {venue === "alpaca" ? tick.mid.toFixed(2) : tick.mid.toFixed(5)}
                </span>
              )}
            </button>
          );
        })}
      </div>

      <LiveChart
        instrument={selected}
        tick={prices[selected]}
        history={priceHistory[selected] ?? []}
        fills={fills}
        signal={signals[selected]}
        signalHistory={signalHistory[selected] ?? []}
        fitness={fitness[selected]}
        labels={labels.rows}
      />
    </div>
  );
}

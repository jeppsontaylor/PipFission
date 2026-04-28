import { useMemo, useState } from "react";

import { Card } from "./Card";
import type { RoutingMode } from "../types";

export interface ManualOrderPanelProps {
  instruments: string[];
  mode: RoutingMode;
  onSubmit: (instrument: string, units: number) => void;
}

const QUICK_AMOUNTS = [
  { label: "+100", units: 100 },
  { label: "+500", units: 500 },
  { label: "+1k", units: 1000 },
  { label: "−100", units: -100 },
  { label: "−500", units: -500 },
  { label: "−1k", units: -1000 },
];

/**
 * Simple test-order panel. Picks an instrument, picks a signed unit
 * amount, fires through the WS. Routes the same way as strategy-emitted
 * intents:
 *
 * - Internal mode → in-process paper book.
 * - OANDA Paper mode → forex to OANDA practice (`MARKET_HALTED` on
 *   weekends), crypto to Alpaca paper (24/7).
 *
 * For Alpaca crypto, units are mapped to dollar notional at the router
 * boundary: 1 unit ≈ $0.10, with a $10 minimum order size.
 */
export function ManualOrderPanel({ instruments, mode, onSubmit }: ManualOrderPanelProps) {
  const sorted = useMemo(() => [...instruments].sort(), [instruments]);
  const [instrument, setInstrument] = useState<string>(sorted[0] ?? "");
  const [customUnits, setCustomUnits] = useState<string>("");

  const isCrypto = instrument.includes("/");
  const venueLabel = isCrypto ? "Alpaca crypto" : "OANDA forex";
  const isOandaPaper = mode === "oanda_practice";
  const targetLabel = !isOandaPaper
    ? "internal paper book"
    : isCrypto
      ? "real Alpaca paper account"
      : "real OANDA practice account";

  const fire = (units: number) => {
    if (!instrument || units === 0) return;
    onSubmit(instrument, units);
  };

  const fireCustom = () => {
    const u = parseInt(customUnits, 10);
    if (Number.isFinite(u) && u !== 0) {
      onSubmit(instrument, u);
      setCustomUnits("");
    }
  };

  return (
    <Card title="Manual test order">
      <div className="space-y-3 text-xs">
        <div>
          <label className="block text-[11px] uppercase tracking-wider text-slate-500 mb-1">
            Instrument
          </label>
          <select
            value={instrument}
            onChange={(e) => setInstrument(e.target.value)}
            className="w-full bg-ink-800 border border-ink-700 rounded px-2 py-1.5 font-mono text-slate-100 focus:outline-none focus:border-accent-500"
          >
            {sorted.map((i) => (
              <option key={i} value={i}>
                {i.replace("_", "/")} {i.includes("/") ? "(Alpaca)" : "(OANDA)"}
              </option>
            ))}
          </select>
        </div>

        <div>
          <div className="text-[11px] uppercase tracking-wider text-slate-500 mb-1">
            Quick orders
          </div>
          <div className="grid grid-cols-3 gap-2">
            {QUICK_AMOUNTS.map((q) => {
              const isBuy = q.units > 0;
              const cls = isBuy
                ? "border-bull-500/40 bg-bull-500/10 text-bull-300 hover:bg-bull-500/20"
                : "border-bear-500/40 bg-bear-500/10 text-bear-300 hover:bg-bear-500/20";
              return (
                <button
                  key={q.label}
                  type="button"
                  onClick={() => fire(q.units)}
                  className={`px-2 py-1.5 rounded border font-mono text-xs ${cls}`}
                >
                  {q.label}
                </button>
              );
            })}
          </div>
        </div>

        <div>
          <label className="block text-[11px] uppercase tracking-wider text-slate-500 mb-1">
            Custom units (signed)
          </label>
          <div className="flex gap-2">
            <input
              type="number"
              value={customUnits}
              onChange={(e) => setCustomUnits(e.target.value)}
              placeholder="e.g. 250 or -300"
              className="flex-1 bg-ink-800 border border-ink-700 rounded px-2 py-1.5 font-mono text-slate-100 focus:outline-none focus:border-accent-500"
              onKeyDown={(e) => {
                if (e.key === "Enter") fireCustom();
              }}
            />
            <button
              type="button"
              onClick={fireCustom}
              className="px-3 py-1.5 rounded bg-accent-500 hover:bg-accent-400 text-white font-medium"
            >
              Fire
            </button>
          </div>
        </div>

        <div className="text-[11px] text-slate-400 leading-relaxed border-t border-ink-800 pt-2">
          → Routes to <strong>{targetLabel}</strong> ({venueLabel}).{" "}
          {isCrypto && isOandaPaper ? (
            <>
              Crypto orders use $X notional: 1 unit ≈ $0.10, $10 minimum (Alpaca rule).
            </>
          ) : isCrypto ? (
            <>Switch to OANDA Paper mode for real Alpaca fills.</>
          ) : (
            <>OANDA forex is closed Sat 22:00 → Sun 22:00 UTC; orders will be MARKET_HALTED.</>
          )}
        </div>
      </div>
    </Card>
  );
}

import type { Venue } from "../types";

export type VenueChoice = "all" | Venue;

export interface VenueFilterProps {
  value: VenueChoice;
  onChange: (v: VenueChoice) => void;
  forexCount: number;
  cryptoCount: number;
}

const buttonBase =
  "px-3 py-1.5 text-xs font-medium tracking-wide transition-colors disabled:opacity-50";

export function VenueFilter({ value, onChange, forexCount, cryptoCount }: VenueFilterProps) {
  return (
    <div className="inline-flex rounded-full border border-ink-700 bg-ink-900 overflow-hidden text-slate-200" role="group" aria-label="Venue filter">
      <button
        type="button"
        onClick={() => onChange("all")}
        className={`${buttonBase} ${value === "all" ? "bg-ink-700 text-slate-100" : "text-slate-400 hover:text-slate-200"}`}
      >
        All ({forexCount + cryptoCount})
      </button>
      <button
        type="button"
        onClick={() => onChange("oanda")}
        className={`${buttonBase} ${value === "oanda" ? "bg-accent-500 text-white" : "text-slate-400 hover:text-slate-200"}`}
      >
        OANDA · Forex ({forexCount})
      </button>
      <button
        type="button"
        onClick={() => onChange("alpaca")}
        className={`${buttonBase} ${value === "alpaca" ? "bg-fuchsia-600 text-white" : "text-slate-400 hover:text-slate-200"}`}
      >
        Alpaca · Crypto ({cryptoCount})
      </button>
    </div>
  );
}

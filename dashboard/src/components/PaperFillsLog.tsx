import { Card } from "./Card";
import { fmtPx, instrumentDecimals } from "../store";
import type { PaperFillEvent } from "../types";

export interface PaperFillsLogProps {
  fills: PaperFillEvent[];
}

export function PaperFillsLog({ fills }: PaperFillsLogProps) {
  const ordered = [...fills].slice().reverse();
  return (
    <Card title={`Paper fills · ${fills.length}`}>
      <div className="overflow-hidden rounded-lg border border-ink-800 max-h-96 overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="bg-ink-800/60 text-slate-400 sticky top-0">
            <tr>
              <th className="text-left px-3 py-2 font-medium">Time</th>
              <th className="text-left px-3 py-2 font-medium">Mode</th>
              <th className="text-left px-3 py-2 font-medium">Instrument</th>
              <th className="text-right px-3 py-2 font-medium">Units</th>
              <th className="text-right px-3 py-2 font-medium">Price</th>
              <th className="text-left px-3 py-2 font-medium">Order ID</th>
            </tr>
          </thead>
          <tbody>
            {ordered.length === 0 && (
              <tr>
                <td className="px-3 py-3 text-slate-500" colSpan={6}>
                  No fills yet — strategy hasn't taken a trade.
                </td>
              </tr>
            )}
            {ordered.map((f) => {
              const ts = new Date(f.time).toLocaleTimeString();
              const dec = instrumentDecimals(f.instrument);
              const unitsClass = f.units > 0 ? "text-bull-400" : "text-bear-400";
              const modeClass =
                f.mode === "oanda_practice"
                  ? "text-amber-300 border-amber-500/40 bg-amber-500/10"
                  : "text-slate-300 border-ink-700 bg-ink-800/60";
              return (
                <tr key={f.order_id} className="border-t border-ink-800">
                  <td className="px-3 py-1.5 font-mono text-slate-400">{ts}</td>
                  <td className="px-3 py-1.5">
                    <span className={`inline-block px-1.5 py-0.5 text-[10px] tracking-widest rounded border font-mono ${modeClass}`}>
                      {f.mode === "oanda_practice" ? "OANDA" : "INTL"}
                    </span>
                  </td>
                  <td className="px-3 py-1.5 font-mono text-slate-300">{f.instrument}</td>
                  <td className={`px-3 py-1.5 text-right font-mono ${unitsClass}`}>
                    {f.units > 0 ? "+" : ""}
                    {f.units}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">
                    {fmtPx(f.price, dec)}
                  </td>
                  <td className="px-3 py-1.5 font-mono text-slate-500 text-[11px] truncate">
                    {f.order_id}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

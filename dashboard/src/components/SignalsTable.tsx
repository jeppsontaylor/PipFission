import { Card } from "./Card";
import type { StrategySignal } from "../types";

export interface SignalsTableProps {
  signals: Record<string, StrategySignal>;
  instruments: string[];
}

const dirLabel: Record<StrategySignal["direction"], string> = {
  long: "LONG",
  flat: "FLAT",
  short: "SHORT",
};

const dirClass: Record<StrategySignal["direction"], string> = {
  long: "text-bull-400 border-bull-500/40 bg-bull-500/10",
  flat: "text-slate-300 border-ink-600 bg-ink-800/60",
  short: "text-bear-400 border-bear-500/40 bg-bear-500/10",
};

export function SignalsTable({ signals, instruments }: SignalsTableProps) {
  const sorted = [...instruments].sort();
  return (
    <Card title="Live signals">
      <div className="overflow-hidden rounded-lg border border-ink-800">
        <table className="w-full text-xs">
          <thead className="bg-ink-800/60 text-slate-400">
            <tr>
              <th className="text-left px-3 py-2 font-medium">Instrument</th>
              <th className="text-left px-3 py-2 font-medium">Signal</th>
              <th className="text-left px-3 py-2 font-medium">Confidence</th>
              <th className="text-left px-3 py-2 font-medium">Model</th>
              <th className="text-right px-3 py-2 font-medium">Updated</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((inst) => {
              const s = signals[inst];
              if (!s) {
                return (
                  <tr key={inst} className="border-t border-ink-800">
                    <td className="px-3 py-2 font-mono text-slate-300">{inst}</td>
                    <td className="px-3 py-2 text-slate-500 italic">warming up</td>
                    <td className="px-3 py-2 text-slate-600 font-mono">—</td>
                    <td className="px-3 py-2 text-slate-600 font-mono">—</td>
                    <td className="px-3 py-2 text-slate-600 text-right font-mono">—</td>
                  </tr>
                );
              }
              const confPct = (s.confidence * 100).toFixed(0);
              const updated = new Date(s.time).toLocaleTimeString();
              return (
                <tr key={inst} className="border-t border-ink-800">
                  <td className="px-3 py-2 font-mono text-slate-300">{inst}</td>
                  <td className="px-3 py-2">
                    <span
                      className={`inline-block px-2 py-0.5 text-[10px] tracking-widest rounded border font-mono ${dirClass[s.direction]}`}
                    >
                      {dirLabel[s.direction]}
                    </span>
                  </td>
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      <div className="relative w-24 h-1.5 bg-ink-800 rounded-full overflow-hidden">
                        <div
                          className="absolute inset-y-0 left-0 bg-accent-500"
                          style={{ width: `${Math.max(33, Math.min(100, s.confidence * 100))}%` }}
                        />
                      </div>
                      <div className="font-mono text-slate-300">{confPct}%</div>
                    </div>
                  </td>
                  <td className="px-3 py-2 font-mono text-slate-500 truncate max-w-[160px]">
                    {s.model_id}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-slate-500">{updated}</td>
                </tr>
              );
            })}
            {sorted.length === 0 && (
              <tr>
                <td className="px-3 py-3 text-slate-500" colSpan={5}>
                  No signals yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

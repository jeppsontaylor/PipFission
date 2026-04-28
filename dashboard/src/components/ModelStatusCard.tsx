import { Card } from "./Card";
import type { ModelFitness } from "../types";

const TRAIN_AFTER = 1000;
const RETRAIN_EVERY = 100;

export interface ModelStatusCardProps {
  /// instrument -> sample count seen so far.
  featureCounts: Record<string, number>;
  /// instrument -> latest fitness snapshot.
  fitness: Record<string, ModelFitness>;
  instruments: string[];
}

export function ModelStatusCard({ featureCounts, fitness, instruments }: ModelStatusCardProps) {
  const sortedInstruments = [...instruments].sort();
  const trained = sortedInstruments.filter((i) => fitness[i]).length;
  const total = sortedInstruments.length || 1;

  return (
    <Card title={`ML Status — ${trained}/${total} trained`}>
      <div className="space-y-2">
        {sortedInstruments.map((inst) => {
          const f = fitness[inst];
          const seen = featureCounts[inst] ?? 0;
          if (!f) {
            // Warmup phase — show progress to first train.
            const pct = Math.min(100, Math.floor((seen / TRAIN_AFTER) * 100));
            return (
              <div key={inst} className="flex items-center gap-3 text-xs">
                <div className="w-20 font-mono text-slate-400">{inst}</div>
                <div className="flex-1 relative h-3 rounded-full bg-ink-800 overflow-hidden">
                  <div
                    className="absolute inset-y-0 left-0 bg-accent-500/70"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <div className="w-32 text-right font-mono text-slate-400">
                  {seen}/{TRAIN_AFTER}
                </div>
              </div>
            );
          }
          // Trained — show OOS metrics + retrain progress bar.
          const sinceLast = Math.max(0, seen - f.samples_seen);
          const retrainPct = Math.min(100, Math.floor((sinceLast / RETRAIN_EVERY) * 100));
          const acc = (f.oos.accuracy * 100).toFixed(1);
          const ll = f.oos.log_loss.toFixed(3);
          const sh = f.oos.sharpe.toFixed(2);
          const accClass =
            f.oos.accuracy >= 0.55
              ? "text-bull-400"
              : f.oos.accuracy >= 0.5
                ? "text-slate-200"
                : "text-bear-400";
          return (
            <div
              key={inst}
              className="grid grid-cols-12 gap-2 text-xs items-center border-t border-ink-800 pt-2 first:border-t-0 first:pt-0"
            >
              <div className="col-span-2 font-mono text-slate-300">{inst}</div>
              <div className="col-span-1 font-mono text-slate-500">v{f.model_version}</div>
              <div className={`col-span-2 font-mono ${accClass}`}>
                acc {acc}%
              </div>
              <div className="col-span-2 font-mono text-slate-400">ll {ll}</div>
              <div className="col-span-2 font-mono text-slate-400">sh {sh}</div>
              <div className="col-span-3">
                <div className="relative h-2 rounded-full bg-ink-800 overflow-hidden">
                  <div
                    className="absolute inset-y-0 left-0 bg-emerald-500/60"
                    style={{ width: `${retrainPct}%` }}
                  />
                </div>
                <div className="text-[10px] font-mono text-slate-500 mt-0.5">
                  retrain in {Math.max(0, RETRAIN_EVERY - sinceLast)} ticks
                </div>
              </div>
            </div>
          );
        })}
        {sortedInstruments.length === 0 && (
          <div className="text-xs text-slate-500">Waiting for instrument list…</div>
        )}
      </div>
    </Card>
  );
}

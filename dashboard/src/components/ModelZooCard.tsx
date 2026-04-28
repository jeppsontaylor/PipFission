// Per-instrument model-zoo comparison.
//
// Shows every candidate that train.side evaluated on the most recent
// training run, sorted by OOS log loss (lower = better — calibrated
// probabilities). The winner gets a badge; runners-up are visible so
// the operator can see the gap and decide whether to enable / disable
// candidates in the orchestrator.
//
// Reads /api/model/candidates?instrument=, polled at 30s — the
// underlying table only changes when a retrain finishes (~17min cadence
// at 100-bar threshold), so 30s is amply fast.

import { useState } from "react";

import { Card } from "./Card";
import {
  useModelCandidates,
  type ModelCandidateRow,
} from "../mlApi";

interface Props {
  instruments: string[];
  pollIntervalMs?: number;
}

export function ModelZooCard({ instruments, pollIntervalMs }: Props) {
  const [selected, setSelected] = useState<string>(instruments[0] ?? "");
  const inst = instruments.includes(selected) ? selected : instruments[0] ?? "";

  const { rows, fetchedAt, loading, error } = useModelCandidates(
    inst,
    pollIntervalMs,
  );

  return (
    <Card
      title="Model Zoo · Candidate Comparison"
      right={
        <span className="text-xs text-slate-500">
          {fetchedAt
            ? `${rows.length} candidates · updated ${formatAge(Date.now() - fetchedAt)} ago`
            : loading
              ? "loading…"
              : "no run yet"}
        </span>
      }
    >
      <div className="space-y-3">
        {/* Instrument tabs */}
        {instruments.length > 1 && (
          <div className="flex items-center gap-1 overflow-x-auto pb-1">
            {instruments.map((i) => (
              <button
                key={i}
                type="button"
                onClick={() => setSelected(i)}
                className={`px-3 py-1 rounded-md border font-mono text-xs whitespace-nowrap ${
                  i === inst
                    ? "border-ink-500 bg-ink-700/80 text-slate-100"
                    : "border-ink-700/60 bg-ink-800/40 text-slate-400 hover:bg-ink-700/60"
                }`}
              >
                {i.replace("_", "/")}
              </button>
            ))}
          </div>
        )}

        {rows.length === 0 ? (
          <div className="text-xs text-slate-500 italic">
            {loading
              ? "Polling…"
              : `No candidate scores yet for ${inst}. Wait for a retrain to land — every train.side run evaluates the full zoo.`}
          </div>
        ) : (
          <CandidatesTable rows={rows} />
        )}

        {error && (
          <div className="text-xs text-amber-300/80 break-all">REST: {error}</div>
        )}
      </div>
    </Card>
  );
}

function CandidatesTable({ rows }: { rows: ModelCandidateRow[] }) {
  // Find the spread: how much worse is the worst vs the winner? Used
  // to scale the bar widths so the visual contrast is meaningful.
  const minLoss = Math.min(...rows.map((r) => r.oos_log_loss));
  const maxLoss = Math.max(...rows.map((r) => r.oos_log_loss));
  const range = Math.max(1e-9, maxLoss - minLoss);

  return (
    <div className="border border-ink-700/70 rounded overflow-hidden">
      <table className="w-full text-xs">
        <thead className="bg-ink-800/70 text-slate-400 uppercase text-[10px] tracking-wide">
          <tr>
            <th className="text-left px-2 py-1">Model</th>
            <th className="text-right px-2 py-1">OOS log loss</th>
            <th className="text-left px-2 py-1 w-32">Δ from winner</th>
            <th className="text-right px-2 py-1">OOS AUC</th>
            <th className="text-right px-2 py-1">Brier</th>
            <th className="text-right px-2 py-1">Bal-acc</th>
            <th className="text-right px-2 py-1">n OOF</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const delta = r.oos_log_loss - minLoss;
            const widthPct = (delta / range) * 100;
            const winnerCls = r.is_winner
              ? "bg-emerald-950/30"
              : "";
            return (
              <tr
                key={`${r.run_id}-${r.spec_name}`}
                className={`border-t border-ink-700/50 ${winnerCls}`}
              >
                <td className="px-2 py-1 font-mono">
                  <span className={r.is_winner ? "text-emerald-200" : "text-slate-200"}>
                    {prettySpec(r.spec_name)}
                  </span>
                  {r.is_winner && (
                    <span className="ml-2 text-[9px] uppercase tracking-wider px-1 py-0.5 rounded bg-emerald-900/50 border border-emerald-700/60 text-emerald-300">
                      winner
                    </span>
                  )}
                </td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-100">
                  {r.oos_log_loss.toFixed(4)}
                </td>
                <td className="px-2 py-1">
                  {r.is_winner ? (
                    <span className="text-emerald-300/80 text-[10px]">— best</span>
                  ) : (
                    <span className="flex items-center gap-2">
                      <span
                        className="block h-1.5 rounded bg-rose-400/60"
                        style={{ width: `${widthPct}%`, minWidth: "2px" }}
                        aria-hidden="true"
                      />
                      <span className="text-rose-300/80 text-[10px] tabular-nums">
                        +{delta.toFixed(3)}
                      </span>
                    </span>
                  )}
                </td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-300">
                  {r.oos_auc.toFixed(3)}
                </td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-400">
                  {r.oos_brier.toFixed(3)}
                </td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-400">
                  {(r.oos_balanced_acc * 100).toFixed(1)}%
                </td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-500">
                  {r.n_oof}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function prettySpec(name: string): string {
  switch (name) {
    case "lgbm":
      return "LightGBM";
    case "xgb":
      return "XGBoost";
    case "catboost":
      return "CatBoost";
    case "logreg":
      return "LogReg";
    case "extratrees":
      return "ExtraTrees";
    default:
      return name;
  }
}

function formatAge(ms: number): string {
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s`;
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m`;
  if (ms < 86_400_000) return `${Math.floor(ms / 3_600_000)}h`;
  return `${Math.floor(ms / 86_400_000)}d`;
}

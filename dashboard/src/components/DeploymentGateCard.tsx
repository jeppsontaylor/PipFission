// Per-instrument deployment-gate viewer.
//
// Shows the most-recent gate decision: did the would-be champion
// meet every performance floor, or was it blocked from going live?
// For each floor (OOS AUC / OOS log loss / OOS balanced acc /
// fine-tune Sortino / fine-tune max DD), the card shows the actual
// vs the threshold with a colour bar so the operator can see at a
// glance how close each metric was to its bar.
//
// Reads /api/model/deployment-gate?instrument= polled at 30s — the
// underlying table only updates when a retrain completes.

import { useState } from "react";

import { Card } from "./Card";
import { useDeploymentGate, type ModelDeploymentGateRow } from "../mlApi";
import {
  evaluateFloors,
  splitBlockedReasons,
  type GateFloorResult,
} from "../deploymentGate";

interface Props {
  instruments: string[];
  pollIntervalMs?: number;
}

export function DeploymentGateCard({ instruments, pollIntervalMs }: Props) {
  const [selected, setSelected] = useState<string>(instruments[0] ?? "");
  const inst = instruments.includes(selected) ? selected : instruments[0] ?? "";
  const { row, fetchedAt, loading, error } = useDeploymentGate(inst, pollIntervalMs);

  return (
    <Card
      title="Deployment Gate · Quality Floors"
      right={
        <span className="text-xs text-slate-500">
          {fetchedAt
            ? `updated ${formatAge(Date.now() - fetchedAt)} ago`
            : loading
              ? "loading…"
              : "no run yet"}
        </span>
      }
    >
      <div className="space-y-3">
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

        {row === null ? (
          <div className="text-xs text-slate-500 italic">
            {loading
              ? "Polling…"
              : `No deployment-gate evaluation yet for ${inst}. The next retrain will populate this card.`}
          </div>
        ) : (
          <GateBody row={row} />
        )}

        {error && (
          <div className="text-xs text-amber-300/80 break-all">REST: {error}</div>
        )}
      </div>
    </Card>
  );
}

function GateBody({ row }: { row: ModelDeploymentGateRow }) {
  const floors = evaluateFloors(row);
  const blocked = splitBlockedReasons(row.blocked_reasons);

  return (
    <div className="space-y-3">
      {/* Headline pass/fail strip */}
      <div
        className={`rounded-md border px-3 py-2 text-xs ${
          row.passed_gate
            ? "border-emerald-700/60 bg-emerald-950/40 text-emerald-100"
            : "border-rose-700/60 bg-rose-950/40 text-rose-100"
        }`}
      >
        <div className="flex items-center gap-2">
          <span className="font-semibold">
            {row.passed_gate ? "PASSED" : "BLOCKED"}
          </span>
          <span className="text-[11px] opacity-80">model</span>
          <span className="font-mono opacity-90">
            {row.model_id.length > 28
              ? `${row.model_id.slice(0, 12)}…${row.model_id.slice(-12)}`
              : row.model_id}
          </span>
          {!row.passed_gate && (
            <span className="ml-auto text-[11px] opacity-90">
              {blocked.length} reason{blocked.length === 1 ? "" : "s"}
            </span>
          )}
        </div>
        {!row.passed_gate && blocked.length > 0 && (
          <ul className="mt-1.5 list-disc list-inside text-[11px] opacity-90 space-y-0.5">
            {blocked.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        )}
      </div>

      {/* Per-floor table */}
      <FloorsTable floors={floors} />
    </div>
  );
}

function FloorsTable({ floors }: { floors: GateFloorResult[] }) {
  return (
    <div className="border border-ink-700/70 rounded overflow-hidden">
      <table className="w-full text-xs">
        <thead className="bg-ink-800/70 text-slate-400 uppercase text-[10px] tracking-wide">
          <tr>
            <th className="text-left px-2 py-1">Metric</th>
            <th className="text-right px-2 py-1">Actual</th>
            <th className="text-right px-2 py-1">Floor</th>
            <th className="text-left px-2 py-1 w-32">Margin</th>
            <th className="text-right px-2 py-1">Status</th>
          </tr>
        </thead>
        <tbody>
          {floors.map((f) => {
            const sign = f.direction === "ge" ? "≥" : "≤";
            const widthPct = Math.min(100, Math.max(2, Math.abs(f.fractionOver) * 100));
            return (
              <tr key={f.label} className="border-t border-ink-700/50">
                <td className="px-2 py-1 text-slate-200">{f.label}</td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-100">
                  {f.actual.toFixed(3)}
                </td>
                <td
                  className="px-2 py-1 text-right tabular-nums text-slate-400"
                  title={`requires ${sign} ${f.floor}`}
                >
                  {sign} {f.floor.toFixed(3)}
                </td>
                <td className="px-2 py-1">
                  <span
                    className={`block h-1.5 rounded ${
                      f.passed ? "bg-emerald-400/70" : "bg-rose-400/70"
                    }`}
                    style={{ width: `${widthPct}%` }}
                    aria-hidden="true"
                  />
                </td>
                <td className="px-2 py-1 text-right">
                  <span
                    className={`text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded border font-mono ${
                      f.passed
                        ? "bg-emerald-900/40 border-emerald-700/60 text-emerald-200"
                        : "bg-rose-900/40 border-rose-700/60 text-rose-200"
                    }`}
                  >
                    {f.passed ? "pass" : "fail"}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function formatAge(ms: number): string {
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s`;
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m`;
  if (ms < 86_400_000) return `${Math.floor(ms / 3_600_000)}h`;
  return `${Math.floor(ms / 86_400_000)}d`;
}

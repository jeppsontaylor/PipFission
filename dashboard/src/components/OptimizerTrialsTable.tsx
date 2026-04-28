// Pareto-trial leaderboard for the trader optimizer. Polls
// /api/optimizer/trials and renders a ranked table — top-K by score,
// with Pareto-1 trials highlighted. The actual ranking, summary and
// param-snippet logic lives in `optimizerTrials.ts` so it can be unit
// tested without rendering.
//
// This finishes M11's "optimizer trial leaderboard" deliverable.

import { useMemo } from "react";

import { Card } from "./Card";
import { useOptimizerTrials } from "../mlApi";
import {
  paretoLabel,
  paretoTone,
  parseParamsJson,
  summarizeTrials,
  topNByScore,
} from "../optimizerTrials";

interface Props {
  /** How many trials to surface in the table. */
  topN?: number;
  /** Optional study filter — when omitted, latest across all studies. */
  study?: string;
  /** Polling cadence (ms). Default 30s. */
  pollIntervalMs?: number;
}

const NUM = (v: number | null | undefined, digits = 3) =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(digits);

const BP = (v: number | null | undefined) =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : `${v.toFixed(0)} bp`;

export function OptimizerTrialsTable({ topN = 25, study, pollIntervalMs }: Props) {
  const { rows, fetchedAt, error, loading } = useOptimizerTrials({
    limit: Math.max(topN * 2, 50),
    study,
    pollIntervalMs,
  });

  const summary = useMemo(() => summarizeTrials(rows), [rows]);
  const top = useMemo(() => topNByScore(rows, topN), [rows, topN]);

  return (
    <Card
      title="Optimizer Trials"
      right={
        <span className="text-xs text-slate-500">
          {fetchedAt
            ? `updated ${formatAge(Date.now() - fetchedAt)} ago`
            : loading
              ? "loading…"
              : "—"}
        </span>
      }
    >
      <div className="space-y-3">
        <SummaryRow summary={summary} />
        {top.length === 0 ? (
          <div className="text-xs text-slate-500 italic">
            {loading
              ? "Polling…"
              : "No trials yet — kick off a fine-tune (research/cli finetune) to populate."}
          </div>
        ) : (
          <TrialsTable rows={top} />
        )}
        {error && (
          <div className="text-xs text-amber-300/80 break-all">REST: {error}</div>
        )}
      </div>
    </Card>
  );
}

function SummaryRow({ summary }: { summary: ReturnType<typeof summarizeTrials> }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
      <Cell label="Trials" value={summary.count.toString()} sub="recent window" />
      <Cell
        label="Pareto-1"
        value={summary.paretoCount.toString()}
        sub="optimal frontier"
      />
      <Cell label="Best score" value={NUM(summary.bestScore, 3)} sub="composite" />
      <Cell label="Best Sortino" value={NUM(summary.bestSortino, 2)} sub="fine-tune" />
      <Cell label="Best max DD" value={BP(summary.bestDrawdownBp)} sub="fine-tune" />
    </div>
  );
}

function Cell({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div>
      <div className="text-slate-500 uppercase tracking-wide text-[10px]">{label}</div>
      <div className="text-slate-100 text-base tabular-nums">{value}</div>
      <div className="text-slate-500 text-[10px]">{sub}</div>
    </div>
  );
}

function TrialsTable({ rows }: { rows: ReturnType<typeof topNByScore> }) {
  return (
    <div className="border border-ink-700/70 rounded overflow-hidden">
      <table className="w-full text-xs">
        <thead className="bg-ink-800/70 text-slate-400 uppercase text-[10px] tracking-wide">
          <tr>
            <th className="text-left px-2 py-1">Rank</th>
            <th className="text-left px-2 py-1">Trial</th>
            <th className="text-left px-2 py-1">Study</th>
            <th className="text-right px-2 py-1">Score</th>
            <th className="text-right px-2 py-1">Sortino</th>
            <th className="text-right px-2 py-1">Max DD</th>
            <th className="text-right px-2 py-1">Turnover</th>
            <th className="text-left px-2 py-1">Params</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const params = parseParamsJson(r.params_json);
            const tipParts: string[] = [];
            if (params) {
              for (const [k, v] of Object.entries(params.raw)) {
                tipParts.push(`${k}=${formatTooltipValue(v)}`);
              }
            }
            return (
              <tr
                key={`${r.study_id}:${r.trial_id}`}
                className="border-t border-ink-700/50"
              >
                <td className="px-2 py-1">
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded border font-mono ${paretoTone(r.pareto_rank)}`}
                  >
                    {paretoLabel(r.pareto_rank)}
                  </span>
                </td>
                <td className="px-2 py-1 font-mono text-slate-300">#{r.trial_id}</td>
                <td className="px-2 py-1 font-mono text-slate-500" title={r.study_id}>
                  {shortStudy(r.study_id)}
                </td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-100">
                  {NUM(r.score, 3)}
                </td>
                <td className="px-2 py-1 text-right tabular-nums">{NUM(r.sortino, 2)}</td>
                <td className="px-2 py-1 text-right tabular-nums text-rose-200/80">
                  {BP(r.max_dd_bp)}
                </td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-400">
                  {NUM(r.turnover, 2)}
                </td>
                <td
                  className="px-2 py-1 font-mono text-slate-400 text-[11px]"
                  title={tipParts.join("\n") || "—"}
                >
                  {params?.snippet || "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function formatTooltipValue(v: unknown): string {
  if (typeof v === "number" && Number.isFinite(v)) {
    if (Number.isInteger(v)) return v.toString();
    return v.toFixed(4);
  }
  if (v === null || v === undefined) return "—";
  return String(v);
}

function shortStudy(id: string): string {
  if (id.length <= 18) return id;
  return `${id.slice(0, 8)}…${id.slice(-6)}`;
}

function formatAge(ms: number): string {
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s`;
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m`;
  return `${Math.floor(ms / 3_600_000)}h`;
}

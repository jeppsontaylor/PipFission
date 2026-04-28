// Operator-visibility panel for the Python research pipeline. Polls
// /api/pipeline/runs and renders:
//   * Status row — counts of success / failed / running, and the
//     newest in-flight command if one is running.
//   * Per-command "last success" chips — when did each retrain stage
//     last complete successfully.
//   * Recent runs table — newest first, with status, elapsed, args.
//
// Aggregations + formatting all live in `pipelineRuns.ts` so this
// component stays a thin presentation shell.

import { useEffect, useMemo, useRef, useState } from "react";

import { Card } from "./Card";
import {
  fetchLastCompletedPipeline,
  triggerPipeline,
  useAutoRetrain,
  usePipelineLog,
  usePipelineRuns,
  usePipelineStatus,
  type PipelineFlight,
  type PipelineRunRow,
} from "../mlApi";
import {
  autoRetrainProgress,
  commandLabel,
  formatElapsed,
  liveElapsed,
  parseArgs,
  pipelineButtonState,
  statusLabel,
  statusTone,
  summarizePipelineRuns,
} from "../pipelineRuns";

interface Props {
  /** How many runs to fetch / show. Server caps at 10k. */
  limit?: number;
  /** Polling cadence (ms). Default 5s. */
  pollIntervalMs?: number;
  /** Optional list of instruments to populate the trigger dropdown. */
  instruments?: string[];
}

const KNOWN_COMMANDS = [
  "label",
  "train.side",
  "finetune",
  "lockbox",
  "export.champion",
];

export function PipelineRunsCard({ limit = 25, pollIntervalMs, instruments = [] }: Props) {
  const { rows, fetchedAt, loading, error } = usePipelineRuns({ limit, pollIntervalMs });
  const triggerStatus = usePipelineStatus();
  const auto = useAutoRetrain();
  const summary = useMemo(() => summarizePipelineRuns(rows), [rows]);
  const recent = useMemo(() => rows.slice(0, 15), [rows]);

  const [selectedInstrument, setSelectedInstrument] = useState<string>(
    instruments[0] ?? "",
  );
  const [submitting, setSubmitting] = useState(false);
  const [triggerError, setTriggerError] = useState<string | null>(null);
  const [lastTriggered, setLastTriggered] = useState<string | null>(null);

  // Log viewer state. When `viewRunId` is set, we render an inline
  // <pre> below the controls with the tail of that run's log file.
  // Polls live (every 2s) only when the run is actually in flight.
  const [viewRunId, setViewRunId] = useState<string | null>(null);
  const isViewedRunInFlight =
    triggerStatus.current?.run_id === viewRunId && viewRunId !== null;

  // Last completed flight for "View log" after the run finishes.
  // Server caches it so we don't lose it the moment the slot clears.
  const [lastCompleted, setLastCompleted] = useState<PipelineFlight | null>(null);
  useEffect(() => {
    let stopped = false;
    const ctrl = new AbortController();
    async function step() {
      try {
        const r = await fetchLastCompletedPipeline(ctrl.signal);
        if (!stopped) setLastCompleted(r);
      } catch {
        /* polling, never bubble */
      }
    }
    void step();
    const id = setInterval(step, 5_000);
    return () => {
      stopped = true;
      ctrl.abort();
      clearInterval(id);
    };
  }, []);

  // Keep the dropdown valid if the parent's instrument list changes.
  if (
    instruments.length > 0 &&
    selectedInstrument !== "" &&
    !instruments.includes(selectedInstrument)
  ) {
    setSelectedInstrument(instruments[0]);
  }
  if (instruments.length > 0 && selectedInstrument === "") {
    setSelectedInstrument(instruments[0]);
  }

  const button = pipelineButtonState({
    enabled: triggerStatus.enabled,
    inflightInstrument: triggerStatus.current?.instrument ?? null,
    selectedInstrument,
    submitting,
  });

  async function onRunClick() {
    if (button.disabled) return;
    setSubmitting(true);
    setTriggerError(null);
    try {
      const r = await triggerPipeline({ instrument: selectedInstrument });
      setLastTriggered(r.run_id);
    } catch (err) {
      setTriggerError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card
      title="Retrain Pipeline"
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
        {/* Trigger row — instrument selector + Run button. Disabled
            when the server gate is off OR a run is in flight. */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <select
            value={selectedInstrument}
            onChange={(e) => setSelectedInstrument(e.target.value)}
            className="bg-ink-800 border border-ink-700 rounded px-2 py-1 font-mono text-slate-200"
            disabled={instruments.length === 0}
            title="Pick the instrument to retrain"
          >
            {instruments.length === 0 && <option value="">no instruments</option>}
            {instruments.map((i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={onRunClick}
            disabled={button.disabled}
            title={button.reason}
            className={`px-3 py-1 rounded border font-mono ${
              button.disabled
                ? "bg-slate-800/60 border-slate-700 text-slate-500 cursor-not-allowed"
                : "bg-emerald-900/40 border-emerald-700/60 text-emerald-100 hover:bg-emerald-900/60"
            }`}
          >
            {button.label}
          </button>
          {triggerStatus.current && (
            <span className="text-[11px] text-sky-200 flex items-center gap-1">
              in flight: {triggerStatus.current.instrument} (
              {formatElapsed(Date.now() - triggerStatus.current.started_ms)})
              <button
                type="button"
                onClick={() => setViewRunId(triggerStatus.current!.run_id)}
                className="ml-1 underline decoration-dotted hover:text-sky-100"
              >
                view log
              </button>
            </span>
          )}
          {!triggerStatus.current && lastCompleted && (
            <span className="text-[11px] text-slate-400 flex items-center gap-1">
              last run: {lastCompleted.instrument}{" "}
              {lastCompleted.exit_status ?? ""}
              <button
                type="button"
                onClick={() => setViewRunId(lastCompleted.run_id)}
                className="ml-1 underline decoration-dotted hover:text-slate-200"
              >
                view log
              </button>
            </span>
          )}
          {lastTriggered && !triggerStatus.current && !lastCompleted && (
            <span className="text-[11px] text-slate-500">
              last spawned: {lastTriggered.slice(0, 16)}…
            </span>
          )}
          {triggerError && (
            <span className="text-[11px] text-rose-300 break-all">{triggerError}</span>
          )}
        </div>

        {viewRunId && (
          <LogViewer
            runId={viewRunId}
            live={isViewedRunInFlight}
            onClose={() => setViewRunId(null)}
          />
        )}

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <Cell label="Runs" value={summary.total.toString()} sub="recent" />
          <Cell
            label="Success"
            value={summary.success.toString()}
            sub="completed OK"
            tone="text-emerald-200"
          />
          <Cell
            label="Failed"
            value={summary.failed.toString()}
            sub="exited non-zero"
            tone={summary.failed > 0 ? "text-rose-200" : "text-slate-300"}
          />
          <Cell
            label="Running"
            value={summary.running.toString()}
            sub={summary.inflight ? commandLabel(summary.inflight.command) : "idle"}
            tone={summary.running > 0 ? "text-sky-200" : "text-slate-300"}
          />
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {KNOWN_COMMANDS.map((cmd) => {
            const ts = summary.lastSuccessByCommand[cmd];
            return (
              <CommandPill
                key={cmd}
                command={cmd}
                lastSuccessTs={ts ?? null}
              />
            );
          })}
        </div>

        {/* Auto-retrain row — shows the per-instrument bar counter so
            the operator can see how close each is to the threshold. */}
        {auto.status && auto.status.instruments.length > 0 && (
          <AutoRetrainRow
            enabled={auto.status.enabled}
            threshold={auto.status.bars_threshold}
            instruments={auto.status.instruments}
          />
        )}

        {recent.length === 0 ? (
          <div className="text-xs text-slate-500 italic">
            {loading
              ? "Polling…"
              : "No pipeline runs yet — invoke `python -m research <cmd>` to populate."}
          </div>
        ) : (
          <RunsTable rows={recent} />
        )}

        {error && (
          <div className="text-xs text-amber-300/80 break-all">REST: {error}</div>
        )}
      </div>
    </Card>
  );
}

function LogViewer({
  runId,
  live,
  onClose,
}: {
  runId: string;
  live: boolean;
  onClose: () => void;
}) {
  const log = usePipelineLog({ runId, live });
  const preRef = useRef<HTMLPreElement | null>(null);

  // Auto-scroll to bottom on new content while live.
  useEffect(() => {
    if (live && preRef.current) {
      preRef.current.scrollTop = preRef.current.scrollHeight;
    }
  }, [log.text, live]);

  return (
    <div className="border border-ink-700/70 rounded p-2 bg-ink-950/40">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-slate-400 mb-1">
        <span>
          Log · <span className="font-mono text-slate-300">{runId.slice(0, 24)}</span>
          {live && (
            <span className="ml-2 px-1.5 py-0.5 rounded bg-sky-900/50 border border-sky-700/60 text-sky-200 normal-case">
              live
            </span>
          )}
        </span>
        <button
          type="button"
          onClick={onClose}
          className="hover:text-slate-200 normal-case"
        >
          close ✕
        </button>
      </div>
      {log.error ? (
        <div className="text-xs text-amber-300/80">
          {log.error === "log not found"
            ? "No log yet — the pipeline hasn't written to its file."
            : `log fetch error: ${log.error}`}
        </div>
      ) : log.text === "" ? (
        <div className="text-xs text-slate-500 italic">
          {log.loading ? "loading…" : "empty"}
        </div>
      ) : (
        <pre
          ref={preRef}
          className="text-[11px] font-mono text-slate-300 overflow-auto max-h-64 whitespace-pre-wrap break-words"
        >
          {log.text}
        </pre>
      )}
    </div>
  );
}

function AutoRetrainRow({
  enabled,
  threshold,
  instruments,
}: {
  enabled: boolean;
  threshold: number;
  instruments: import("../mlApi").AutoRetrainInstrumentStatus[];
}) {
  return (
    <div className="border border-ink-700/70 rounded p-2 space-y-1.5">
      <div className="flex items-center gap-2 text-[10px] uppercase tracking-wide text-slate-500">
        <span>Auto-retrain</span>
        <span
          className={`px-1.5 py-0.5 rounded border font-mono ${
            enabled
              ? "bg-emerald-900/40 border-emerald-700/60 text-emerald-200"
              : "bg-slate-800/60 border-slate-700 text-slate-400"
          }`}
        >
          {enabled ? "ON" : "OFF"}
        </span>
        <span className="text-slate-500">threshold {threshold} bars</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {instruments.map((s) => {
          const p = autoRetrainProgress(s.bars_since_last_fire, threshold);
          return (
            <div
              key={s.instrument}
              className="border border-ink-700/60 rounded px-2 py-1 text-[11px] flex items-center gap-2"
              title={
                s.last_skip_reason
                  ? `last skip: ${s.last_skip_reason}`
                  : s.last_fired_ms
                    ? `last fired: ${new Date(s.last_fired_ms).toLocaleString()}`
                    : "never fired"
              }
            >
              <span className="font-mono text-slate-200">{s.instrument}</span>
              <span className={`tabular-nums ${p.tone}`}>{p.ratio}</span>
              <span
                className="inline-block h-1 w-12 rounded-full overflow-hidden bg-ink-800"
                aria-hidden="true"
              >
                <span
                  className={`block h-full ${
                    p.fraction >= 1
                      ? "bg-rose-400"
                      : p.fraction >= 0.8
                        ? "bg-amber-400"
                        : "bg-emerald-400"
                  }`}
                  style={{ width: `${Math.min(100, p.fraction * 100)}%` }}
                />
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Cell({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub: string;
  tone?: string;
}) {
  return (
    <div>
      <div className="text-slate-500 uppercase tracking-wide text-[10px]">{label}</div>
      <div className={`text-base tabular-nums ${tone ?? "text-slate-100"}`}>{value}</div>
      <div className="text-slate-500 text-[10px]">{sub}</div>
    </div>
  );
}

function CommandPill({
  command,
  lastSuccessTs,
}: {
  command: string;
  lastSuccessTs: number | null;
}) {
  const tone =
    lastSuccessTs !== null
      ? "bg-emerald-900/30 border-emerald-700/40 text-emerald-200"
      : "bg-slate-800/60 border-slate-700 text-slate-400";
  const sub = lastSuccessTs !== null ? formatAge(Date.now() - lastSuccessTs) + " ago" : "never";
  return (
    <span
      className={`text-[11px] px-2 py-1 rounded border ${tone}`}
      title={lastSuccessTs !== null ? new Date(lastSuccessTs).toLocaleString() : "no successful run yet"}
    >
      <span className="font-mono">{commandLabel(command)}</span>
      <span className="ml-2 opacity-75">{sub}</span>
    </span>
  );
}

function RunsTable({ rows }: { rows: PipelineRunRow[] }) {
  return (
    <div className="border border-ink-700/70 rounded overflow-hidden">
      <table className="w-full text-xs">
        <thead className="bg-ink-800/70 text-slate-400 uppercase text-[10px] tracking-wide">
          <tr>
            <th className="text-left px-2 py-1">Status</th>
            <th className="text-left px-2 py-1">Command</th>
            <th className="text-left px-2 py-1">Inst</th>
            <th className="text-left px-2 py-1">Started</th>
            <th className="text-right px-2 py-1">Elapsed</th>
            <th className="text-left px-2 py-1">Args</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const args = parseArgs(r.args_json);
            const argSummary = formatArgSummary(args);
            return (
              <tr key={r.run_id} className="border-t border-ink-700/50">
                <td className="px-2 py-1">
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded border font-mono ${statusTone(r.status)}`}
                  >
                    {statusLabel(r.status)}
                  </span>
                </td>
                <td className="px-2 py-1 font-mono text-slate-200">
                  {commandLabel(r.command)}
                </td>
                <td className="px-2 py-1 font-mono text-slate-400">
                  {r.instrument ?? "—"}
                </td>
                <td className="px-2 py-1 font-mono text-slate-500">
                  {formatTime(r.ts_started_ms)}
                </td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-300">
                  {formatElapsed(liveElapsed(r))}
                </td>
                <td
                  className="px-2 py-1 font-mono text-slate-400 text-[11px] truncate max-w-[260px]"
                  title={JSON.stringify(args, null, 2)}
                >
                  {r.error_msg ? (
                    <span className="text-rose-300">{truncate(r.error_msg, 60)}</span>
                  ) : (
                    argSummary
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function formatArgSummary(args: Record<string, unknown>): string {
  const keys = ["instrument", "model_id", "params_id", "n_bars", "n_trials", "n_fine_tune"];
  const parts: string[] = [];
  for (const k of keys) {
    const v = args[k];
    if (v === undefined || v === null) continue;
    parts.push(`${k}=${typeof v === "number" ? v : String(v).slice(0, 32)}`);
  }
  return parts.join(" ");
}

function formatTime(ms: number): string {
  return new Date(ms).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatAge(ms: number): string {
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s`;
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m`;
  if (ms < 86_400_000) return `${Math.floor(ms / 3_600_000)}h`;
  return `${Math.floor(ms / 86_400_000)}d`;
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return `${s.slice(0, n - 1)}…`;
}

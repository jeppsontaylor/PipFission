// Panel surfacing the ML pipeline's live state for one instrument.
// Stitches together:
//   * REST polling of /api/strategy/champion + /api/model/metrics +
//     /api/trader/metrics + /api/lockbox/result (via useMlPipeline)
//   * WS-fed live state from the store: champion status, latest
//     ChampionSignal, recent TraderDecision events
//
// Wraps a tab strip — one tab per instrument — so the operator can
// click between champions when several are deployed at once. The body
// (`MLPipelinePanelBody`) is unchanged; the tabs sit above it.

import { useEffect, useMemo, useState } from "react";
import { LineChart, Line, ResponsiveContainer, YAxis } from "recharts";

import { Card } from "./Card";
import { useMlPipeline } from "../mlApi";
import { actionLabel, reasonLabel, type DecisionPoint } from "../mlState";
import {
  pickSelectedInstrument,
  pickTabTone,
  tabToneClasses,
  type TabTone,
} from "../mlPanelTabs";
import type { State } from "../store";

interface Props {
  state: State;
  /** Instruments to expose as tabs. The first is selected by default. */
  instruments: string[];
}

interface BodyProps {
  state: State;
  instrument: string;
}

export function MLPipelinePanel({ state, instruments }: Props) {
  const [selected, setSelected] = useState<string>(instruments[0] ?? "");
  // Keep `selected` valid as the parent's instrument list changes
  // (e.g. user toggles the venue filter).
  useEffect(() => {
    const next = pickSelectedInstrument(instruments, selected);
    if (next !== selected) setSelected(next);
  }, [instruments, selected]);

  if (instruments.length === 0) {
    return null;
  }

  if (instruments.length === 1) {
    // Skip the tab strip entirely when there's only one instrument —
    // saves vertical space and matches the pre-tab layout.
    return <MLPipelinePanelBody state={state} instrument={instruments[0]} />;
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1 overflow-x-auto pb-1">
        {instruments.map((inst) => (
          <InstrumentTab
            key={inst}
            instrument={inst}
            selected={inst === selected}
            tone={pickTabTone(state.ml.championSignals[inst])}
            onClick={() => setSelected(inst)}
          />
        ))}
      </div>
      <MLPipelinePanelBody state={state} instrument={selected} />
    </div>
  );
}

function InstrumentTab({
  instrument,
  selected,
  tone,
  onClick,
}: {
  instrument: string;
  selected: boolean;
  tone: TabTone;
  onClick: () => void;
}) {
  // The tone helper returns one classname string with both the dot
  // colour AND the surrounding tab styling — we split here just to
  // place the dot before the label.
  const cls = tabToneClasses(tone, selected);
  // Pull the dot class (the `bg-…-400/700` token) for the inner span.
  const dotMatch = cls.match(/bg-(emerald|rose|slate)-(400|700)/);
  const dotClass = dotMatch ? dotMatch[0] : "bg-slate-700";
  return (
    <button
      type="button"
      onClick={onClick}
      className={`px-3 py-1 rounded-md border font-mono text-xs whitespace-nowrap flex items-center gap-2 transition-colors ${cls}`}
      title={`${instrument} · ${tone}`}
    >
      <span className={`inline-block h-2 w-2 rounded-full ${dotClass}`} aria-hidden="true" />
      {instrument.replace("_", "/")}
    </button>
  );
}


const SAFE_PCT = (v: number | null | undefined) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(1)}%`;

const SAFE_NUM = (v: number | null | undefined, digits = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits);

const SAFE_BP = (v: number | null | undefined) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${v.toFixed(0)} bp`;

function MLPipelinePanelBody({ state, instrument }: BodyProps) {
  const summary = useMlPipeline({ instrument });
  const wsChampion = state.ml.championStatus;

  // Prefer WS state over REST when both are present — WS updates within
  // milliseconds of a hot-swap, REST polls every 10s.
  const championModelId =
    wsChampion.model_id !== "uninitialised"
      ? wsChampion.model_id
      : summary.champion?.model_id ?? "uninitialised";
  const championKind =
    wsChampion.kind !== "uninitialised" ? wsChampion.kind : summary.champion?.kind ?? "uninitialised";

  const recentDecisions = state.ml.traderDecisionHistory[instrument] ?? [];
  const latestSignal = state.ml.championSignals[instrument] ?? null;
  const championHistory = state.ml.championHistory[instrument] ?? [];

  const lockbox = useMemo(() => parseLockbox(summary.lockbox?.summary_json), [
    summary.lockbox?.summary_json,
  ]);

  const sparkData = useMemo(
    () => championHistory.slice(-120).map((p) => ({ t: p.t, p_long: p.p_long })),
    [championHistory],
  );

  return (
    <Card
      title="ML Pipeline"
      right={
        <span className="text-xs text-slate-500">
          {summary.fetchedAt
            ? `updated ${formatAge(Date.now() - summary.fetchedAt)} ago`
            : summary.loading
              ? "loading…"
              : "—"}
        </span>
      }
    >
      <div className="space-y-3">
        {/* Row 1: champion + lockbox state */}
        <div className="flex items-center gap-3 flex-wrap">
          <ChampionPill modelId={championModelId} kind={championKind} />
          {wsChampion.last_error && (
            <span
              className="text-xs px-2 py-1 rounded bg-rose-900/40 border border-rose-700/60 text-rose-200"
              title={wsChampion.last_error}
            >
              load error
            </span>
          )}
          <LockboxPill lockbox={lockbox} sealed={summary.lockbox?.sealed ?? false} />
          {sparkData.length > 1 && (
            <div className="ml-auto h-8 w-32" aria-label="recent p_long">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sparkData}>
                  <YAxis hide domain={[0, 1]} />
                  <Line
                    type="monotone"
                    dataKey="p_long"
                    stroke="#60a5fa"
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Row 2: latest signal + metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <MetricBlock
            label="Latest p_long"
            value={latestSignal ? SAFE_PCT(latestSignal.p_long) : "—"}
            sub={latestSignal?.kind ?? "—"}
          />
          <MetricBlock
            label="OOS AUC"
            value={SAFE_NUM(summary.modelMetrics?.oos_auc, 3)}
            sub={`OOF n=${summary.modelMetrics?.n_oof ?? 0}`}
          />
          <MetricBlock
            label="Fine-tune Sortino"
            value={SAFE_NUM(summary.traderMetrics?.fine_tune_sortino, 2)}
            sub={`hits ${SAFE_PCT(summary.traderMetrics?.hit_rate)}`}
          />
          <MetricBlock
            label="Fine-tune max DD"
            value={SAFE_BP(summary.traderMetrics?.max_dd_bp)}
            sub={`${summary.traderMetrics?.n_trades ?? 0} trades`}
          />
        </div>

        {/* Row 3: trader decision feed */}
        <DecisionFeed decisions={recentDecisions.slice(-20).reverse()} />

        {summary.error && (
          <div className="text-xs text-amber-300/80 break-all">REST: {summary.error}</div>
        )}
      </div>
    </Card>
  );
}

function MetricBlock({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <div>
      <div className="text-slate-500 uppercase tracking-wide text-[10px]">{label}</div>
      <div className="text-slate-100 text-base tabular-nums">{value}</div>
      <div className="text-slate-500 text-[10px]">{sub}</div>
    </div>
  );
}

function ChampionPill({ modelId, kind }: { modelId: string; kind: string }) {
  const tone =
    kind === "onnx"
      ? "bg-emerald-900/40 border-emerald-700/60 text-emerald-200"
      : kind === "fallback"
        ? "bg-amber-900/40 border-amber-700/60 text-amber-200"
        : "bg-slate-800/60 border-slate-700 text-slate-300";
  const short =
    modelId === "uninitialised"
      ? "no champion"
      : modelId.length > 24
        ? `${modelId.slice(0, 8)}…${modelId.slice(-12)}`
        : modelId;
  return (
    <span
      className={`text-xs px-2 py-1 rounded border ${tone}`}
      title={modelId}
    >
      <span className="font-mono">{short}</span>
      <span className="ml-2 opacity-75">{kind}</span>
    </span>
  );
}

interface ParsedLockbox {
  pass: boolean | null;
  dsr: number | null;
  reasons: string[];
  n_trades: number | null;
}

function parseLockbox(json: string | null | undefined): ParsedLockbox | null {
  if (!json) return null;
  try {
    const obj = JSON.parse(json);
    return {
      pass: typeof obj.pass === "boolean" ? obj.pass : null,
      dsr: typeof obj.dsr === "number" ? obj.dsr : null,
      reasons: Array.isArray(obj.reasons) ? obj.reasons.map(String) : [],
      n_trades: typeof obj.n_trades === "number" ? obj.n_trades : null,
    };
  } catch {
    return null;
  }
}

function LockboxPill({
  lockbox,
  sealed,
}: {
  lockbox: ParsedLockbox | null;
  sealed: boolean;
}) {
  if (!lockbox) {
    return (
      <span className="text-xs px-2 py-1 rounded border bg-slate-800/60 border-slate-700 text-slate-400">
        lockbox: pending
      </span>
    );
  }
  const tone = lockbox.pass
    ? "bg-emerald-900/40 border-emerald-700/60 text-emerald-200"
    : "bg-rose-900/40 border-rose-700/60 text-rose-200";
  const label = lockbox.pass ? "lockbox: PASS" : "lockbox: FAIL";
  const tip = sealed ? "sealed" : "unsealed";
  return (
    <span
      className={`text-xs px-2 py-1 rounded border ${tone}`}
      title={`${tip} · DSR ${lockbox.dsr?.toFixed(2) ?? "—"} · n_trades ${lockbox.n_trades ?? "—"}${
        lockbox.reasons.length ? ` · ${lockbox.reasons.join("; ")}` : ""
      }`}
    >
      {label}
    </span>
  );
}

function DecisionFeed({ decisions }: { decisions: DecisionPoint[] }) {
  if (decisions.length === 0) {
    return (
      <div className="text-xs text-slate-500 italic">
        No trader decisions yet — waiting for live-trader to start emitting.
      </div>
    );
  }
  return (
    <div className="border border-ink-700/70 rounded overflow-hidden">
      <table className="w-full text-xs">
        <thead className="bg-ink-800/70 text-slate-400 uppercase text-[10px] tracking-wide">
          <tr>
            <th className="text-left px-2 py-1">Time</th>
            <th className="text-left px-2 py-1">Action</th>
            <th className="text-left px-2 py-1">Reason</th>
            <th className="text-right px-2 py-1">Price</th>
            <th className="text-right px-2 py-1">PnL</th>
          </tr>
        </thead>
        <tbody>
          {decisions.map((d) => (
            <tr
              key={`${d.t}-${d.bar_idx}`}
              className={`border-t border-ink-700/50 ${actionTone(d.action)}`}
            >
              <td className="px-2 py-1 font-mono text-slate-400">{formatTime(d.t)}</td>
              <td className="px-2 py-1">{actionLabel(d.action)}</td>
              <td className="px-2 py-1 text-slate-400">{reasonLabel(d.reason)}</td>
              <td className="px-2 py-1 text-right tabular-nums">{d.price.toFixed(5)}</td>
              <td className="px-2 py-1 text-right tabular-nums">
                {d.realized_r === null ? "—" : `${(d.realized_r * 100).toFixed(2)}%`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function actionTone(action: string): string {
  switch (action) {
    case "open_long":
      return "bg-emerald-950/40";
    case "open_short":
      return "bg-rose-950/40";
    case "close":
      return "bg-sky-950/40";
    default:
      return "";
  }
}

function formatTime(t: number): string {
  return new Date(t).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatAge(ms: number): string {
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s`;
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m`;
  return `${Math.floor(ms / 3_600_000)}h`;
}

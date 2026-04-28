// Per-trade forensic drawer.
//
// Click a row in `LiveTradeTape` → this component fetches
// `/api/trade/context?run_id=…&ts_in_ms=…` and renders the bundled
// snapshot: entry/exit signal probabilities, the bar windows, the
// decision chain, exit reason, model + params id. Lets analysts
// reconstruct WHY a trade was entered + how the market moved during
// the hold without joining tables that may have been shed.
//
// Two render paths:
//   * `source === "snapshot_file"` — fully-populated payload from the
//     forensics writer (Phase D3). Renders pre/in/post bar windows
//     as a sparkline + tabular peek.
//   * `source === "trade_ledger_only"` — fallback when the snapshot
//     file is missing (legacy trade or writer error). Shows the
//     denormalised columns only.

import { useEffect, useState } from "react";
import { Line, LineChart, ReferenceLine, ResponsiveContainer, YAxis } from "recharts";

import { fetchTradeContext, type TradeContext, type TradeContextBar } from "../mlApi";

interface Props {
  runId: string;
  tsInMs: number;
  onClose: () => void;
}

export function TradeContextDrawer({ runId, tsInMs, onClose }: Props) {
  const [ctx, setCtx] = useState<TradeContext | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let stopped = false;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetchTradeContext(runId, tsInMs, controller.signal)
      .then((c) => {
        if (!stopped) {
          setCtx(c);
          setLoading(false);
        }
      })
      .catch((e: unknown) => {
        if (stopped) return;
        const msg = e instanceof Error ? e.message : String(e);
        if (!msg.includes("aborted")) setError(msg);
        setLoading(false);
      });
    return () => {
      stopped = true;
      controller.abort();
    };
  }, [runId, tsInMs]);

  return (
    <div
      role="dialog"
      aria-label="Trade context"
      className="fixed inset-y-0 right-0 z-50 w-full max-w-3xl bg-ink-900 border-l border-ink-700 shadow-2xl shadow-black/40 overflow-y-auto"
    >
      <div className="sticky top-0 z-10 flex items-center justify-between gap-2 px-4 py-3 border-b border-ink-700 bg-ink-900/95 backdrop-blur">
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold tracking-wide">Trade context</span>
          <span className="text-[11px] font-mono text-slate-500">
            run_id={runId.slice(0, 12)}… · ts_in_ms={tsInMs}
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-xs text-slate-300 hover:text-slate-100 px-2 py-1 rounded border border-ink-700"
        >
          close ✕
        </button>
      </div>

      <div className="p-4 space-y-4">
        {loading && (
          <div className="text-xs text-slate-400">Loading trade context…</div>
        )}
        {error && (
          <div className="text-xs text-rose-300 break-all">
            {error.includes("404") ? "Trade not found in ledger." : error}
          </div>
        )}
        {ctx && <Body ctx={ctx} />}
      </div>
    </div>
  );
}

function Body({ ctx }: { ctx: TradeContext }) {
  const sideLabel = ctx.side > 0 ? "LONG" : ctx.side < 0 ? "SHORT" : "FLAT";
  const sideTone = ctx.side > 0 ? "text-emerald-200" : ctx.side < 0 ? "text-rose-200" : "text-slate-300";
  const r = ctx.realized_r;
  const rTone =
    r > 0 ? "text-emerald-200" : r < 0 ? "text-rose-200" : "text-slate-300";
  const decisionParts: string[] = (() => {
    if (Array.isArray(ctx.decision_chain)) return ctx.decision_chain;
    if (typeof ctx.decision_chain === "string") {
      try {
        const parsed = JSON.parse(ctx.decision_chain);
        return Array.isArray(parsed) ? parsed.map(String) : [ctx.decision_chain];
      } catch {
        return [ctx.decision_chain];
      }
    }
    return [];
  })();

  return (
    <>
      {ctx.source === "trade_ledger_only" && (
        <div className="text-[11px] px-2 py-1 rounded border border-amber-700/60 bg-amber-950/40 text-amber-200 inline-block">
          Snapshot file missing — showing denormalised ledger row only.
        </div>
      )}

      <Section title="Outcome">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <Stat label="Side" value={sideLabel} tone={sideTone} />
          <Stat label="Realized R" value={`${(r * 100).toFixed(2)}%`} tone={rTone} />
          <Stat label="Exit reason" value={ctx.exit_reason} />
          <Stat
            label="Hold"
            value={formatHold(ctx.exit_ts_ms - ctx.entry_ts_ms)}
          />
        </div>
      </Section>

      <Section title="Entry signal">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <Stat label="p_long" value={fmtPct(ctx.entry_p_long)} />
          <Stat label="p_short" value={fmtPct(ctx.entry_p_short)} />
          <Stat label="calibrated" value={fmtPct(ctx.entry_calibrated)} />
          <Stat label="spread (bp)" value={fmtNum(ctx.entry_spread_bp, 2)} />
          <Stat label="ATR(14)" value={fmtNum(ctx.entry_atr_14, 5)} />
          <Stat label="entry_px" value={ctx.entry_price.toFixed(5)} />
        </div>
      </Section>

      <Section title="Exit signal">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <Stat label="p_long" value={fmtPct(ctx.exit_p_long)} />
          <Stat label="p_short" value={fmtPct(ctx.exit_p_short)} />
          <Stat label="exit_px" value={ctx.exit_price.toFixed(5)} />
        </div>
      </Section>

      {decisionParts.length > 0 && (
        <Section title="Decision chain">
          <ul className="text-xs font-mono text-slate-300 list-disc list-inside space-y-0.5">
            {decisionParts.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </Section>
      )}

      <Section title="Champion + params">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
          <Stat label="model_id" value={ctx.model_id ?? "—"} mono />
          <Stat label="params_id" value={ctx.params_id ?? "—"} mono />
        </div>
      </Section>

      {(ctx.pre_trade_bars.length > 0 ||
        ctx.in_trade_bars.length > 0 ||
        ctx.post_trade_bars.length > 0) && (
        <Section title="Bar context">
          <BarChart
            preBars={ctx.pre_trade_bars}
            inBars={ctx.in_trade_bars}
            postBars={ctx.post_trade_bars}
            entryPx={ctx.entry_price}
            exitPx={ctx.exit_price}
          />
        </Section>
      )}

      {ctx.snapshot_path && (
        <Section title="Snapshot path">
          <div className="text-[11px] font-mono text-slate-500 break-all">
            {ctx.snapshot_path}
          </div>
        </Section>
      )}
    </>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">
        {title}
      </div>
      {children}
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
  mono,
}: {
  label: string;
  value: string;
  tone?: string;
  mono?: boolean;
}) {
  return (
    <div>
      <div className="text-slate-500 uppercase tracking-wide text-[10px]">{label}</div>
      <div
        className={`text-sm tabular-nums ${tone ?? "text-slate-100"} ${mono ? "font-mono break-all" : ""}`}
      >
        {value}
      </div>
    </div>
  );
}

function fmtPct(v: number | null): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtNum(v: number | null, digits: number): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

function formatHold(ms: number): string {
  if (ms < 0) return "—";
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s`;
  if (ms < 3_600_000) {
    const m = Math.floor(ms / 60_000);
    const s = Math.floor((ms % 60_000) / 1000);
    return `${m}m ${s.toString().padStart(2, "0")}s`;
  }
  const h = Math.floor(ms / 3_600_000);
  const m = Math.floor((ms % 3_600_000) / 60_000);
  return `${h}h ${m.toString().padStart(2, "0")}m`;
}

function BarChart({
  preBars,
  inBars,
  postBars,
  entryPx,
  exitPx,
}: {
  preBars: TradeContextBar[];
  inBars: TradeContextBar[];
  postBars: TradeContextBar[];
  entryPx: number;
  exitPx: number;
}) {
  const all = [...preBars, ...inBars, ...postBars];
  if (all.length === 0) return null;
  const data = all.map((b, i) => ({
    idx: i,
    close: b.close,
    ts_ms: b.ts_ms,
    region:
      i < preBars.length
        ? "pre"
        : i < preBars.length + inBars.length
          ? "in"
          : "post",
  }));
  const closes = data.map((d) => d.close);
  const lo = Math.min(...closes, entryPx, exitPx);
  const hi = Math.max(...closes, entryPx, exitPx);
  const pad = (hi - lo) * 0.08 + 1e-9;

  return (
    <div>
      <div className="h-32 border border-ink-700/70 rounded">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 4 }}>
            <YAxis
              hide
              type="number"
              domain={[lo - pad, hi + pad]}
            />
            <Line
              type="monotone"
              dataKey="close"
              stroke="#60a5fa"
              dot={false}
              strokeWidth={1.5}
              isAnimationActive={false}
            />
            <ReferenceLine
              y={entryPx}
              stroke="#22c55e"
              strokeDasharray="3 3"
              strokeOpacity={0.7}
            />
            <ReferenceLine
              y={exitPx}
              stroke="#ef4444"
              strokeDasharray="3 3"
              strokeOpacity={0.7}
            />
            <ReferenceLine
              x={preBars.length - 1}
              stroke="#475569"
              strokeOpacity={0.4}
            />
            <ReferenceLine
              x={preBars.length + inBars.length - 1}
              stroke="#475569"
              strokeOpacity={0.4}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex items-center gap-3 text-[10px] text-slate-500 mt-1">
        <span>
          <span className="inline-block h-2 w-2 mr-1 rounded-sm bg-slate-400/60" /> close
        </span>
        <span>
          <span className="inline-block h-0.5 w-3 mr-1 bg-emerald-400" /> entry
        </span>
        <span>
          <span className="inline-block h-0.5 w-3 mr-1 bg-rose-400" /> exit
        </span>
        <span className="text-slate-600">
          pre {preBars.length} · in {inBars.length} · post {postBars.length}
        </span>
      </div>
    </div>
  );
}

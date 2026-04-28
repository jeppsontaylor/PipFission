// Live trade tape — recent closed round-trips from the deployed
// trader, plus an aggregate scorecard. Polls /api/trade/ledger.
//
// Layout: scorecard row (count, hit rate, profit factor, cum-R) → small
// cumulative-R sparkline → most-recent rows table (newest first).

import { useMemo, useState } from "react";
import { LineChart, Line, ResponsiveContainer, YAxis } from "recharts";

import { Card } from "./Card";
import { TradeContextDrawer } from "./TradeContextDrawer";
import { useTradeLedger, type TradeLedgerRow } from "../mlApi";
import {
  cumulativeRCurve,
  exitReasonLabel,
  sideLabel,
  sideTone,
  summarizeLedger,
} from "../tradeLedger";

interface Props {
  /** Optional instrument filter. */
  instrument?: string;
  /** How many trades to load. Server caps at 10k. */
  limit?: number;
  /** Polling cadence (ms). Default 15s. */
  pollIntervalMs?: number;
}

export function LiveTradeTape({ instrument, limit = 200, pollIntervalMs }: Props) {
  const { rows, fetchedAt, loading, error } = useTradeLedger({
    instrument,
    limit,
    pollIntervalMs,
  });
  const summary = useMemo(() => summarizeLedger(rows), [rows]);
  const curve = useMemo(() => cumulativeRCurve(rows), [rows]);
  const recent = useMemo(
    () => [...rows].sort((a, b) => b.ts_out_ms - a.ts_out_ms).slice(0, 20),
    [rows],
  );
  // Phase D8: track which trade is open in the context drawer.
  const [openTrade, setOpenTrade] = useState<{ runId: string; tsInMs: number } | null>(
    null,
  );

  return (
    <Card
      title={instrument ? `Live Trades · ${instrument}` : "Live Trades"}
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
        <div className="flex items-center gap-3 flex-wrap">
          <Cell label="Trades" value={summary.count.toString()} sub="closed" />
          <Cell
            label="Hit rate"
            value={summary.count === 0 ? "—" : `${(summary.hitRate * 100).toFixed(1)}%`}
            sub={`${summary.wins}W / ${summary.losses}L`}
          />
          <Cell
            label="Profit factor"
            value={
              summary.profitFactor >= 999
                ? "∞"
                : summary.profitFactor.toFixed(2)
            }
            sub="Σwin / Σloss"
          />
          <Cell
            label="Cum R"
            value={`${(summary.cumulativeR * 100).toFixed(2)}%`}
            sub="net of costs"
            tone={summary.cumulativeR >= 0 ? "text-emerald-200" : "text-rose-200"}
          />
          <Cell
            label="Best / Worst"
            value={
              summary.bestTradeR === null || summary.worstTradeR === null
                ? "—"
                : `${(summary.bestTradeR * 100).toFixed(2)}% / ${(summary.worstTradeR * 100).toFixed(2)}%`
            }
            sub="single trade"
          />
          {curve.length > 1 && (
            <div className="ml-auto h-10 w-40" aria-label="cumulative R curve">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={curve}>
                  <YAxis hide />
                  <Line
                    type="monotone"
                    dataKey="cum_r"
                    stroke={summary.cumulativeR >= 0 ? "#34d399" : "#fb7185"}
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {recent.length === 0 ? (
          <div className="text-xs text-slate-500 italic">
            {loading
              ? "Polling…"
              : "No closed trades yet — live-trader hasn't routed a round-trip."}
          </div>
        ) : (
          <TradesTable
            rows={recent}
            onOpenContext={(r) =>
              setOpenTrade({ runId: r.run_id, tsInMs: r.ts_in_ms })
            }
          />
        )}
        {openTrade && (
          <TradeContextDrawer
            runId={openTrade.runId}
            tsInMs={openTrade.tsInMs}
            onClose={() => setOpenTrade(null)}
          />
        )}

        {error && (
          <div className="text-xs text-amber-300/80 break-all">REST: {error}</div>
        )}
      </div>
    </Card>
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

function TradesTable({
  rows,
  onOpenContext,
}: {
  rows: TradeLedgerRow[];
  onOpenContext: (r: TradeLedgerRow) => void;
}) {
  return (
    <div className="border border-ink-700/70 rounded overflow-hidden">
      <table className="w-full text-xs">
        <thead className="bg-ink-800/70 text-slate-400 uppercase text-[10px] tracking-wide">
          <tr>
            <th className="text-left px-2 py-1">Closed</th>
            <th className="text-left px-2 py-1">Inst</th>
            <th className="text-left px-2 py-1">Side</th>
            <th className="text-right px-2 py-1">Qty</th>
            <th className="text-right px-2 py-1">Entry</th>
            <th className="text-right px-2 py-1">Exit</th>
            <th className="text-right px-2 py-1">R</th>
            <th className="text-left px-2 py-1">Reason</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={`${r.run_id}:${r.ts_out_ms}:${r.instrument}`}
              className="border-t border-ink-700/50 hover:bg-ink-800/40 cursor-pointer"
              onClick={() => onOpenContext(r)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onOpenContext(r);
                }
              }}
              title="Click for full trade context"
            >
              <td className="px-2 py-1 font-mono text-slate-400">
                {formatTime(r.ts_out_ms)}
              </td>
              <td className="px-2 py-1 font-mono text-slate-300">{r.instrument}</td>
              <td className={`px-2 py-1 ${sideTone(r.side)}`}>{sideLabel(r.side)}</td>
              <td className="px-2 py-1 text-right tabular-nums text-slate-300">
                {formatQty(r.qty)}
              </td>
              <td className="px-2 py-1 text-right tabular-nums">{r.entry_px.toFixed(5)}</td>
              <td className="px-2 py-1 text-right tabular-nums">{r.exit_px.toFixed(5)}</td>
              <td
                className={`px-2 py-1 text-right tabular-nums ${
                  r.realized_r > 0
                    ? "text-emerald-300"
                    : r.realized_r < 0
                      ? "text-rose-300"
                      : "text-slate-400"
                }`}
              >
                {(r.realized_r * 100).toFixed(2)}%
              </td>
              <td className="px-2 py-1 text-slate-400">{exitReasonLabel(r.exit_reason)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatTime(ms: number): string {
  return new Date(ms).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatQty(q: number): string {
  if (Number.isInteger(q)) return q.toString();
  if (Math.abs(q) >= 100) return q.toFixed(0);
  return q.toFixed(2);
}

function formatAge(ms: number): string {
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s`;
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m`;
  return `${Math.floor(ms / 3_600_000)}h`;
}

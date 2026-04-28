// Deep performance breakdown for the model's paper trading. Reads the
// same /api/trade/ledger endpoint as LiveTradeTape but surfaces:
//   * overall stats (Sharpe/Sortino per trade, expectancy, drawdown)
//   * per-instrument breakdown (best/worst instrument)
//   * exit-reason histogram (where do trades close — take_profit?
//     stop_loss? max_hold?)

import { useMemo } from "react";

import { Card } from "./Card";
import { useTradeLedger } from "../mlApi";
import {
  computePerfStats,
  fmtNum,
  fmtPctR,
  instrumentBreakdown,
  reasonBreakdown,
  type ReasonBucket,
  type InstrumentPerf,
} from "../tradePerformance";

interface Props {
  limit?: number;
  pollIntervalMs?: number;
}

export function TradePerformanceBreakdown({
  limit = 500,
  pollIntervalMs,
}: Props) {
  const { rows, fetchedAt, loading, error } = useTradeLedger({
    limit,
    pollIntervalMs,
  });
  const stats = useMemo(() => computePerfStats(rows), [rows]);
  const reasons = useMemo(() => reasonBreakdown(rows), [rows]);
  const instruments = useMemo(() => instrumentBreakdown(rows), [rows]);
  const totalTrades = stats.count;

  return (
    <Card
      title="Trading Performance · Breakdown"
      right={
        <span className="text-xs text-slate-500">
          {fetchedAt
            ? `${stats.count} trades · updated ${formatAge(Date.now() - fetchedAt)} ago`
            : loading
              ? "loading…"
              : "—"}
        </span>
      }
    >
      <div className="space-y-4">
        {/* Top stats row — "key numbers at a glance" */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <Cell
            label="Cum R"
            value={fmtPctR(stats.cumulativeR)}
            sub="net of costs"
            tone={stats.cumulativeR >= 0 ? "text-emerald-200" : "text-rose-200"}
          />
          <Cell
            label="Hit rate"
            value={
              totalTrades > 0
                ? `${(stats.hitRate * 100).toFixed(1)}%`
                : "—"
            }
            sub={`${stats.wins}W / ${stats.losses}L / ${stats.scratched}–`}
          />
          <Cell
            label="Profit factor"
            value={
              stats.profitFactor >= 999
                ? "∞"
                : fmtNum(stats.profitFactor, 2)
            }
            sub="Σwin / Σloss"
          />
          <Cell
            label="Max drawdown"
            value={fmtPctR(-stats.maxDrawdownR)}
            sub="peak-to-trough cum R"
            tone={stats.maxDrawdownR > 0 ? "text-rose-200" : "text-slate-300"}
          />
        </div>

        {/* Second row — relative comparison metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <Cell
            label="Avg win"
            value={fmtPctR(stats.avgWinR)}
            sub={`${stats.wins} wins`}
            tone="text-emerald-200"
          />
          <Cell
            label="Avg loss"
            value={fmtPctR(stats.avgLossR)}
            sub={`${stats.losses} losses`}
            tone="text-rose-200"
          />
          <Cell
            label="Expectancy"
            value={fmtPctR(stats.expectancyR, 3)}
            sub="per trade"
          />
          <Cell
            label="Sharpe / Sortino"
            value={`${fmtNum(stats.sharpePerTrade, 2)} / ${fmtNum(stats.sortinoPerTrade, 2)}`}
            sub="per trade (relative)"
          />
        </div>

        {/* Per-instrument breakdown */}
        {instruments.length > 0 && (
          <Section title="Per-instrument">
            <InstrumentTable rows={instruments} />
          </Section>
        )}

        {/* Exit-reason histogram */}
        {reasons.length > 0 && (
          <Section title="Exit reasons">
            <ReasonTable rows={reasons} totalTrades={totalTrades} />
          </Section>
        )}

        {totalTrades === 0 && !loading && (
          <div className="text-xs text-slate-500 italic">
            No closed trades yet — once the live-trader closes a round-trip,
            this card will populate.
          </div>
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

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-1.5">
        {title}
      </div>
      {children}
    </div>
  );
}

function InstrumentTable({ rows }: { rows: InstrumentPerf[] }) {
  return (
    <div className="border border-ink-700/70 rounded overflow-hidden">
      <table className="w-full text-xs">
        <thead className="bg-ink-800/70 text-slate-400 uppercase text-[10px] tracking-wide">
          <tr>
            <th className="text-left px-2 py-1">Instrument</th>
            <th className="text-right px-2 py-1">Cum R</th>
            <th className="text-right px-2 py-1">Trades</th>
            <th className="text-right px-2 py-1">Hit rate</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const tone =
              r.cumulativeR > 0
                ? "text-emerald-200"
                : r.cumulativeR < 0
                  ? "text-rose-200"
                  : "text-slate-300";
            return (
              <tr key={r.instrument} className="border-t border-ink-700/50">
                <td className="px-2 py-1 font-mono text-slate-200">{r.instrument}</td>
                <td className={`px-2 py-1 text-right tabular-nums ${tone}`}>
                  {fmtPctR(r.cumulativeR)}
                </td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-400">
                  {r.count}
                </td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-300">
                  {(r.hitRate * 100).toFixed(0)}%
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ReasonTable({
  rows,
  totalTrades,
}: {
  rows: ReasonBucket[];
  totalTrades: number;
}) {
  const maxCount = Math.max(1, ...rows.map((r) => r.count));
  return (
    <div className="border border-ink-700/70 rounded overflow-hidden">
      <table className="w-full text-xs">
        <thead className="bg-ink-800/70 text-slate-400 uppercase text-[10px] tracking-wide">
          <tr>
            <th className="text-left px-2 py-1">Reason</th>
            <th className="text-right px-2 py-1">Trades</th>
            <th className="text-right px-2 py-1">% of all</th>
            <th className="text-left px-2 py-1 w-28">Histogram</th>
            <th className="text-right px-2 py-1">Avg R</th>
            <th className="text-right px-2 py-1">Cum R</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const tone =
              r.cumulativeR > 0
                ? "text-emerald-200"
                : r.cumulativeR < 0
                  ? "text-rose-200"
                  : "text-slate-300";
            const widthPct = (r.count / maxCount) * 100;
            return (
              <tr key={r.reason} className="border-t border-ink-700/50">
                <td className="px-2 py-1 text-slate-300">{prettyReason(r.reason)}</td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-400">
                  {r.count}
                </td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-500">
                  {totalTrades > 0
                    ? `${((r.count / totalTrades) * 100).toFixed(0)}%`
                    : "—"}
                </td>
                <td className="px-2 py-1">
                  <span
                    className="block h-1.5 rounded bg-sky-400/70"
                    style={{ width: `${widthPct}%` }}
                    aria-hidden="true"
                  />
                </td>
                <td className={`px-2 py-1 text-right tabular-nums ${tone}`}>
                  {fmtPctR(r.avgR)}
                </td>
                <td className={`px-2 py-1 text-right tabular-nums ${tone}`}>
                  {fmtPctR(r.cumulativeR)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function prettyReason(reason: string): string {
  switch (reason) {
    case "take_profit":
      return "Take profit";
    case "stop_loss":
      return "Stop loss";
    case "max_hold":
      return "Max hold";
    case "trailing_stop":
      return "Trailing stop";
    case "signal_flip":
      return "Signal flip";
    case "manual":
      return "Manual";
    case "shutdown":
      return "Shutdown";
    default:
      return reason;
  }
}

function formatAge(ms: number): string {
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s`;
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m`;
  return `${Math.floor(ms / 3_600_000)}h`;
}

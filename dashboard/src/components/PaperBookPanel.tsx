import { Card } from "./Card";
import { fmtCcy, fmtPx, instrumentDecimals } from "../store";
import type { PaperBookSnapshot, Reconciliation, RoutingMode } from "../types";

export interface PaperBookPanelProps {
  paperBook: PaperBookSnapshot | null;
  reconciliation: Reconciliation | null;
  mode: RoutingMode;
}

export function PaperBookPanel({ paperBook, reconciliation, mode }: PaperBookPanelProps) {
  const equity = paperBook?.equity ?? null;
  const cash = paperBook?.cash ?? null;
  const realized = paperBook?.realized_pl ?? null;
  const unrealized = paperBook?.unrealized_pl ?? null;
  const positions = paperBook?.positions ?? [];

  const equityClass =
    equity !== null && equity > 100_000
      ? "text-bull-400"
      : equity !== null && equity < 100_000
        ? "text-bear-400"
        : "text-slate-100";

  return (
    <Card
      title={`Internal paper book ${mode === "oanda_practice" ? "(also OANDA-routing)" : ""}`}
    >
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <Stat label="Equity" value={fmtCcy(equity)} valueClass={`text-lg ${equityClass}`} />
        <Stat label="Cash" value={fmtCcy(cash)} />
        <Stat
          label="Realized P/L"
          value={fmtCcy(realized)}
          valueClass={(realized ?? 0) >= 0 ? "text-bull-400" : "text-bear-400"}
        />
        <Stat
          label="Unrealized P/L"
          value={fmtCcy(unrealized)}
          valueClass={(unrealized ?? 0) >= 0 ? "text-bull-400" : "text-bear-400"}
        />
      </div>

      {mode === "oanda_practice" && reconciliation && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 px-3 py-2 mb-3 text-xs flex items-center justify-between">
          <div>
            <span className="text-slate-300">OANDA NAV vs internal:</span>{" "}
            <span className="font-mono text-amber-300">
              {fmtCcy(reconciliation.oanda_actual_equity)}
            </span>{" "}
            <span className="text-slate-500">vs</span>{" "}
            <span className="font-mono text-slate-100">
              {fmtCcy(reconciliation.internal_paper_equity)}
            </span>
          </div>
          <div className="font-mono text-amber-300">
            Δ {reconciliation.oanda_minus_internal_bp.toFixed(1)} bp
          </div>
        </div>
      )}

      <div className="overflow-hidden rounded-lg border border-ink-800">
        <table className="w-full text-xs">
          <thead className="bg-ink-800/60 text-slate-400">
            <tr>
              <th className="text-left px-3 py-2 font-medium">Instrument</th>
              <th className="text-right px-3 py-2 font-medium">Units</th>
              <th className="text-right px-3 py-2 font-medium">Avg price</th>
              <th className="text-right px-3 py-2 font-medium">Realized</th>
            </tr>
          </thead>
          <tbody>
            {positions.length === 0 && (
              <tr>
                <td className="px-3 py-3 text-slate-500" colSpan={4}>
                  No open positions.
                </td>
              </tr>
            )}
            {positions
              .filter((p) => p.units !== 0 || p.realized !== 0)
              .map((p) => {
                const dec = instrumentDecimals(p.instrument);
                const unitClass =
                  p.units > 0
                    ? "text-bull-400"
                    : p.units < 0
                      ? "text-bear-400"
                      : "text-slate-400";
                return (
                  <tr key={p.instrument} className="border-t border-ink-800">
                    <td className="px-3 py-2 font-mono text-slate-300">{p.instrument}</td>
                    <td className={`px-3 py-2 text-right font-mono ${unitClass}`}>
                      {p.units.toLocaleString()}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-slate-300">
                      {fmtPx(p.avg_price, dec)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-slate-400">
                      {fmtCcy(p.realized)}
                    </td>
                  </tr>
                );
              })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function Stat({
  label,
  value,
  valueClass = "text-slate-100",
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`font-mono ${valueClass}`}>{value}</div>
    </div>
  );
}

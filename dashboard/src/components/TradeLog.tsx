import type { TransactionEvent } from "../types";
import { Card } from "./Card";
import { fmtCcy, fmtPx, instrumentDecimals } from "../store";

interface Props {
  transactions: TransactionEvent[];
}

const INTERESTING = new Set([
  "ORDER_FILL",
  "MARKET_ORDER",
  "MARKET_ORDER_REJECT",
  "LIMIT_ORDER",
  "STOP_LOSS_ORDER",
  "TAKE_PROFIT_ORDER",
  "TRAILING_STOP_LOSS_ORDER",
  "TRADE_CLOSE",
  "DAILY_FINANCING",
  "TRANSFER_FUNDS",
]);

/** Trade / transaction stream — newest at the top, fills highlighted. */
export function TradeLog({ transactions }: Props) {
  const filtered = transactions
    .filter((t) => INTERESTING.has(t.kind) || t.kind.endsWith("FILL"))
    .slice()
    .reverse();

  return (
    <Card
      title="Algorithmic trades & transactions"
      right={
        <span className="text-xs text-slate-400">{transactions.length} events</span>
      }
    >
      {filtered.length === 0 ? (
        <div className="text-sm text-slate-500 py-6 text-center">
          No fills yet. The dashboard will light up as soon as the algo (or you)
          submits orders to OANDA.
        </div>
      ) : (
        <div className="overflow-y-auto max-h-96">
          <table className="w-full text-sm font-mono">
            <thead className="text-xs uppercase text-slate-500 border-b border-ink-700">
              <tr>
                <th className="text-left py-1.5 pr-2">Time</th>
                <th className="text-left pr-2">Kind</th>
                <th className="text-left pr-2">Instr.</th>
                <th className="text-right pr-2">Units</th>
                <th className="text-right pr-2">Price</th>
                <th className="text-right pr-2">P/L</th>
                <th className="text-left">Reason</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((t) => (
                <Row key={t.id} t={t} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

function Row({ t }: { t: TransactionEvent }) {
  const time = new Date(t.time).toLocaleTimeString();
  const isFill = t.kind === "ORDER_FILL";
  const plClass = (t.pl ?? 0) > 0 ? "text-bull" : (t.pl ?? 0) < 0 ? "text-bear" : "text-slate-400";
  const dirClass =
    t.units !== null && t.units !== undefined
      ? t.units > 0
        ? "text-bull"
        : t.units < 0
          ? "text-bear"
          : "text-slate-300"
      : "text-slate-300";
  return (
    <tr className={`border-b border-ink-800/60 ${isFill ? "" : "text-slate-400"}`}>
      <td className="py-1 pr-2 text-slate-400 whitespace-nowrap">{time}</td>
      <td className="pr-2">{t.kind}</td>
      <td className="pr-2">{t.instrument ?? "—"}</td>
      <td className={`pr-2 text-right ${dirClass}`}>
        {t.units !== null && t.units !== undefined ? t.units.toLocaleString() : "—"}
      </td>
      <td className="pr-2 text-right">
        {t.price !== null && t.price !== undefined && t.instrument
          ? fmtPx(t.price, instrumentDecimals(t.instrument))
          : "—"}
      </td>
      <td className={`pr-2 text-right ${plClass}`}>
        {t.pl !== null && t.pl !== undefined ? fmtCcy(t.pl) : "—"}
      </td>
      <td className="text-slate-400 truncate max-w-[16ch]">{t.reason ?? ""}</td>
    </tr>
  );
}

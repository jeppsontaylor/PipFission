import { Card } from "./Card";
import { fmtCcy } from "../store";
import type { AlpacaAccountSnapshot } from "../types";

export interface AlpacaAccountCardProps {
  account: AlpacaAccountSnapshot | null;
}

export function AlpacaAccountCard({ account }: AlpacaAccountCardProps) {
  if (!account) {
    return (
      <Card title="Alpaca paper · crypto">
        <div className="text-xs text-slate-500">Not configured (set ALPACA_KEY / ALPACA_SECRET).</div>
      </Card>
    );
  }
  const equityClass =
    account.equity > 100_000
      ? "text-bull-400"
      : account.equity < 100_000
        ? "text-bear-400"
        : "text-slate-100";
  return (
    <Card title="Alpaca paper · crypto">
      <div className="grid grid-cols-2 gap-4">
        <Stat label="Equity" value={fmtCcy(account.equity)} valueClass={`text-lg ${equityClass}`} />
        <Stat label="Cash" value={fmtCcy(account.cash)} />
        <Stat label="Buying power" value={fmtCcy(account.buying_power)} />
        <Stat
          label="Status"
          value={account.status}
          valueClass={account.status === "ACTIVE" ? "text-bull-400" : "text-amber-300"}
        />
        <Stat label="Long market value" value={fmtCcy(account.long_market_value)} />
        <Stat label="Short market value" value={fmtCcy(account.short_market_value)} />
      </div>
      <div className="mt-3 text-[11px] text-slate-500 font-mono">
        currency: {account.currency} · updated {new Date(account.time).toLocaleTimeString()}
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

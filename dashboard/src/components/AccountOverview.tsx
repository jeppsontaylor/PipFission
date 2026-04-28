import type { AccountSnapshot, EstimateTick } from "../types";
import { Card } from "./Card";
import { fmtCcy } from "../store";

interface Props {
  account: AccountSnapshot | null;
  estimate: EstimateTick | null;
  accountId: string | null;
  environment: string | null;
}

/** Reproduces the OANDA "Account Information" card, plus the estimated-vs-actual. */
export function AccountOverview({ account, estimate, accountId, environment }: Props) {
  const ccy = account?.currency ?? "USD";
  const drift = estimate?.drift ?? 0;
  const driftBps = estimate?.drift_bps ?? 0;
  const driftClass = drift === 0 ? "text-slate-400" : drift > 0 ? "text-bull" : "text-bear";

  return (
    <Card
      title="Account"
      right={
        <span className="text-xs font-mono text-slate-400">
          {accountId ?? "—"} · {environment ?? "—"}
        </span>
      }
    >
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Stat label="NAV" value={fmtCcy(account?.nav, ccy)} big />
        <Stat label="Balance" value={fmtCcy(account?.balance, ccy)} />
        <Stat
          label="Unrealized P/L"
          value={fmtCcy(account?.unrealized_pl, ccy)}
          tone={(account?.unrealized_pl ?? 0) >= 0 ? "good" : "bad"}
        />
        <Stat
          label="Realized P/L"
          value={fmtCcy(account?.realized_pl, ccy)}
          tone={(account?.realized_pl ?? 0) >= 0 ? "good" : "bad"}
        />
        <Stat label="Margin Used" value={fmtCcy(account?.margin_used, ccy)} />
        <Stat label="Margin Available" value={fmtCcy(account?.margin_available, ccy)} />
        <Stat
          label="Leverage"
          value={account?.leverage ? `${account.leverage.toFixed(0)}:1` : "—"}
        />
        <Stat
          label="Open positions / trades"
          value={`${account?.open_position_count ?? 0} / ${account?.open_trade_count ?? 0}`}
        />
      </div>
      <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
        <Stat
          label="Estimated"
          value={fmtCcy(estimate?.estimated_balance, ccy)}
          mono
        />
        <Stat label="Actual NAV" value={fmtCcy(estimate?.actual_balance, ccy)} mono />
        <div>
          <div className="text-xs uppercase tracking-wider text-slate-400">Drift</div>
          <div className={`mt-1 font-mono ${driftClass}`}>
            {fmtCcy(drift, ccy)}{" "}
            <span className="text-xs text-slate-500">({driftBps.toFixed(2)} bps)</span>
          </div>
        </div>
      </div>
    </Card>
  );
}

function Stat({
  label,
  value,
  big,
  mono,
  tone,
}: {
  label: string;
  value: string;
  big?: boolean;
  mono?: boolean;
  tone?: "good" | "bad";
}) {
  const toneClass =
    tone === "good" ? "text-bull" : tone === "bad" ? "text-bear" : "text-slate-100";
  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-slate-400">{label}</div>
      <div
        className={`mt-1 ${big ? "text-2xl" : "text-base"} ${
          mono ? "font-mono" : ""
        } ${toneClass}`}
      >
        {value}
      </div>
    </div>
  );
}

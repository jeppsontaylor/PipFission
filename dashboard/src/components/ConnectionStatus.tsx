import type { ConnStatus } from "../types";
import type { WsStatus } from "../ws";

interface Props {
  ws: WsStatus;
  conn: ConnStatus | null;
}

export function ConnectionStatus({ ws, conn }: Props) {
  const items: { name: string; ok: boolean; detail: string }[] = [
    { name: "WebSocket", ok: ws === "open", detail: ws },
    {
      name: "Pricing",
      ok: !!conn?.pricing_stream.connected,
      detail: conn?.pricing_stream.last_error ?? `${conn?.pricing_stream.messages_received ?? 0} msgs`,
    },
    {
      name: "Transactions",
      ok: !!conn?.transaction_stream.connected,
      detail: conn?.transaction_stream.last_error ?? `${conn?.transaction_stream.messages_received ?? 0} msgs`,
    },
    {
      name: "Account",
      ok: !!conn?.account_poll.connected,
      detail: conn?.account_poll.last_error ?? `${conn?.account_poll.messages_received ?? 0} polls`,
    },
  ];

  return (
    <div className="flex items-center gap-3 text-xs font-mono">
      {items.map((it) => (
        <div key={it.name} className="flex items-center gap-1.5">
          <span
            className={`live-dot inline-block w-2 h-2 rounded-full ${
              it.ok ? "bg-bull" : "bg-bear"
            }`}
          />
          <span className="text-slate-300">{it.name}</span>
          <span className="text-slate-500 hidden md:inline">· {it.detail}</span>
        </div>
      ))}
    </div>
  );
}

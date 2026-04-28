import { useState } from "react";

import type { RoutingMode } from "../types";
import { ConfirmModal } from "./ConfirmModal";

export interface ModeToggleProps {
  mode: RoutingMode;
  pendingMode: RoutingMode | null;
  lastError: string | null;
  onChange: (mode: RoutingMode) => void;
}

export function ModeToggle({ mode, pendingMode, lastError, onChange }: ModeToggleProps) {
  const [confirmOpen, setConfirmOpen] = useState(false);
  const isOanda = mode === "oanda_practice";
  const isPending = pendingMode !== null && pendingMode !== mode;

  const requestMode = (next: RoutingMode) => {
    if (next === mode) return;
    if (next === "oanda_practice") {
      setConfirmOpen(true);
      return;
    }
    onChange(next);
  };

  const buttonBase =
    "px-3 py-1.5 text-xs font-medium tracking-wide transition-colors disabled:opacity-50";
  return (
    <div className="flex flex-col items-end gap-1">
      <div
        className="inline-flex rounded-full border border-ink-700 bg-ink-900 overflow-hidden"
        role="group"
        aria-label="Routing mode"
      >
        <button
          type="button"
          disabled={isPending}
          onClick={() => requestMode("internal")}
          className={`${buttonBase} ${
            !isOanda
              ? "bg-ink-700 text-slate-100"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          Internal
        </button>
        <button
          type="button"
          disabled={isPending}
          onClick={() => requestMode("oanda_practice")}
          className={`${buttonBase} ${
            isOanda
              ? "bg-amber-500 text-ink-950"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          OANDA Paper
        </button>
      </div>
      {isPending ? (
        <div className="text-[11px] text-amber-300 font-mono">switching…</div>
      ) : lastError ? (
        <div className="text-[11px] text-bear-400 font-mono max-w-[260px] truncate" title={lastError}>
          {lastError}
        </div>
      ) : isOanda ? (
        <div className="text-[11px] text-amber-300 font-mono">live → OANDA practice</div>
      ) : (
        <div className="text-[11px] text-slate-500 font-mono">internal-only</div>
      )}

      <ConfirmModal
        open={confirmOpen}
        title="Switch to OANDA Paper?"
        confirmLabel="Switch to OANDA Paper"
        variant="amber"
        body={
          <div className="space-y-2">
            <p>
              Your strategy will start submitting <strong>real orders</strong> to your OANDA practice
              account ($100k starting balance — no real money). Fills, P/L and positions will reflect
              actual OANDA execution including spread and slippage.
            </p>
            <p className="text-slate-400">
              Your internal paper book keeps running in parallel for side-by-side reconciliation.
            </p>
          </div>
        }
        onCancel={() => setConfirmOpen(false)}
        onConfirm={() => {
          setConfirmOpen(false);
          onChange("oanda_practice");
        }}
      />
    </div>
  );
}

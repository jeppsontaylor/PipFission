import { useEffect } from "react";

export interface ConfirmModalProps {
  open: boolean;
  title: string;
  body: React.ReactNode;
  confirmLabel: string;
  cancelLabel?: string;
  variant?: "default" | "amber";
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmModal({
  open,
  title,
  body,
  confirmLabel,
  cancelLabel = "Cancel",
  variant = "default",
  onConfirm,
  onCancel,
}: ConfirmModalProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel]);

  if (!open) return null;
  const confirmClass =
    variant === "amber"
      ? "bg-amber-500 hover:bg-amber-400 text-ink-950"
      : "bg-accent-500 hover:bg-accent-400 text-white";
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-ink-950/70 backdrop-blur-sm"
        onClick={onCancel}
      />
      <div className="relative max-w-md w-full mx-4 rounded-2xl border border-ink-700 bg-ink-900/95 shadow-2xl shadow-black/40 p-6">
        <div className="text-lg font-semibold text-slate-100 mb-3">{title}</div>
        <div className="text-sm text-slate-300 leading-relaxed mb-6">{body}</div>
        <div className="flex gap-3 justify-end">
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 rounded-lg text-sm font-medium border border-ink-700 text-slate-200 hover:bg-ink-800"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className={`px-4 py-2 rounded-lg text-sm font-semibold ${confirmClass}`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

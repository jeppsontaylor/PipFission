import { ReactNode } from "react";

interface Props {
  title?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}

/** A common dark panel with a title row. */
export function Card({ title, right, children, className }: Props) {
  return (
    <div
      className={
        "rounded-xl border border-ink-700 bg-ink-900/60 backdrop-blur-sm shadow-lg shadow-black/20 " +
        (className ?? "")
      }
    >
      {(title || right) && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-ink-700/70">
          <div className="text-sm font-medium tracking-wide uppercase text-slate-300">
            {title}
          </div>
          <div>{right}</div>
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}

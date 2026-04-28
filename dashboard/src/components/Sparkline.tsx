import { useMemo } from "react";

interface Props {
  values: number[];
  width?: number;
  height?: number;
  positive?: boolean;
}

/** Minimal SVG sparkline. Auto-scales y; flat when only one point. */
export function Sparkline({ values, width = 100, height = 28, positive }: Props) {
  const path = useMemo(() => {
    if (values.length < 2) return "";
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of values) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (hi === lo) {
      hi += 0.0001;
      lo -= 0.0001;
    }
    const step = width / (values.length - 1);
    return values
      .map((v, i) => {
        const x = i * step;
        const y = height - ((v - lo) / (hi - lo)) * height;
        return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
  }, [values, width, height]);

  const stroke =
    positive === undefined ? "#94a3b8" : positive ? "#22c55e" : "#ef4444";

  return (
    <svg width={width} height={height} className="overflow-visible">
      <path d={path} stroke={stroke} strokeWidth={1.4} fill="none" />
    </svg>
  );
}

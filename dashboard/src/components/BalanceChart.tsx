import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useMemo } from "react";
import type { NavPoint } from "../store";
import { Card } from "./Card";
import { fmtCcy } from "../store";

interface Props {
  history: NavPoint[];
}

/** Estimated vs actual NAV. The whole point of the dashboard. */
export function BalanceChart({ history }: Props) {
  const data = useMemo(
    () =>
      history.map((p) => ({
        t: p.t,
        label: new Date(p.t).toLocaleTimeString(),
        estimated: p.estimated,
        actual: p.actual,
        drift: p.drift,
      })),
    [history],
  );

  // Pick a y-domain that gives a few hundred dollars of headroom either side
  // of the visible range so small drift moves are readable.
  const domain = useMemo<[number, number]>(() => {
    if (data.length === 0) return [0, 0];
    let lo = Infinity;
    let hi = -Infinity;
    for (const d of data) {
      lo = Math.min(lo, d.estimated, d.actual);
      hi = Math.max(hi, d.estimated, d.actual);
    }
    if (hi === lo) {
      hi += 1;
      lo -= 1;
    }
    const pad = (hi - lo) * 0.15;
    return [lo - pad, hi + pad];
  }, [data]);

  return (
    <Card
      title="Estimated vs actual NAV"
      right={
        <span className="text-xs text-slate-400">
          {history.length} pts · est = synthetic ledger, actual = OANDA NAV
        </span>
      }
    >
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#1c2030" vertical={false} />
            <XAxis
              dataKey="label"
              stroke="#64748b"
              tick={{ fontSize: 11 }}
              minTickGap={48}
            />
            <YAxis
              domain={domain}
              stroke="#64748b"
              tick={{ fontSize: 11 }}
              tickFormatter={(v) => fmtCcy(v as number).replace("$", "$ ")}
              width={88}
            />
            <Tooltip
              contentStyle={{
                background: "#0d0f15",
                border: "1px solid #262b3d",
                borderRadius: 6,
                fontSize: 12,
              }}
              formatter={(value: unknown) =>
                typeof value === "number" ? fmtCcy(value) : String(value)
              }
              labelFormatter={(l) => `Time ${l}`}
            />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#3b82f6"
              dot={false}
              strokeWidth={2}
              name="Actual NAV"
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="estimated"
              stroke="#22c55e"
              dot={false}
              strokeWidth={2}
              name="Estimated"
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

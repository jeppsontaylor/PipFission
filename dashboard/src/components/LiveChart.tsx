import { useMemo } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ModelFitness, PaperFillEvent, PriceTick, StrategySignal } from "../types";
import { instrumentVenue } from "../types";
import type { PricePoint, SignalPoint } from "../store";
import { fmtPx, instrumentDecimals } from "../store";
import type { LabelRow } from "../mlApi";
import { buildLabelMarkers, makePriceLookup, partitionMarkers } from "../labelOverlay";

// Mirror of strategy::IntentEmitter::EmitterConfig::confidence_threshold.
// Signals below this don't fire orders.
const CONFIDENCE_THRESHOLD = 0.55;

interface Props {
  instrument: string;
  tick: PriceTick | undefined;
  history: PricePoint[];
  fills: PaperFillEvent[];
  signal: StrategySignal | undefined;
  signalHistory: SignalPoint[];
  fitness: ModelFitness | undefined;
  /** Optional. When present, the optimizer's ideal long/short entries
   *  for this instrument's most-recent label run are drawn as small
   *  diamonds on the price line. */
  labels?: LabelRow[];
}

/**
 * Big streaming chart for a single instrument. Top row is the live mid
 * price with buy/sell fill markers overlaid; bottom row is the model's
 * signed confidence trace with a horizontal threshold line.
 *
 * Both rows share an x-axis. Designed to fill the upper viewport — give
 * it `h-[70vh]` or similar from the parent.
 */
export function LiveChart({
  instrument,
  tick,
  history,
  fills,
  signal,
  signalHistory,
  fitness,
  labels,
}: Props) {
  const decimals = instrumentDecimals(instrument);
  const venue = instrumentVenue(instrument);

  // Time domain spans whatever's in the price history. Fills + signals
  // anchor to that window so a stale fill from minutes ago disappears
  // off the left edge naturally.
  const { priceData, domain, yDomain } = useMemo(() => {
    const pts = history.map((p) => ({
      t: p.t,
      mid: p.mid,
      bid: p.bid,
      ask: p.ask,
    }));
    if (pts.length === 0) {
      return { priceData: pts, domain: [0, 1] as [number, number], yDomain: [0, 1] as [number, number] };
    }
    const t0 = pts[0].t;
    const t1 = pts[pts.length - 1].t;
    let lo = Infinity;
    let hi = -Infinity;
    for (const p of pts) {
      if (p.bid < lo) lo = p.bid;
      if (p.ask > hi) hi = p.ask;
    }
    if (hi === lo) {
      hi += hi * 0.0005 + 1e-9;
      lo -= lo * 0.0005 + 1e-9;
    }
    const pad = (hi - lo) * 0.08;
    return {
      priceData: pts,
      domain: [t0, t1] as [number, number],
      yDomain: [lo - pad, hi + pad] as [number, number],
    };
  }, [history]);

  const { buyFills, sellFills } = useMemo(() => {
    const buys: { t: number; price: number; units: number }[] = [];
    const sells: { t: number; price: number; units: number }[] = [];
    for (const f of fills) {
      if (f.instrument !== instrument) continue;
      const t = Date.parse(f.time);
      if (t < domain[0] || t > domain[1]) continue;
      if (f.units > 0) buys.push({ t, price: f.price, units: f.units });
      else if (f.units < 0) sells.push({ t, price: f.price, units: f.units });
    }
    return { buyFills: buys, sellFills: sells };
  }, [fills, instrument, domain]);

  const confData = useMemo(() => {
    return signalHistory
      .filter((p) => p.t >= domain[0] && p.t <= domain[1])
      .map((p) => ({ t: p.t, signed: p.signedConfidence, conf: p.confidence }));
  }, [signalHistory, domain]);

  // Label-overlay markers — the optimizer's ideal long/short entries on
  // the trailing 1000-bar window. Filtered to the chart's x-domain and
  // the most recent label_run_id, joined to bar-mid via binary search.
  const { longs: longLabels, shorts: shortLabels } = useMemo(() => {
    if (!labels || labels.length === 0 || priceData.length === 0) {
      return { longs: [], shorts: [] };
    }
    const lookup = makePriceLookup(priceData);
    const filtered = labels.filter((r) => r.instrument === instrument);
    const markers = buildLabelMarkers(filtered, domain, lookup);
    return partitionMarkers(markers);
  }, [labels, instrument, priceData, domain]);

  const tsFmt = (t: number | string) => {
    const ms = typeof t === "number" ? t : Date.parse(String(t));
    if (!Number.isFinite(ms)) return "";
    return new Date(ms).toLocaleTimeString();
  };

  const venueColor = venue === "alpaca" ? "#d946ef" : "#3b82f6";
  const lastTime = tick ? new Date(Date.parse(tick.time)).toLocaleTimeString() : "—";
  const spreadBps = tick && tick.mid > 0 ? (tick.spread / tick.mid) * 10_000 : 0;
  const oosPct = fitness ? (fitness.oos.accuracy * 100).toFixed(1) : null;
  const samples = fitness?.samples_seen ?? 0;
  const isStrong = fitness ? fitness.oos.accuracy >= 0.40 && samples >= 200 : false;

  const sigLabel = signal
    ? `${signal.direction.toUpperCase()} @ ${(signal.confidence * 100).toFixed(0)}%`
    : "warming up";
  const sigClass = !signal
    ? "text-slate-500"
    : signal.direction === "long"
      ? "text-bull"
      : signal.direction === "short"
        ? "text-bear"
        : "text-slate-300";

  return (
    <div className="rounded-xl border border-ink-700 bg-ink-900/60 backdrop-blur-sm shadow-lg shadow-black/20">
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 border-b border-ink-700/70">
        <div className="flex items-center gap-3 min-w-0">
          <div className="text-lg font-semibold tracking-wide font-mono">
            {instrument.replace("_", "/")}
          </div>
          <span
            className="inline-block px-1.5 py-0.5 text-[10px] tracking-widest rounded border font-mono"
            style={{ color: venueColor, borderColor: `${venueColor}66`, background: `${venueColor}1a` }}
          >
            {venue.toUpperCase()}
          </span>
          <div className="text-xs text-slate-400 font-mono">
            mid <span className="text-slate-100">{fmtPx(tick?.mid, decimals)}</span> · spread{" "}
            {spreadBps.toFixed(1)}bps · {lastTime}
          </div>
        </div>
        <div className="flex items-center gap-3 text-xs font-mono">
          <span className="text-slate-500">signal</span>
          <span className={sigClass}>{sigLabel}</span>
          <span className="text-slate-700">·</span>
          <span className="text-slate-500">model</span>
          <span className={isStrong ? "text-bull" : "text-slate-400"}>
            {oosPct !== null ? `OOS ${oosPct}%` : "untrained"}
          </span>
          <span className="text-slate-600">({samples} pts)</span>
        </div>
      </div>

      {/* Price chart — fills the bulk of the viewport */}
      <div className="px-2 pt-3" style={{ height: "calc(70vh - 90px)" }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={priceData} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
            <CartesianGrid stroke="#1c2030" vertical={false} />
            <XAxis
              dataKey="t"
              type="number"
              domain={domain}
              stroke="#64748b"
              tick={{ fontSize: 11 }}
              tickFormatter={tsFmt}
              minTickGap={64}
            />
            <YAxis
              yAxisId="px"
              domain={yDomain}
              stroke="#64748b"
              tick={{ fontSize: 11 }}
              tickFormatter={(v) => fmtPx(v as number, decimals)}
              width={88}
              orientation="right"
            />
            <Tooltip
              contentStyle={{
                background: "#0d0f15",
                border: "1px solid #262b3d",
                borderRadius: 6,
                fontSize: 12,
              }}
              labelFormatter={(l) => `Time ${tsFmt(l as number)}`}
              formatter={(v: unknown, name) => {
                const num = typeof v === "number" ? v : Number(v);
                if (!Number.isFinite(num)) return [String(v), String(name)];
                if (name === "units") return [num, "units"];
                return [fmtPx(num, decimals), String(name)];
              }}
            />
            <Line
              yAxisId="px"
              type="monotone"
              dataKey="mid"
              stroke={venueColor}
              dot={false}
              strokeWidth={1.5}
              isAnimationActive={false}
              name="mid"
            />
            {/* Fill markers — use Scatter so we can colour each point. */}
            <Scatter
              yAxisId="px"
              data={buyFills.map((f) => ({ t: f.t, price: f.price, units: f.units }))}
              shape={renderTriangleUp}
              dataKey="price"
              name="buy fills"
              isAnimationActive={false}
            />
            <Scatter
              yAxisId="px"
              data={sellFills.map((f) => ({ t: f.t, price: f.price, units: f.units }))}
              shape={renderTriangleDown}
              dataKey="price"
              name="sell fills"
              isAnimationActive={false}
            />
            {/* Optimizer label markers — ideal long/short entries from
                the rolling label optimizer. Diamonds so they're visually
                distinct from the live-fill triangles above. */}
            <Scatter
              yAxisId="px"
              data={longLabels.map((m) => ({ t: m.t, price: m.price }))}
              shape={renderDiamondLong}
              dataKey="price"
              name="ideal long"
              isAnimationActive={false}
            />
            <Scatter
              yAxisId="px"
              data={shortLabels.map((m) => ({ t: m.t, price: m.price }))}
              shape={renderDiamondShort}
              dataKey="price"
              name="ideal short"
              isAnimationActive={false}
            />
            {/* Latest tick marker — emphasised dot at the right edge. */}
            {tick && priceData.length > 0 && (
              <ReferenceLine
                yAxisId="px"
                y={tick.mid}
                stroke={venueColor}
                strokeDasharray="4 4"
                strokeOpacity={0.4}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Confidence chart — small strip at the bottom */}
      <div className="px-2 pb-2 pt-1 border-t border-ink-800" style={{ height: 140 }}>
        <div className="flex items-center justify-between px-2 pb-1 text-[10px] uppercase tracking-widest text-slate-500">
          <span>signed model confidence</span>
          <span>
            threshold ±{(CONFIDENCE_THRESHOLD * 100).toFixed(0)}% · long+ / short−
          </span>
        </div>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={confData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
            <CartesianGrid stroke="#1c2030" vertical={false} />
            <XAxis
              dataKey="t"
              type="number"
              domain={domain}
              stroke="#475569"
              tick={{ fontSize: 10 }}
              tickFormatter={tsFmt}
              minTickGap={64}
            />
            <YAxis
              domain={[-1, 1]}
              stroke="#475569"
              tick={{ fontSize: 10 }}
              width={48}
              orientation="right"
              tickFormatter={(v) => `${(v as number).toFixed(1)}`}
            />
            <ReferenceLine y={CONFIDENCE_THRESHOLD} stroke="#22c55e" strokeDasharray="3 3" />
            <ReferenceLine y={-CONFIDENCE_THRESHOLD} stroke="#ef4444" strokeDasharray="3 3" />
            <ReferenceLine y={0} stroke="#334155" />
            <Tooltip
              contentStyle={{
                background: "#0d0f15",
                border: "1px solid #262b3d",
                borderRadius: 6,
                fontSize: 11,
              }}
              labelFormatter={(l) => tsFmt(l as number)}
              formatter={(v: unknown) => {
                const n = typeof v === "number" ? v : Number(v);
                return [Number.isFinite(n) ? n.toFixed(2) : String(v), "signed conf"];
              }}
            />
            <Line
              type="stepAfter"
              dataKey="signed"
              stroke="#a78bfa"
              dot={false}
              strokeWidth={1.5}
              isAnimationActive={false}
              name="signed conf"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="px-4 py-2 border-t border-ink-800 flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-500 font-mono">
        <span>
          {history.length} ticks · {buyFills.length + sellFills.length} fills in window ·{" "}
          {confData.length} signals
          {longLabels.length + shortLabels.length > 0 && (
            <>
              {" "}· {longLabels.length + shortLabels.length} ideal entries
            </>
          )}
        </span>
        <span>
          {!fitness
            ? "model not trained yet — fills come from manual orders"
            : isStrong
              ? "model is producing tradeable signals"
              : "model still warming up — signals may not fire yet"}
        </span>
      </div>
    </div>
  );
}

// Recharts custom Scatter shapes. cx/cy come in as pixel coords; recharts
// types these as unknown, so we cast inside the body.
function renderTriangleUp(props: unknown): JSX.Element {
  const { cx, cy } = props as { cx?: number; cy?: number };
  if (cx == null || cy == null) return <g />;
  return (
    <polygon
      points={`${cx},${cy - 7} ${cx - 6},${cy + 5} ${cx + 6},${cy + 5}`}
      fill="#22c55e"
      stroke="#0d0f15"
      strokeWidth={1}
    />
  );
}

function renderTriangleDown(props: unknown): JSX.Element {
  const { cx, cy } = props as { cx?: number; cy?: number };
  if (cx == null || cy == null) return <g />;
  return (
    <polygon
      points={`${cx},${cy + 7} ${cx - 6},${cy - 5} ${cx + 6},${cy - 5}`}
      fill="#ef4444"
      stroke="#0d0f15"
      strokeWidth={1}
    />
  );
}

// Diamond shapes for the optimizer's ideal-entry overlay. Hollow fill +
// thick stroke so they read as "research target" not "executed fill".
function renderDiamondLong(props: unknown): JSX.Element {
  const { cx, cy } = props as { cx?: number; cy?: number };
  if (cx == null || cy == null) return <g />;
  return (
    <polygon
      points={`${cx},${cy - 5} ${cx + 5},${cy} ${cx},${cy + 5} ${cx - 5},${cy}`}
      fill="#0d0f15"
      stroke="#22c55e"
      strokeWidth={2}
    />
  );
}

function renderDiamondShort(props: unknown): JSX.Element {
  const { cx, cy } = props as { cx?: number; cy?: number };
  if (cx == null || cy == null) return <g />;
  return (
    <polygon
      points={`${cx},${cy - 5} ${cx + 5},${cy} ${cx},${cy + 5} ${cx - 5},${cy}`}
      fill="#0d0f15"
      stroke="#ef4444"
      strokeWidth={2}
    />
  );
}

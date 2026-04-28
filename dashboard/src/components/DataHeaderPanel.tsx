import { useMemo, useState } from "react";

import {
  CRYPTO_ONLY_FEATURE_START,
  FEATURE_COUNT,
  FEATURE_NAMES,
  FIRST_MODEL_TICK_TARGET,
  RETRAIN_EVERY,
  TRAIN_AFTER,
  WARMUP_TICKS,
} from "../featureSchema";
import { fmtDuration, rateHz } from "../store";
import type { ModelFitness } from "../types";
import { instrumentVenue } from "../types";
import { Card } from "./Card";

interface Props {
  instruments: string[];
  priceCounts: Record<string, number>;
  featureCounts: Record<string, number>;
  signalCounts: Record<string, number>;
  fillCounts: Record<string, number>;
  priceTimes: Record<string, number[]>;
  fitness: Record<string, ModelFitness>;
}

/**
 * Top-of-page summary: per-instrument data totals + countdown to the
 * first trained model + scorecard for the latest model. Pairs are
 * sorted by venue (crypto first) then alphabetically. Click any row
 * to expand its full ModelFitness scorecard.
 */
export function DataHeaderPanel({
  instruments,
  priceCounts,
  featureCounts,
  signalCounts,
  fillCounts,
  priceTimes,
  fitness,
}: Props) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [showSchema, setShowSchema] = useState(false);

  const ordered = useMemo(() => {
    const crypto = instruments.filter((i) => instrumentVenue(i) === "alpaca");
    const fx = instruments.filter((i) => instrumentVenue(i) === "oanda");
    return [...crypto.sort(), ...fx.sort()];
  }, [instruments]);

  const toggle = (inst: string) =>
    setExpanded((m) => ({ ...m, [inst]: !m[inst] }));

  return (
    <Card
      title="Data pipeline"
      right={
        <div className="flex items-center gap-3 text-[11px] text-slate-400">
          <span>
            warmup {WARMUP_TICKS} · train@{TRAIN_AFTER} · retrain/{RETRAIN_EVERY}
          </span>
          <button
            type="button"
            onClick={() => setShowSchema((v) => !v)}
            className="px-2 py-1 rounded border border-ink-700 hover:border-ink-600 hover:bg-ink-800/60"
          >
            {showSchema ? "Hide" : "Show"} feature schema ({FEATURE_COUNT})
          </button>
        </div>
      }
    >
      {showSchema && <FeatureSchemaList />}

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="text-slate-500">
            <tr className="border-b border-ink-800">
              <th className="text-left px-3 py-2 font-medium">Instrument</th>
              <th className="text-right px-3 py-2 font-medium">Prices</th>
              <th className="text-right px-3 py-2 font-medium">Features</th>
              <th className="text-right px-3 py-2 font-medium">Signals</th>
              <th className="text-right px-3 py-2 font-medium">Fills</th>
              <th className="text-right px-3 py-2 font-medium">Hz</th>
              <th className="text-left px-3 py-2 font-medium">First model</th>
              <th className="text-left px-3 py-2 font-medium">ETA</th>
              <th className="text-left px-3 py-2 font-medium">Last model</th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((inst) => {
              const pCount = priceCounts[inst] ?? 0;
              const fCount = featureCounts[inst] ?? 0;
              const sCount = signalCounts[inst] ?? 0;
              const fillCount = fillCounts[inst] ?? 0;
              const hz = rateHz(priceTimes[inst]);
              const fit = fitness[inst];
              const isExpanded = !!expanded[inst];
              return (
                <InstrumentRow
                  key={inst}
                  instrument={inst}
                  pCount={pCount}
                  fCount={fCount}
                  sCount={sCount}
                  fillCount={fillCount}
                  hz={hz}
                  fitness={fit}
                  expanded={isExpanded}
                  onToggle={() => toggle(inst)}
                />
              );
            })}
            {ordered.length === 0 && (
              <tr>
                <td className="px-3 py-3 text-slate-500" colSpan={9}>
                  Waiting for instrument list…
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function InstrumentRow({
  instrument,
  pCount,
  fCount,
  sCount,
  fillCount,
  hz,
  fitness,
  expanded,
  onToggle,
}: {
  instrument: string;
  pCount: number;
  fCount: number;
  sCount: number;
  fillCount: number;
  hz: number;
  fitness: ModelFitness | undefined;
  expanded: boolean;
  onToggle: () => void;
}) {
  const venue = instrumentVenue(instrument);
  const venueColor = venue === "alpaca" ? "#d946ef" : "#3b82f6";
  const isTrained = !!fitness;
  const remaining = Math.max(0, FIRST_MODEL_TICK_TARGET - pCount);
  const etaSec = hz > 0 && remaining > 0 ? remaining / hz : null;
  const etaText = isTrained ? "trained" : hz === 0 ? "—" : fmtDuration(etaSec);

  // Two-stage progress: warmup [0, 300] then features [300, 1300].
  const warmupPct = Math.min(100, (Math.min(pCount, WARMUP_TICKS) / WARMUP_TICKS) * 100);
  const featPct =
    Math.max(0, Math.min(pCount - WARMUP_TICKS, TRAIN_AFTER) / TRAIN_AFTER) * 100;
  const stage = pCount < WARMUP_TICKS ? "warmup" : isTrained ? "ready" : "filling";

  const oosAcc = fitness ? (fitness.oos.accuracy * 100).toFixed(1) : null;
  const oosLL = fitness ? fitness.oos.log_loss.toFixed(2) : null;
  const oosSh = fitness ? fitness.oos.sharpe.toFixed(2) : null;

  return (
    <>
      <tr
        className={`border-b border-ink-800 cursor-pointer transition-colors ${
          expanded ? "bg-ink-800/40" : "hover:bg-ink-800/30"
        }`}
        onClick={onToggle}
      >
        <td className="px-3 py-2 font-mono">
          <div className="flex items-center gap-2">
            <span
              className="inline-block px-1.5 py-0.5 text-[9px] tracking-widest rounded border"
              style={{ color: venueColor, borderColor: `${venueColor}66`, background: `${venueColor}1a` }}
            >
              {venue.toUpperCase()}
            </span>
            <span className="text-slate-200">{instrument.replace("_", "/")}</span>
            <span className="text-slate-600 text-[10px]">{expanded ? "▼" : "▸"}</span>
          </div>
        </td>
        <td className="px-3 py-2 font-mono text-right text-slate-300">
          {pCount.toLocaleString()}
        </td>
        <td className="px-3 py-2 font-mono text-right text-slate-300">
          {fCount.toLocaleString()}
        </td>
        <td className="px-3 py-2 font-mono text-right text-slate-300">
          {sCount.toLocaleString()}
        </td>
        <td className="px-3 py-2 font-mono text-right text-slate-300">
          {fillCount.toLocaleString()}
        </td>
        <td className="px-3 py-2 font-mono text-right text-slate-400">
          {hz > 0 ? hz.toFixed(1) : "—"}
        </td>
        <td className="px-3 py-2 min-w-[180px]">
          <ProgressBar warmupPct={warmupPct} featPct={featPct} stage={stage} />
          <div className="text-[10px] text-slate-500 font-mono mt-0.5">
            {pCount.toLocaleString()} / {FIRST_MODEL_TICK_TARGET.toLocaleString()} ·{" "}
            {stage}
          </div>
        </td>
        <td className="px-3 py-2 font-mono text-slate-300 whitespace-nowrap">
          {etaText}
        </td>
        <td className="px-3 py-2 font-mono">
          {fitness ? (
            <div className="flex items-center gap-2">
              <span className="text-slate-300">v{fitness.model_version}</span>
              <span
                className={
                  fitness.oos.accuracy >= 0.5
                    ? "text-bull"
                    : fitness.oos.accuracy >= 0.4
                      ? "text-slate-300"
                      : "text-bear"
                }
              >
                OOS {oosAcc}%
              </span>
              <span className="text-slate-500">LL {oosLL}</span>
              <span className="text-slate-500">Sh {oosSh}</span>
            </div>
          ) : (
            <span className="text-slate-500 italic">untrained</span>
          )}
        </td>
      </tr>
      {expanded && (
        <tr className="border-b border-ink-800 bg-ink-900/40">
          <td colSpan={9} className="px-3 py-3">
            <FitnessDetails fitness={fitness} pCount={pCount} fCount={fCount} />
          </td>
        </tr>
      )}
    </>
  );
}

function ProgressBar({
  warmupPct,
  featPct,
  stage,
}: {
  warmupPct: number;
  featPct: number;
  stage: "warmup" | "filling" | "ready";
}) {
  // Warmup occupies the first ~23% of the track (300/1300), feature
  // fill the remaining 77% (1000/1300).
  const warmupShare = (WARMUP_TICKS / FIRST_MODEL_TICK_TARGET) * 100;
  const featShare = (TRAIN_AFTER / FIRST_MODEL_TICK_TARGET) * 100;
  return (
    <div className="relative w-full h-2 rounded-full overflow-hidden bg-ink-800 border border-ink-700">
      {/* Warmup section */}
      <div
        className="absolute inset-y-0 left-0"
        style={{
          width: `${warmupShare}%`,
          background:
            stage === "warmup"
              ? `linear-gradient(90deg, #f59e0b ${warmupPct}%, transparent ${warmupPct}%)`
              : "#92400e",
          opacity: stage === "warmup" ? 1 : 0.6,
        }}
      />
      {/* Feature-fill section */}
      <div
        className="absolute inset-y-0"
        style={{
          left: `${warmupShare}%`,
          width: `${featShare}%`,
          background:
            stage === "filling"
              ? `linear-gradient(90deg, #3b82f6 ${featPct}%, transparent ${featPct}%)`
              : stage === "ready"
                ? "#22c55e"
                : "transparent",
        }}
      />
      {/* Section divider */}
      <div
        className="absolute inset-y-0"
        style={{ left: `${warmupShare}%`, width: 1, background: "#1c2030" }}
      />
    </div>
  );
}

function FitnessDetails({
  fitness,
  pCount,
  fCount,
}: {
  fitness: ModelFitness | undefined;
  pCount: number;
  fCount: number;
}) {
  if (!fitness) {
    const remaining = Math.max(0, FIRST_MODEL_TICK_TARGET - pCount);
    return (
      <div className="text-xs text-slate-400 leading-relaxed font-mono">
        <div>
          No model trained yet. Need {remaining.toLocaleString()} more price
          ticks (currently {pCount.toLocaleString()} / {FIRST_MODEL_TICK_TARGET.toLocaleString()},{" "}
          {fCount.toLocaleString()} feature samples post-warmup).
        </div>
        <div className="text-slate-500 mt-1">
          After the first model trains it will retrain every {RETRAIN_EVERY}{" "}
          additional features.
        </div>
      </div>
    );
  }
  const trained = new Date(fitness.trained_at).toLocaleString();
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-[11px] font-mono">
      <div>
        <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">
          Model
        </div>
        <div className="text-slate-200">{fitness.model_id}</div>
        <div className="text-slate-500">version {fitness.model_version}</div>
        <div className="text-slate-500">trained {trained}</div>
        <div className="text-slate-500 mt-1">
          samples seen {fitness.samples_seen.toLocaleString()}
        </div>
        <div className="text-slate-500">
          windows train [{fitness.train_window[0]}, {fitness.train_window[1]}) · oos [
          {fitness.oos_window[0]}, {fitness.oos_window[1]})
        </div>
      </div>
      <ScoreBlock label="Train" m={fitness.train} />
      <ScoreBlock label="OOS" m={fitness.oos} highlight />
    </div>
  );
}

function ScoreBlock({
  label,
  m,
  highlight,
}: {
  label: string;
  m: ModelFitness["train"];
  highlight?: boolean;
}) {
  const accCls = highlight
    ? m.accuracy >= 0.5
      ? "text-bull"
      : m.accuracy >= 0.4
        ? "text-slate-200"
        : "text-bear"
    : "text-slate-200";
  const total = m.class_distribution.reduce((a, b) => a + b, 0) || 1;
  const dist = m.class_distribution.map((c) => `${((c / total) * 100).toFixed(0)}%`);
  return (
    <div>
      <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">
        {label}
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
        <span className="text-slate-500">samples</span>
        <span className="text-slate-200 text-right">{m.samples.toLocaleString()}</span>
        <span className="text-slate-500">accuracy</span>
        <span className={`${accCls} text-right`}>{(m.accuracy * 100).toFixed(2)}%</span>
        <span className="text-slate-500">log loss</span>
        <span className="text-slate-200 text-right">{m.log_loss.toFixed(3)}</span>
        <span className="text-slate-500">sharpe</span>
        <span className="text-slate-200 text-right">{m.sharpe.toFixed(2)}</span>
        <span className="text-slate-500 col-span-2 mt-1 text-[10px]">predicted classes (long / short)</span>
        <span className="col-span-2 text-right text-slate-300">
          {m.class_distribution[0]} / {m.class_distribution[2]}{" "}
          <span className="text-slate-500">
            ({dist[0]} / {dist[2]})
          </span>
        </span>
      </div>
    </div>
  );
}

function FeatureSchemaList() {
  return (
    <div className="border-b border-ink-800 pb-3 mb-3">
      <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">
        Feature schema ({FEATURE_COUNT} columns; indices ≥{CRYPTO_ONLY_FEATURE_START}{" "}
        are crypto-only)
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-x-4 gap-y-1 text-[11px] font-mono">
        {FEATURE_NAMES.map((name, i) => (
          <div key={name} className="flex items-baseline gap-2">
            <span className="text-slate-600 w-6 text-right">{i}</span>
            <span className="text-slate-300 truncate">{name}</span>
            {i >= CRYPTO_ONLY_FEATURE_START && (
              <span className="text-[9px] text-fuchsia-400 ml-auto">crypto</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

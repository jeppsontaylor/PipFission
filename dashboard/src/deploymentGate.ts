// Pure helpers for the DeploymentGateCard. Parses the threshold JSON
// the gate persists alongside each evaluation, computes the per-floor
// pass/fail status, and produces the bar-chart "delta" so the card
// can show how close (or far) each metric was to its floor.

import type { ModelDeploymentGateRow } from "./mlApi";

export interface GateThresholds {
  min_oos_auc: number;
  max_oos_log_loss: number;
  min_oos_balanced_acc: number;
  min_fine_tune_sortino: number;
  max_fine_tune_dd_bp: number;
  require_lockbox_pass: boolean;
}

export interface GateFloorResult {
  /** Display label, e.g. "OOS AUC". */
  label: string;
  /** Actual measured value. */
  actual: number;
  /** Floor value. */
  floor: number;
  /** True iff the metric passed its floor. */
  passed: boolean;
  /** "≥" or "≤" — direction of the constraint, for tooltip. */
  direction: "ge" | "le";
  /** Distance the actual is from the floor as a fraction of the floor.
   *  Positive = above floor (pass), negative = below floor (fail). */
  fractionOver: number;
}

const DEFAULTS: GateThresholds = {
  min_oos_auc: 0.55,
  max_oos_log_loss: 0.7,
  min_oos_balanced_acc: 0.52,
  min_fine_tune_sortino: 0.3,
  max_fine_tune_dd_bp: 1500,
  require_lockbox_pass: true,
};

export function parseThresholds(json: string): GateThresholds {
  try {
    const parsed = JSON.parse(json);
    if (parsed && typeof parsed === "object") {
      return { ...DEFAULTS, ...parsed };
    }
  } catch {
    /* fall through */
  }
  return { ...DEFAULTS };
}

export function evaluateFloors(row: ModelDeploymentGateRow): GateFloorResult[] {
  const t = parseThresholds(row.gate_thresholds_json);
  const out: GateFloorResult[] = [
    {
      label: "OOS AUC",
      actual: row.oos_auc,
      floor: t.min_oos_auc,
      passed: row.oos_auc >= t.min_oos_auc,
      direction: "ge",
      fractionOver: t.min_oos_auc > 0
        ? (row.oos_auc - t.min_oos_auc) / t.min_oos_auc
        : 0,
    },
    {
      label: "OOS log loss",
      actual: row.oos_log_loss,
      floor: t.max_oos_log_loss,
      passed: row.oos_log_loss <= t.max_oos_log_loss,
      direction: "le",
      fractionOver: t.max_oos_log_loss > 0
        ? (t.max_oos_log_loss - row.oos_log_loss) / t.max_oos_log_loss
        : 0,
    },
    {
      label: "OOS balanced acc",
      actual: row.oos_balanced_acc,
      floor: t.min_oos_balanced_acc,
      passed: row.oos_balanced_acc >= t.min_oos_balanced_acc,
      direction: "ge",
      fractionOver: t.min_oos_balanced_acc > 0
        ? (row.oos_balanced_acc - t.min_oos_balanced_acc) / t.min_oos_balanced_acc
        : 0,
    },
    {
      label: "Fine-tune Sortino",
      actual: row.fine_tune_sortino,
      floor: t.min_fine_tune_sortino,
      passed: row.fine_tune_sortino >= t.min_fine_tune_sortino,
      direction: "ge",
      fractionOver: t.min_fine_tune_sortino > 0
        ? (row.fine_tune_sortino - t.min_fine_tune_sortino) / t.min_fine_tune_sortino
        : 0,
    },
    {
      label: "Fine-tune max DD (bp)",
      actual: row.fine_tune_max_dd_bp,
      floor: t.max_fine_tune_dd_bp,
      passed: row.fine_tune_max_dd_bp <= t.max_fine_tune_dd_bp,
      direction: "le",
      fractionOver: t.max_fine_tune_dd_bp > 0
        ? (t.max_fine_tune_dd_bp - row.fine_tune_max_dd_bp) / t.max_fine_tune_dd_bp
        : 0,
    },
  ];
  return out;
}

export function splitBlockedReasons(s: string): string[] {
  return s
    .split(";")
    .map((p) => p.trim())
    .filter((p) => p.length > 0);
}

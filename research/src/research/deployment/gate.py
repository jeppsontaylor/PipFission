"""Deployment quality gate.

Sits between `train.side` (chooses winner from the model zoo) and
`export.champion` (writes ONNX + flips the live symlink). Evaluates
the would-be champion against a set of performance floors. If ANY
floor fails, the export step is skipped and the prior champion stays
live — the system never deploys a model that's quantifiably worse
than the bar we set.

Thresholds default to a conservative "no-skill defence":
  * `min_oos_auc` 0.55  — beat coin-flip + tiny edge
  * `max_oos_log_loss` 0.70  — calibrated probabilities
  * `min_oos_balanced_acc` 0.52
  * `min_fine_tune_sortino` 0.30  — must show some return per unit
                                     downside on the next 100 bars
  * `max_fine_tune_dd_bp` 1500  — drawdown ceiling on fine-tune

Each is independently overridable by env. The thresholds are
serialised into the `model_deployment_gate.gate_thresholds_json`
column so dashboard + post-mortem analysts know which bar a given
model was held to.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from research.data.duckdb_io import rw_conn


@dataclass(frozen=True)
class DeploymentGateThresholds:
    min_oos_auc: float = 0.55
    max_oos_log_loss: float = 0.70
    min_oos_balanced_acc: float = 0.52
    min_fine_tune_sortino: float = 0.30
    max_fine_tune_dd_bp: float = 1500.0
    """Whether failing the lockbox auto-fails the gate. Belt-and-suspenders
    — the orchestrator already short-circuits export on lockbox FAIL,
    but recording the gate decision keeps the audit trail uniform."""
    require_lockbox_pass: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_env(cls) -> "DeploymentGateThresholds":
        """Override any threshold via env. e.g. MIN_OOS_AUC=0.60.
        Unset env vars use the defaults defined on this dataclass."""
        def _f(name: str, default: float) -> float:
            v = os.environ.get(name)
            if v is None or v == "":
                return default
            try:
                return float(v)
            except ValueError:
                return default

        def _b(name: str, default: bool) -> bool:
            v = os.environ.get(name)
            if v is None:
                return default
            return v.strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            min_oos_auc=_f("MIN_OOS_AUC", 0.55),
            max_oos_log_loss=_f("MAX_OOS_LOG_LOSS", 0.70),
            min_oos_balanced_acc=_f("MIN_OOS_BALANCED_ACC", 0.52),
            min_fine_tune_sortino=_f("MIN_FINE_TUNE_SORTINO", 0.30),
            max_fine_tune_dd_bp=_f("MAX_FINE_TUNE_DD_BP", 1500.0),
            require_lockbox_pass=_b("REQUIRE_LOCKBOX_PASS", True),
        )


DEFAULT_GATE_THRESHOLDS = DeploymentGateThresholds()


@dataclass
class GateResult:
    """Outcome of the deployment gate for one would-be champion."""
    model_id: str
    instrument: str
    ts_ms: int
    passed_gate: bool
    blocked_reasons: list[str] = field(default_factory=list)
    # Snapshot of inputs the gate evaluated, for post-mortem.
    oos_auc: float = 0.0
    oos_log_loss: float = 0.0
    oos_brier: float = 0.0
    oos_balanced_acc: float = 0.0
    fine_tune_sortino: float = 0.0
    fine_tune_max_dd_bp: float = 0.0
    thresholds: DeploymentGateThresholds = field(
        default_factory=DeploymentGateThresholds
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["thresholds"] = self.thresholds.as_dict()
        return d


def evaluate_deployment_gate(
    *,
    model_id: str,
    instrument: str,
    oos_auc: float,
    oos_log_loss: float,
    oos_brier: float,
    oos_balanced_acc: float,
    fine_tune_sortino: float,
    fine_tune_max_dd_bp: float,
    lockbox_passed: bool,
    thresholds: DeploymentGateThresholds | None = None,
    ts_ms: int | None = None,
) -> GateResult:
    """Evaluate the gate for a would-be champion. Returns a GateResult
    with `passed_gate` and the list of failed thresholds.

    The function is pure — it doesn't touch the DB, doesn't write the
    ONNX, doesn't print anything. Callers persist the result via
    `persist_gate_result`.
    """
    th = thresholds or DEFAULT_GATE_THRESHOLDS
    reasons: list[str] = []
    if oos_auc < th.min_oos_auc:
        reasons.append(
            f"oos_auc {oos_auc:.3f} < {th.min_oos_auc:.3f}"
        )
    if oos_log_loss > th.max_oos_log_loss:
        reasons.append(
            f"oos_log_loss {oos_log_loss:.3f} > {th.max_oos_log_loss:.3f}"
        )
    if oos_balanced_acc < th.min_oos_balanced_acc:
        reasons.append(
            f"oos_balanced_acc {oos_balanced_acc:.3f} < {th.min_oos_balanced_acc:.3f}"
        )
    if fine_tune_sortino < th.min_fine_tune_sortino:
        reasons.append(
            f"fine_tune_sortino {fine_tune_sortino:.3f} < {th.min_fine_tune_sortino:.3f}"
        )
    if fine_tune_max_dd_bp > th.max_fine_tune_dd_bp:
        reasons.append(
            f"fine_tune_max_dd_bp {fine_tune_max_dd_bp:.1f} > {th.max_fine_tune_dd_bp:.1f}"
        )
    if th.require_lockbox_pass and not lockbox_passed:
        reasons.append("lockbox did not pass")

    return GateResult(
        model_id=model_id,
        instrument=instrument,
        ts_ms=ts_ms if ts_ms is not None else int(time.time() * 1000),
        passed_gate=len(reasons) == 0,
        blocked_reasons=reasons,
        oos_auc=oos_auc,
        oos_log_loss=oos_log_loss,
        oos_brier=oos_brier,
        oos_balanced_acc=oos_balanced_acc,
        fine_tune_sortino=fine_tune_sortino,
        fine_tune_max_dd_bp=fine_tune_max_dd_bp,
        thresholds=th,
    )


def persist_gate_result(result: GateResult) -> None:
    """Write the gate result to `model_deployment_gate`. Idempotent
    via UNIQUE(model_id); re-running for the same model is a no-op
    rather than an error."""
    with rw_conn() as conn:
        conn.execute(
            """
            INSERT INTO model_deployment_gate
              (model_id, instrument, ts_ms, oos_auc, oos_log_loss,
               oos_brier, oos_balanced_acc, fine_tune_sortino,
               fine_tune_max_dd_bp, passed_gate, blocked_reasons,
               gate_thresholds_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO NOTHING
            """,
            [
                result.model_id, result.instrument, result.ts_ms,
                result.oos_auc, result.oos_log_loss, result.oos_brier,
                result.oos_balanced_acc, result.fine_tune_sortino,
                result.fine_tune_max_dd_bp, result.passed_gate,
                "; ".join(result.blocked_reasons),
                json.dumps(result.thresholds.as_dict()),
            ],
        )

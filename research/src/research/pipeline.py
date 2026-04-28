"""End-to-end pipeline orchestrator.

Sequences `label → train.side → finetune → lockbox → export.champion`
in-process, threading `model_id` / `params_id` between steps so the
operator doesn't have to copy-paste CLI args.

Each underlying step still calls `track_run` for fine-grained
visibility, so a dashboard pipeline view sees one parent
`pipeline.full` row plus five child rows. If a step fails, the parent
exception propagates and the parent row is marked `failed` with the
underlying error.

Lockbox is treated as a *gate*: a failing lockbox does not raise, but
it does skip the export step (the prior champion stays live). The
return value records what actually ran and whether the lockbox passed
so callers can act on it.

Idempotency: each underlying writer uses ON CONFLICT clauses or
generates a fresh id, so re-running the pipeline against the same
window doesn't fight existing rows. The rolling 10k retention sweep on
the Rust side handles cleanup.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research.data.extract import extract_bars_10s
from research.deployment.gate import (
    DeploymentGateThresholds,
    evaluate_deployment_gate,
    persist_gate_result,
)
from research.export.manifest import write_manifest
from research.export.onnx_export import export_calibrated_to_onnx, verify_onnx_roundtrip
from research.labeling.label_opt import LabelOptConfig, run_label_opt, write_labels
from research.lockbox.gate import LockboxConfig, seal_lockbox
from research.observability import track_run
from research.paths import PATHS
from research.trader.optimizer import TraderFineTuneConfig, fine_tune_trader
from research.training.side_train import SideTrainConfig, train_side


@dataclass
class PipelineConfig:
    """Tunables for one pipeline run.

    Defaults match the documented walk-forward training rhythm: 1000
    bars to train + 100 bars to fine-tune trader + 100 bars reserved
    for the lockbox gate.
    """

    instrument: str

    # Label optimiser
    n_bars: int = 1000
    sigma_span: int = 60
    cusum_h_mult: float = 2.0
    pt_atr: float = 2.0
    sl_atr: float = 2.0
    vert_horizon: int = 36
    # 0 — disabling the label-side cost floor. Most forex bar-to-bar
    # moves are sub-pip; setting an absolute return floor here
    # double-filters (the trader's `min_edge_after_costs` already
    # accounts for spread + slippage downstream). Verified on 1000
    # EUR_USD bars: edge 0.0005 → 0 chosen, edge 0.0001 → 6, edge 0
    # → 98 well-balanced (~48% minority). 6 was too few for the
    # 6-fold CPCV.
    min_edge: float = 0.0

    # Side classifier
    n_splits: int = 6
    n_test_groups: int = 2
    embargo_pct: float = 0.01
    n_optuna_trials: int = 25
    inner_cv_splits: int = 3
    seed: int = 42
    # Comma-separated candidate names (None = full zoo: lgbm, xgb,
    # catboost, logreg, extratrees). The pipeline picks the OOS-log-loss
    # winner.
    candidates: str | None = None

    # Fine-tune
    n_fine_tune: int = 100
    n_trader_trials: int = 50
    cost_stress: float = 1.0

    # Lockbox
    n_seen: int = 1100
    n_lockbox: int = 100
    min_n_trades: int = 3
    min_dsr: float = 0.50
    max_dd_bp_limit: float = 1500.0

    # Export
    # 1e-3 — empirically tree models calibrated via Platt + ONNX zoo
    # exporters (lgbm/xgb/catboost/histgb) sometimes drift to 1.5e-4
    # under fp32. The 1e-4 default was too tight; 1e-3 still catches
    # any meaningful calibration breakage without false positives.
    onnx_atol: float = 1e-3

    # Whether a lockbox failure should still publish the export. False
    # means "lockbox is a gate" — the safer default.
    publish_on_lockbox_fail: bool = False


@dataclass
class PipelineReport:
    """Structured summary of one pipeline run.

    None values mean the step didn't run (typically because an earlier
    step short-circuited or the lockbox gate blocked publishing).
    """

    instrument: str
    parent_run_id: str
    started_ms: int
    finished_ms: int | None = None
    elapsed_ms: int | None = None

    label_run_id: str | None = None
    n_labels: int = 0

    model_id: str | None = None
    oos_auc: float | None = None
    n_oof: int = 0

    params_id: str | None = None
    fine_tune_sortino: float | None = None
    fine_tune_max_dd_bp: float | None = None

    lockbox_run_id: str | None = None
    lockbox_passed: bool | None = None
    lockbox_reasons: list[str] = field(default_factory=list)

    onnx_path: str | None = None
    onnx_sha256: str | None = None
    published: bool = False

    # Deployment gate outcome (Phase A4). `gate_passed=False` means the
    # would-be champion was blocked from going live for one or more
    # quality reasons; `gate_blocked_reasons` lists each failed
    # threshold. The prior live champion stays serving signals.
    gate_passed: bool | None = None
    gate_blocked_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "parent_run_id": self.parent_run_id,
            "started_ms": self.started_ms,
            "finished_ms": self.finished_ms,
            "elapsed_ms": self.elapsed_ms,
            "label_run_id": self.label_run_id,
            "n_labels": self.n_labels,
            "model_id": self.model_id,
            "oos_auc": self.oos_auc,
            "n_oof": self.n_oof,
            "params_id": self.params_id,
            "fine_tune_sortino": self.fine_tune_sortino,
            "fine_tune_max_dd_bp": self.fine_tune_max_dd_bp,
            "lockbox_run_id": self.lockbox_run_id,
            "lockbox_passed": self.lockbox_passed,
            "lockbox_reasons": list(self.lockbox_reasons),
            "onnx_path": self.onnx_path,
            "onnx_sha256": self.onnx_sha256,
            "published": self.published,
            "gate_passed": self.gate_passed,
            "gate_blocked_reasons": list(self.gate_blocked_reasons),
        }


def run_pipeline(cfg: PipelineConfig) -> PipelineReport:
    """Run the full retrain pipeline. Returns a structured report.

    Raises only on hard failures (e.g. no bars in DuckDB, classifier
    blew up). A lockbox FAIL is *not* an exception — it's reflected in
    `report.lockbox_passed = False` and `report.published = False`.
    """
    parent_id = uuid.uuid4().hex
    started_ms = int(time.time() * 1000)
    report = PipelineReport(
        instrument=cfg.instrument,
        parent_run_id=parent_id,
        started_ms=started_ms,
    )

    args = {"instrument": cfg.instrument, "n_bars": cfg.n_bars, "n_fine_tune": cfg.n_fine_tune}
    with track_run("pipeline.full", args, instrument=cfg.instrument):
        # --- 1. Label ---------------------------------------------------------
        with track_run(
            "label",
            {"instrument": cfg.instrument, "n_bars": cfg.n_bars},
            instrument=cfg.instrument,
        ):
            bars = extract_bars_10s(instrument=cfg.instrument, n_recent=cfg.n_bars)
            if bars.is_empty():
                raise RuntimeError(f"no bars in DuckDB for {cfg.instrument}")
            label_payload = run_label_opt(
                bars,
                LabelOptConfig(
                    sigma_span=cfg.sigma_span,
                    cusum_h_mult=cfg.cusum_h_mult,
                    pt_atr=cfg.pt_atr,
                    sl_atr=cfg.sl_atr,
                    vert_horizon=cfg.vert_horizon,
                    min_edge=cfg.min_edge,
                ),
            )
            label_run_id = write_labels(cfg.instrument, label_payload, label_run_id=None)
            report.label_run_id = label_run_id
            report.n_labels = len(label_payload.get("labels", []))

        # --- 2. Train side ---------------------------------------------------
        with track_run(
            "train.side",
            {
                "instrument": cfg.instrument,
                "n_bars": cfg.n_bars,
                "n_optuna_trials": cfg.n_optuna_trials,
                "label_run_id": label_run_id,
            },
            instrument=cfg.instrument,
        ):
            train_res = train_side(
                SideTrainConfig(
                    instrument=cfg.instrument,
                    n_bars=cfg.n_bars,
                    n_splits=cfg.n_splits,
                    n_test_groups=cfg.n_test_groups,
                    embargo_pct=cfg.embargo_pct,
                    n_optuna_trials=cfg.n_optuna_trials,
                    inner_cv_splits=cfg.inner_cv_splits,
                    label_run_id=label_run_id,
                    seed=cfg.seed,
                    candidates=cfg.candidates,
                )
            )
            report.model_id = train_res.model_version
            report.oos_auc = train_res.oos_auc
            report.n_oof = train_res.n_oof

        # --- 3. Fine-tune trader ---------------------------------------------
        with track_run(
            "finetune",
            {
                "instrument": cfg.instrument,
                "model_id": train_res.model_version,
                "n_fine_tune": cfg.n_fine_tune,
                "n_trials": cfg.n_trader_trials,
            },
            instrument=cfg.instrument,
        ):
            ft = fine_tune_trader(
                TraderFineTuneConfig(
                    instrument=cfg.instrument,
                    model_id=train_res.model_version,
                    n_train=cfg.n_bars,
                    n_fine_tune=cfg.n_fine_tune,
                    n_trials=cfg.n_trader_trials,
                    seed=cfg.seed,
                    cost_stress=cfg.cost_stress,
                )
            )
            report.params_id = ft["params_id"]
            ft_summary = ft.get("fine_tune", {})
            report.fine_tune_sortino = float(ft_summary.get("sortino", 0.0))
            report.fine_tune_max_dd_bp = float(ft_summary.get("max_drawdown_bp", 0.0))

        # --- 4. Lockbox ------------------------------------------------------
        with track_run(
            "lockbox",
            {
                "instrument": cfg.instrument,
                "model_id": train_res.model_version,
                "params_id": ft["params_id"],
            },
            instrument=cfg.instrument,
        ):
            lock = seal_lockbox(
                LockboxConfig(
                    instrument=cfg.instrument,
                    model_id=train_res.model_version,
                    params_id=ft["params_id"],
                    n_seen=cfg.n_seen,
                    n_lockbox=cfg.n_lockbox,
                    cost_stress=cfg.cost_stress,
                    min_n_trades=cfg.min_n_trades,
                    min_dsr=cfg.min_dsr,
                    max_dd_bp_limit=cfg.max_dd_bp_limit,
                )
            )
            report.lockbox_run_id = lock.run_id
            report.lockbox_passed = lock.passed
            report.lockbox_reasons = list(lock.reasons)

        # --- 5. Deployment quality gate -------------------------------------
        # Evaluate the would-be champion against performance floors
        # (env-overridable via MIN_OOS_AUC / MAX_OOS_LOG_LOSS / etc).
        # If ANY floor fails, the export is skipped — prior champion
        # stays live. This is the "stop weak models from going live"
        # safeguard the operator asked for.
        gate_thresholds = DeploymentGateThresholds.from_env()
        gate = evaluate_deployment_gate(
            model_id=train_res.model_version,
            instrument=cfg.instrument,
            oos_auc=train_res.oos_auc,
            oos_log_loss=train_res.oos_log_loss,
            oos_brier=train_res.oos_brier,
            oos_balanced_acc=train_res.oos_balanced_acc,
            fine_tune_sortino=report.fine_tune_sortino or 0.0,
            fine_tune_max_dd_bp=report.fine_tune_max_dd_bp or 0.0,
            lockbox_passed=bool(lock.passed),
            thresholds=gate_thresholds,
        )
        try:
            persist_gate_result(gate)
        except Exception as exc:  # noqa: BLE001
            print(f"[gate] failed to persist gate result: {exc}")
        report.gate_passed = gate.passed_gate
        report.gate_blocked_reasons = list(gate.blocked_reasons)

        # Append agent-readable previews to `trade_logs/<v>/...jsonl`.
        # Failures here are logged but never block the pipeline — the
        # JSONL files are for committed-to-git review, not load-bearing.
        try:
            from research.observability.jsonl_log import append as jsonl_append
            duration_secs = (
                (int(time.time() * 1000) - started_ms) / 1000.0
            )
            jsonl_append(
                "training_log.jsonl",
                {
                    "v": "auto",  # filled by jsonl_log.repo_version()
                    "ts_ms": int(time.time() * 1000),
                    "instrument": cfg.instrument,
                    "run_id": parent_id,
                    "model_id": train_res.model_version,
                    "model_kind": getattr(train_res, "spec_name", None),
                    "n_features": int(train_res.n_oof and 24 or 24),
                    "n_train": int(train_res.n_train),
                    "n_oof": int(train_res.n_oof),
                    "oos_auc": float(train_res.oos_auc),
                    "oos_log_loss": float(train_res.oos_log_loss),
                    "oos_brier": float(train_res.oos_brier),
                    "oos_balanced_acc": float(train_res.oos_balanced_acc),
                    "train_sortino": float(getattr(train_res, "train_sortino", 0.0)),
                    "fine_tune_sortino": report.fine_tune_sortino,
                    "fine_tune_max_dd_bp": report.fine_tune_max_dd_bp,
                    "passed_gate": bool(gate.passed_gate),
                    "blocked_reasons": ",".join(gate.blocked_reasons or []),
                    "lockbox_passed": report.lockbox_passed,
                    "candidates": [
                        {
                            "spec": c.get("spec_name"),
                            "oos_log_loss": c.get("oos_log_loss"),
                            "oos_auc": c.get("oos_auc"),
                            "is_winner": c.get("spec_name") == train_res.spec_name,
                        }
                        for c in (train_res.candidates or [])
                    ],
                    "params_id": report.params_id,
                    "duration_secs": duration_secs,
                },
            )
            jsonl_append(
                "deployment_gate.jsonl",
                {
                    "v": "auto",
                    "ts_ms": int(time.time() * 1000),
                    "instrument": cfg.instrument,
                    "model_id": train_res.model_version,
                    "passed": bool(gate.passed_gate),
                    "blocked": list(gate.blocked_reasons or []),
                    "thresholds": gate_thresholds.__dict__
                    if hasattr(gate_thresholds, "__dict__")
                    else {},
                },
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[jsonl] preview write failed: {exc}")

        # --- 6. Export ONNX (gated by lockbox AND deployment gate) ----------
        should_publish = (
            (lock.passed or cfg.publish_on_lockbox_fail)
            and gate.passed_gate
        )
        if should_publish:
            with track_run(
                "export.champion",
                {"model_id": train_res.model_version, "atol": cfg.onnx_atol},
            ):
                info = export_calibrated_to_onnx(train_res.model_version)
                verify_onnx_roundtrip(train_res.model_version, atol=cfg.onnx_atol)
                manifest = write_manifest(
                    train_res.model_version,
                    info["onnx_path"],
                    info["feature_names"],
                )
                report.onnx_path = info["onnx_path"]
                report.onnx_sha256 = manifest.get("sha256")
                report.published = True

    finished_ms = int(time.time() * 1000)
    report.finished_ms = finished_ms
    report.elapsed_ms = finished_ms - started_ms
    return report


def live_champion_dir() -> Path:
    """Return the directory the Rust hot-swap watcher polls for new
    ONNX models. Useful for tests/scripts that want to verify a
    successful pipeline run published artifacts."""
    return PATHS.artifacts_dir / "models" / "live"

"""Tests for the deployment quality gate.

The gate's job is to STOP a low-performance candidate from being
exported as the live champion. These tests pin the priority order
and the env-override surface so a future refactor can't silently
weaken the floors.
"""
from __future__ import annotations

import pytest

from research.deployment.gate import (
    DEFAULT_GATE_THRESHOLDS,
    DeploymentGateThresholds,
    evaluate_deployment_gate,
)


def _good_kwargs(**overrides):
    """Baseline candidate that passes every floor by a comfortable
    margin. Tests override one field at a time to show that floor
    catches the failure."""
    base = dict(
        model_id="test_model_001",
        instrument="EUR_USD",
        oos_auc=0.62,
        oos_log_loss=0.45,
        oos_brier=0.20,
        oos_balanced_acc=0.58,
        fine_tune_sortino=0.55,
        fine_tune_max_dd_bp=400.0,
        lockbox_passed=True,
    )
    base.update(overrides)
    return base


def test_passing_candidate_clears_gate():
    r = evaluate_deployment_gate(**_good_kwargs())
    assert r.passed_gate is True
    assert r.blocked_reasons == []


def test_low_oos_auc_blocks():
    r = evaluate_deployment_gate(**_good_kwargs(oos_auc=0.51))
    assert r.passed_gate is False
    assert any("oos_auc" in s for s in r.blocked_reasons)


def test_high_oos_log_loss_blocks():
    r = evaluate_deployment_gate(**_good_kwargs(oos_log_loss=0.85))
    assert r.passed_gate is False
    assert any("oos_log_loss" in s for s in r.blocked_reasons)


def test_low_balanced_acc_blocks():
    r = evaluate_deployment_gate(**_good_kwargs(oos_balanced_acc=0.50))
    assert r.passed_gate is False
    assert any("oos_balanced_acc" in s for s in r.blocked_reasons)


def test_low_fine_tune_sortino_blocks():
    r = evaluate_deployment_gate(**_good_kwargs(fine_tune_sortino=0.10))
    assert r.passed_gate is False
    assert any("fine_tune_sortino" in s for s in r.blocked_reasons)


def test_high_fine_tune_drawdown_blocks():
    r = evaluate_deployment_gate(**_good_kwargs(fine_tune_max_dd_bp=2500.0))
    assert r.passed_gate is False
    assert any("fine_tune_max_dd_bp" in s for s in r.blocked_reasons)


def test_failed_lockbox_blocks_when_required():
    """Default thresholds require lockbox pass; failed lockbox is
    auto-failure for the gate even when every other floor clears."""
    r = evaluate_deployment_gate(**_good_kwargs(lockbox_passed=False))
    assert r.passed_gate is False
    assert any("lockbox" in s for s in r.blocked_reasons)


def test_failed_lockbox_can_be_disabled():
    """If `require_lockbox_pass=False`, a lockbox FAIL doesn't auto-fail
    the deployment gate. The orchestrator still skips export on lockbox
    fail elsewhere — this is just for tests / development."""
    th = DeploymentGateThresholds(require_lockbox_pass=False)
    r = evaluate_deployment_gate(**_good_kwargs(lockbox_passed=False), thresholds=th)
    assert r.passed_gate is True


def test_multiple_failures_all_listed():
    r = evaluate_deployment_gate(
        **_good_kwargs(oos_auc=0.40, fine_tune_sortino=0.0)
    )
    assert r.passed_gate is False
    assert len(r.blocked_reasons) >= 2
    joined = "; ".join(r.blocked_reasons)
    assert "oos_auc" in joined and "fine_tune_sortino" in joined


def test_thresholds_from_env(monkeypatch):
    monkeypatch.setenv("MIN_OOS_AUC", "0.65")
    monkeypatch.setenv("MAX_OOS_LOG_LOSS", "0.50")
    monkeypatch.setenv("REQUIRE_LOCKBOX_PASS", "false")
    th = DeploymentGateThresholds.from_env()
    assert th.min_oos_auc == 0.65
    assert th.max_oos_log_loss == 0.50
    assert th.require_lockbox_pass is False
    # Other thresholds keep defaults.
    assert th.min_oos_balanced_acc == DEFAULT_GATE_THRESHOLDS.min_oos_balanced_acc


def test_thresholds_env_garbage_falls_back_to_default(monkeypatch):
    """A malformed env var doesn't blow up the gate — falls back
    to the dataclass default rather than refusing to deploy."""
    monkeypatch.setenv("MIN_OOS_AUC", "not-a-number")
    th = DeploymentGateThresholds.from_env()
    assert th.min_oos_auc == DEFAULT_GATE_THRESHOLDS.min_oos_auc


def test_gate_result_dict_includes_thresholds():
    r = evaluate_deployment_gate(**_good_kwargs())
    d = r.to_dict()
    assert d["passed_gate"] is True
    assert isinstance(d["thresholds"], dict)
    assert "min_oos_auc" in d["thresholds"]

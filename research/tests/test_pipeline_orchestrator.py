"""End-to-end test for the in-process pipeline orchestrator.

Mirrors `test_full_pipeline.py` but invokes the single
`research.pipeline.run_pipeline` entry point. Verifies that the
parent `pipeline.full` run-row is recorded and that ONNX artifacts
land in the live champion directory when the lockbox gate passes.
"""
from __future__ import annotations

import math

import duckdb
import pytest

from research.paths import PATHS

if not PATHS.label_opt_bin.exists() or not PATHS.trader_backtest_bin.exists():
    pytest.skip("Rust binaries missing", allow_module_level=True)

from research.data.duckdb_io import open_ro, open_rw  # noqa: E402
from research.pipeline import PipelineConfig, live_champion_dir, run_pipeline  # noqa: E402

INSTRUMENT = "TEST_PIPE"

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars_10s (
    instrument     VARCHAR  NOT NULL,
    ts_ms          BIGINT   NOT NULL,
    open           DOUBLE   NOT NULL,
    high           DOUBLE   NOT NULL,
    low            DOUBLE   NOT NULL,
    close          DOUBLE   NOT NULL,
    n_ticks        BIGINT   NOT NULL,
    spread_bp_avg  DOUBLE   NOT NULL
);
CREATE TABLE IF NOT EXISTS labels (
    instrument     VARCHAR  NOT NULL,
    ts_ms          BIGINT   NOT NULL,
    t1_ms          BIGINT   NOT NULL,
    side           TINYINT  NOT NULL,
    meta_y         TINYINT  NOT NULL,
    realized_r     DOUBLE   NOT NULL,
    barrier_hit    VARCHAR  NOT NULL,
    oracle_score   DOUBLE   NOT NULL,
    label_run_id   VARCHAR  NOT NULL
);
CREATE TABLE IF NOT EXISTS oof_predictions (
    instrument      VARCHAR  NOT NULL,
    ts_ms           BIGINT   NOT NULL,
    fold            INTEGER  NOT NULL,
    p_long          DOUBLE   NOT NULL,
    p_short         DOUBLE   NOT NULL,
    p_take          DOUBLE   NOT NULL,
    calibrated_p    DOUBLE   NOT NULL,
    model_version   VARCHAR  NOT NULL
);
CREATE TABLE IF NOT EXISTS model_metrics (
    model_id           VARCHAR  NOT NULL,
    instrument         VARCHAR  NOT NULL,
    ts_ms              BIGINT   NOT NULL,
    oos_auc            DOUBLE   NOT NULL,
    oos_log_loss       DOUBLE   NOT NULL,
    oos_brier          DOUBLE   NOT NULL,
    oos_balanced_acc   DOUBLE   NOT NULL,
    calib_slope        DOUBLE   NOT NULL,
    calib_intercept    DOUBLE   NOT NULL,
    train_sharpe       DOUBLE   NOT NULL,
    train_sortino      DOUBLE   NOT NULL,
    max_train_sortino  DOUBLE   NOT NULL,
    max_train_sharpe   DOUBLE   NOT NULL,
    n_train            BIGINT   NOT NULL,
    n_oof              BIGINT   NOT NULL,
    UNIQUE(model_id)
);
CREATE TABLE IF NOT EXISTS optimizer_trials (
    study_id        VARCHAR  NOT NULL,
    trial_id        BIGINT   NOT NULL,
    ts_ms           BIGINT   NOT NULL,
    params_json     VARCHAR  NOT NULL,
    score           DOUBLE   NOT NULL,
    sortino         DOUBLE   NOT NULL,
    max_dd_bp       DOUBLE   NOT NULL,
    turnover        DOUBLE   NOT NULL,
    pareto_rank     INTEGER  NOT NULL
);
CREATE TABLE IF NOT EXISTS trader_metrics (
    params_id              VARCHAR  NOT NULL,
    model_id               VARCHAR  NOT NULL,
    ts_ms                  BIGINT   NOT NULL,
    in_sample_sharpe       DOUBLE   NOT NULL,
    in_sample_sortino      DOUBLE   NOT NULL,
    fine_tune_sharpe       DOUBLE   NOT NULL,
    fine_tune_sortino      DOUBLE   NOT NULL,
    max_dd_bp              DOUBLE   NOT NULL,
    turnover_per_day       DOUBLE   NOT NULL,
    hit_rate               DOUBLE   NOT NULL,
    profit_factor          DOUBLE   NOT NULL,
    n_trades               BIGINT   NOT NULL,
    params_json            VARCHAR  NOT NULL,
    UNIQUE(params_id)
);
CREATE TABLE IF NOT EXISTS lockbox_results (
    run_id        VARCHAR  NOT NULL,
    ts_ms         BIGINT   NOT NULL,
    model_id      VARCHAR  NOT NULL,
    params_id     VARCHAR  NOT NULL,
    summary_json  VARCHAR  NOT NULL,
    sealed        BOOLEAN  NOT NULL,
    UNIQUE(run_id)
);
CREATE TABLE IF NOT EXISTS model_artifacts (
    model_id     VARCHAR  NOT NULL,
    ts_ms        BIGINT   NOT NULL,
    kind         VARCHAR  NOT NULL,
    version      VARCHAR  NOT NULL,
    onnx_blob    BLOB     NOT NULL,
    sha256       VARCHAR  NOT NULL,
    calib_json   VARCHAR  NOT NULL,
    UNIQUE(model_id)
);
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          VARCHAR  NOT NULL,
    command         VARCHAR  NOT NULL,
    instrument      VARCHAR,
    args_json       VARCHAR  NOT NULL,
    ts_started_ms   BIGINT   NOT NULL,
    ts_finished_ms  BIGINT,
    status          VARCHAR  NOT NULL,
    elapsed_ms      BIGINT,
    error_msg       VARCHAR,
    UNIQUE(run_id)
);
"""


@pytest.fixture(scope="module")
def seeded():
    c = open_rw()
    try:
        c.execute(SCHEMA)
        # Tables that own an `instrument` column.
        for tbl in ["bars_10s", "labels", "oof_predictions", "model_metrics", "pipeline_runs"]:
            c.execute(f"DELETE FROM {tbl} WHERE instrument = ?", [INSTRUMENT])
        # Tables keyed by model_id / params_id / study_id — broad sweep of
        # any rows our test runs would have produced previously.
        c.execute("DELETE FROM trader_metrics WHERE model_id LIKE 'side_lgbm_%'")
        c.execute("DELETE FROM lockbox_results WHERE model_id LIKE 'side_lgbm_%'")
        c.execute("DELETE FROM model_artifacts WHERE model_id LIKE 'side_lgbm_%'")
        c.execute("DELETE FROM optimizer_trials WHERE study_id LIKE 'trader_side_lgbm_%'")
        # 1200 bars: 1000 train + 100 fine-tune + 100 lockbox.
        for i in range(1200):
            p = 1.0 + 0.0040 * math.sin(i / 25.0) + 0.0015 * math.cos(i / 7.0)
            hi = p + 4e-4
            lo = p - 4e-4
            c.execute(
                "INSERT INTO bars_10s (instrument, ts_ms, open, high, low, close, n_ticks, spread_bp_avg) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [INSTRUMENT, i * 10_000, p, hi, lo, p, 1, 0.5],
            )
    finally:
        c.close()
    yield INSTRUMENT


def test_run_pipeline_end_to_end(seeded):
    cfg = PipelineConfig(
        instrument=seeded,
        n_bars=1000,
        n_fine_tune=100,
        n_optuna_trials=4,
        n_trader_trials=8,
        inner_cv_splits=2,
        n_splits=4,
        seed=11,
        cost_stress=0.0,
        min_n_trades=0,
        min_dsr=0.0,
        max_dd_bp_limit=1e9,
        sigma_span=40,
        cusum_h_mult=1.5,
        pt_atr=1.5,
        sl_atr=1.5,
        vert_horizon=24,
        min_edge=0.0,
        onnx_atol=1e-3,
    )
    rep = run_pipeline(cfg)

    # Each step contributed.
    assert rep.label_run_id is not None
    assert rep.n_labels > 0
    assert rep.model_id is not None
    assert rep.params_id is not None
    assert rep.lockbox_run_id is not None
    assert rep.lockbox_passed is True  # gates set to be lenient
    assert rep.published is True
    assert rep.onnx_path is not None
    assert rep.onnx_sha256

    # Rust hot-swap watcher's directory got the artifacts.
    live = live_champion_dir()
    assert (live / "champion.onnx").exists()
    assert (live / "manifest.json").exists()

    # Parent pipeline.full row landed; child rows landed too.
    c = open_ro()
    try:
        rows = c.execute(
            "SELECT command, status FROM pipeline_runs "
            "WHERE instrument = ? ORDER BY ts_started_ms",
            [seeded],
        ).fetchall()
    finally:
        c.close()
    commands = [r[0] for r in rows]
    assert "pipeline.full" in commands
    # Children: label, train.side, finetune, lockbox. (export.champion has
    # no instrument, so it's filtered out by the WHERE clause.)
    for child in ("label", "train.side", "finetune", "lockbox"):
        assert child in commands, f"missing child {child} in {commands}"
    # Every recorded run reached a terminal state.
    statuses = [r[1] for r in rows]
    assert all(s in {"success", "failed"} for s in statuses), f"stuck rows: {statuses}"
    # The parent succeeded (lockbox passed → orchestrator returned cleanly).
    assert any(c == "pipeline.full" and s == "success" for c, s in rows)


def test_lockbox_failure_skips_export_but_does_not_raise(seeded, tmp_path, monkeypatch):
    """When the lockbox gate fails, run_pipeline should still return
    cleanly with `published=False` so the prior champion stays live.
    """
    cfg = PipelineConfig(
        instrument=seeded,
        n_bars=1000,
        n_fine_tune=100,
        n_optuna_trials=2,
        n_trader_trials=4,
        inner_cv_splits=2,
        n_splits=4,
        seed=11,
        cost_stress=0.0,
        sigma_span=40,
        cusum_h_mult=1.5,
        pt_atr=1.5,
        sl_atr=1.5,
        vert_horizon=24,
        min_edge=0.0,
        # Make the lockbox impossible to pass:
        min_dsr=10.0,
        min_n_trades=10_000,
    )
    rep = run_pipeline(cfg)
    assert rep.lockbox_passed is False
    assert rep.published is False
    assert rep.onnx_path is None

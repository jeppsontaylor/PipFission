"""Hardest end-to-end test: synthetic data → labels → train → fine-tune
→ lockbox → ONNX export → roundtrip verify.

Demonstrates that every milestone in the research layer composes.
Anything failing here is a regression in the research/ → live handoff.
"""
from __future__ import annotations

import math

import pytest

from research.paths import PATHS

if not PATHS.label_opt_bin.exists() or not PATHS.trader_backtest_bin.exists():
    pytest.skip("Rust binaries missing", allow_module_level=True)

from research.data.duckdb_io import open_rw, open_ro  # noqa: E402
from research.data.extract import extract_bars_10s  # noqa: E402
from research.export.manifest import write_manifest  # noqa: E402
from research.export.onnx_export import export_calibrated_to_onnx, verify_onnx_roundtrip  # noqa: E402
from research.labeling.label_opt import LabelOptConfig, run_label_opt, write_labels  # noqa: E402
from research.lockbox.gate import LockboxConfig, seal_lockbox  # noqa: E402
from research.training.side_train import SideTrainConfig, train_side  # noqa: E402
from research.trader.optimizer import TraderFineTuneConfig, fine_tune_trader  # noqa: E402

INSTRUMENT = "TEST_E2E"

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
"""


@pytest.fixture(scope="module")
def seeded():
    c = open_rw()
    try:
        c.execute(SCHEMA)
        for tbl in [
            "bars_10s", "labels", "oof_predictions", "model_metrics",
        ]:
            c.execute(f"DELETE FROM {tbl} WHERE instrument = ?", [INSTRUMENT])
        c.execute("DELETE FROM lockbox_results WHERE run_id LIKE 'lockbox_side_lgbm_%'")
        c.execute("DELETE FROM optimizer_trials WHERE study_id LIKE 'trader_side_lgbm_%'")
        c.execute("DELETE FROM trader_metrics WHERE model_id LIKE 'side_lgbm_%'")
        c.execute("DELETE FROM model_artifacts WHERE model_id LIKE 'side_lgbm_%'")
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


def test_full_pipeline(seeded):
    bars = extract_bars_10s(seeded, n_recent=1000)
    payload = run_label_opt(
        bars,
        LabelOptConfig(
            sigma_span=40, cusum_h_mult=1.5, pt_atr=1.5, sl_atr=1.5,
            vert_horizon=24, min_edge=0.0,
        ),
    )
    write_labels(seeded, payload, label_run_id="e2e")
    train_res = train_side(SideTrainConfig(
        instrument=seeded, n_bars=1000, n_splits=4, n_test_groups=2,
        embargo_pct=0.01, n_optuna_trials=4, inner_cv_splits=2,
        label_run_id="e2e", seed=11,
    ))
    ft = fine_tune_trader(TraderFineTuneConfig(
        instrument=seeded, model_id=train_res.model_version,
        n_train=1000, n_fine_tune=100, n_trials=8, seed=11, cost_stress=0.0,
    ))
    lock = seal_lockbox(LockboxConfig(
        instrument=seeded, model_id=train_res.model_version,
        params_id=ft["params_id"], n_seen=1100, n_lockbox=100,
        cost_stress=0.0, min_n_trades=0, min_dsr=0.0, max_dd_bp_limit=1e9,
    ))
    assert lock.sealed
    assert lock.passed
    info = export_calibrated_to_onnx(train_res.model_version)
    rt = verify_onnx_roundtrip(train_res.model_version, atol=1e-3)
    assert rt["max_err"] < 1e-3
    manifest = write_manifest(train_res.model_version, info["onnx_path"], info["feature_names"])
    assert manifest["sha256"]
    assert (PATHS.artifacts_dir / "models" / "live" / "champion.onnx").exists()
    assert (PATHS.artifacts_dir / "models" / "live" / "manifest.json").exists()

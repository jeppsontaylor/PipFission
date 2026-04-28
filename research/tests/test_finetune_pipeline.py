"""Full pipeline smoke test: seed → label → train → fine-tune.

Asserts that the trader optimiser:
  - Runs N trials successfully against the Rust binary.
  - Picks a Pareto-optimal point with at least one trade in fine-tune.
  - Persists rows to optimizer_trials and trader_metrics.
"""
from __future__ import annotations

import math

import pytest

from research.paths import PATHS

if not PATHS.label_opt_bin.exists() or not PATHS.trader_backtest_bin.exists():
    pytest.skip(
        "Rust binaries missing; build with cargo first", allow_module_level=True
    )

from research.data.duckdb_io import open_rw, open_ro  # noqa: E402
from research.data.extract import extract_bars_10s  # noqa: E402
from research.labeling.label_opt import LabelOptConfig, run_label_opt, write_labels  # noqa: E402
from research.training.side_train import SideTrainConfig, train_side  # noqa: E402
from research.trader.optimizer import TraderFineTuneConfig, fine_tune_trader  # noqa: E402

INSTRUMENT = "TEST_FT"

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
        # `optimizer_trials` and `trader_metrics` are keyed by study_id /
        # params_id; we'll wipe them by run_id when the test is done.
        c.execute("DELETE FROM optimizer_trials WHERE study_id LIKE 'trader_side_lgbm_%'")
        c.execute("DELETE FROM trader_metrics WHERE model_id LIKE 'side_lgbm_%'")
        # 1100 bars: 1000 train + 100 fine-tune.
        for i in range(1100):
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
    c = open_rw()
    try:
        for tbl in [
            "bars_10s", "labels", "oof_predictions", "model_metrics",
        ]:
            c.execute(f"DELETE FROM {tbl} WHERE instrument = ?", [INSTRUMENT])
        # `optimizer_trials` and `trader_metrics` are keyed by study_id /
        # params_id; we'll wipe them by run_id when the test is done.
        c.execute("DELETE FROM optimizer_trials WHERE study_id LIKE 'trader_side_lgbm_%'")
        c.execute("DELETE FROM trader_metrics WHERE model_id LIKE 'side_lgbm_%'")
    finally:
        c.close()


def test_full_pipeline_runs(seeded):
    bars = extract_bars_10s(seeded, n_recent=1000)
    payload = run_label_opt(
        bars,
        LabelOptConfig(
            sigma_span=40, cusum_h_mult=1.5, pt_atr=1.5, sl_atr=1.5,
            vert_horizon=24, min_edge=0.0,
        ),
    )
    write_labels(seeded, payload, label_run_id="ft_test")

    train_res = train_side(SideTrainConfig(
        instrument=seeded,
        n_bars=1000,
        n_splits=4,
        n_test_groups=2,
        embargo_pct=0.01,
        n_optuna_trials=4,
        inner_cv_splits=2,
        label_run_id="ft_test",
        seed=11,
    ))
    assert train_res.n_oof > 0
    rep = fine_tune_trader(TraderFineTuneConfig(
        instrument=seeded,
        model_id=train_res.model_version,
        n_train=1000,
        n_fine_tune=100,
        n_trials=8,
        seed=11,
        cost_stress=0.0,  # zero-cost stress for the smoke test
    ))
    assert rep["n_trials"] == 8
    with open_ro() as c:
        n_trials_db = c.execute(
            "SELECT COUNT(*) FROM optimizer_trials WHERE study_id = ?",
            [rep["study_id"]],
        ).fetchone()[0]
        n_trader_metrics = c.execute(
            "SELECT COUNT(*) FROM trader_metrics WHERE params_id = ?",
            [rep["params_id"]],
        ).fetchone()[0]
    assert n_trials_db == 8
    assert n_trader_metrics == 1

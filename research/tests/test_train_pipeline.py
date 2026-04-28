"""End-to-end: seed → label → train → assert OOF written.

This test is the smoke check that proves the M5 pipeline runs over
synthetic data. The synthetic generator produces an oscillating drift
that is deliberately learnable so the classifier should beat coin-flip
even on a tiny window.
"""
from __future__ import annotations

import math

import pytest

from research.paths import PATHS

if not PATHS.label_opt_bin.exists():
    pytest.skip("label_opt binary missing; build with cargo first", allow_module_level=True)

from research.data.duckdb_io import open_rw, open_ro  # noqa: E402
from research.data.extract import extract_bars_10s  # noqa: E402
from research.labeling.label_opt import LabelOptConfig, run_label_opt, write_labels  # noqa: E402
from research.training.side_train import SideTrainConfig, train_side  # noqa: E402

INSTRUMENT = "TEST_TRAIN"

# Minimal schema (subset; init-schema CLI applies the full one).
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
"""


@pytest.fixture(scope="module")
def seeded():
    c = open_rw()
    try:
        c.execute(SCHEMA)
        c.execute("DELETE FROM bars_10s WHERE instrument = ?", [INSTRUMENT])
        c.execute("DELETE FROM labels WHERE instrument = ?", [INSTRUMENT])
        c.execute("DELETE FROM oof_predictions WHERE instrument = ?", [INSTRUMENT])
        # Pure oscillation (zero drift) so the label optimiser sees
        # both up- and down-side wins, producing a balanced label set.
        n = 1100
        for i in range(n):
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
        c.execute("DELETE FROM bars_10s WHERE instrument = ?", [INSTRUMENT])
        c.execute("DELETE FROM labels WHERE instrument = ?", [INSTRUMENT])
        c.execute("DELETE FROM oof_predictions WHERE instrument = ?", [INSTRUMENT])
        c.execute("DELETE FROM model_metrics WHERE instrument = ?", [INSTRUMENT])
    finally:
        c.close()


def test_train_side_runs_end_to_end(seeded):
    bars = extract_bars_10s(seeded, n_recent=1000)
    payload = run_label_opt(
        bars,
        LabelOptConfig(
            sigma_span=40,
            cusum_h_mult=1.5,
            pt_atr=1.5,
            sl_atr=1.5,
            vert_horizon=24,
            min_edge=0.0,
        ),
    )
    rid = write_labels(seeded, payload, label_run_id="train_test_run")
    assert rid == "train_test_run"
    assert len(payload["labels"]) >= 30, f"need >= 30 labels, got {len(payload['labels'])}"

    cfg = SideTrainConfig(
        instrument=seeded,
        n_bars=1000,
        n_splits=4,
        n_test_groups=2,
        embargo_pct=0.01,
        n_optuna_trials=4,  # tiny for the test
        inner_cv_splits=2,
        label_run_id=rid,
        seed=7,
    )
    res = train_side(cfg)
    assert res.n_oof > 0
    assert 0.0 <= res.oos_auc <= 1.0
    assert res.oos_log_loss >= 0.0
    with open_ro() as c:
        n_oof = c.execute(
            "SELECT COUNT(*) FROM oof_predictions WHERE instrument = ? AND model_version = ?",
            [seeded, res.model_version],
        ).fetchone()[0]
    assert n_oof == res.n_oof

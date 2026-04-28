"""End-to-end test: seed DuckDB with synthetic bars, run the Rust
label_opt binary via the Python wrapper, and assert the chosen labels
were written back to the `labels` table.

Uses the real on-disk DuckDB file so the test exercises the same
path the live pipeline will run. The test cleans up its own seeded
rows on teardown via a unique label_run_id.
"""
from __future__ import annotations

import math
import sys

import pytest

# Skip if Rust binaries haven't been built (CI safety).
from research.paths import PATHS

if not PATHS.label_opt_bin.exists():
    pytest.skip("label_opt binary missing; build with cargo first", allow_module_level=True)

from research.data.duckdb_io import open_rw, open_ro  # noqa: E402
from research.data.extract import extract_bars_10s  # noqa: E402
from research.labeling.label_opt import LabelOptConfig, run_label_opt, write_labels  # noqa: E402

SCHEMA = """
CREATE TABLE IF NOT EXISTS price_ticks (
    instrument  VARCHAR  NOT NULL,
    ts_ms       BIGINT   NOT NULL,
    bid         DOUBLE   NOT NULL,
    ask         DOUBLE   NOT NULL,
    mid         DOUBLE   NOT NULL,
    spread_bp   DOUBLE   NOT NULL,
    status      VARCHAR
);
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
"""


@pytest.fixture(scope="module")
def seeded_db():
    """Ensure DuckDB exists with the minimal schema and synthetic bars."""
    # Open RW just long enough to seed; close before label_opt's RW write.
    c = open_rw()
    try:
        c.execute(SCHEMA)
        # Wipe any prior synthetic rows from a previous test run.
        c.execute("DELETE FROM bars_10s WHERE instrument = ?", ["TEST_PAIR"])
        c.execute("DELETE FROM labels WHERE instrument = ?", ["TEST_PAIR"])
        n = 800
        for i in range(n):
            p = 1.0 + 0.0005 * i + 0.002 * math.sin(i / 40.0)
            c.execute(
                "INSERT INTO bars_10s (instrument, ts_ms, open, high, low, close, n_ticks, spread_bp_avg) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ["TEST_PAIR", i * 10_000, p, p + 3e-4, p - 3e-4, p, 1, 0.5],
            )
    finally:
        c.close()
    yield "TEST_PAIR"
    # Cleanup
    c = open_rw()
    try:
        c.execute("DELETE FROM bars_10s WHERE instrument = ?", ["TEST_PAIR"])
        c.execute("DELETE FROM labels WHERE instrument = ?", ["TEST_PAIR"])
    finally:
        c.close()


def test_extract_returns_seeded_bars(seeded_db):
    bars = extract_bars_10s(seeded_db, n_recent=300)
    assert len(bars) == 300
    assert bars["instrument"].unique().to_list() == ["TEST_PAIR"]
    assert bars["ts_ms"].is_sorted()


def test_label_optimiser_writes_to_duckdb(seeded_db):
    bars = extract_bars_10s(seeded_db, n_recent=500)
    cfg = LabelOptConfig(
        sigma_span=40,
        cusum_h_mult=2.0,
        pt_atr=1.5,
        sl_atr=1.5,
        vert_horizon=24,
        min_edge=0.0,
    )
    payload = run_label_opt(bars, cfg)
    assert payload["n_bars"] == 500
    assert payload["n_events"] >= 1, "synthetic series should fire at least one event"
    assert len(payload["labels"]) >= 1

    run_id = write_labels(seeded_db, payload, label_run_id="pytest_run")
    assert run_id == "pytest_run"

    with open_ro() as c:
        n_labels = c.execute(
            "SELECT COUNT(*) FROM labels WHERE instrument = ? AND label_run_id = ?",
            [seeded_db, run_id],
        ).fetchone()[0]
    assert n_labels == len(payload["labels"])


def test_each_chosen_label_has_t1_strictly_greater_than_t0(seeded_db):
    bars = extract_bars_10s(seeded_db, n_recent=300)
    payload = run_label_opt(bars, LabelOptConfig(min_edge=0.0))
    for r in payload["labels"]:
        assert r["t1_ms"] > r["ts_ms"], r
        assert r["barrier_hit"] in {"Pt", "Sl", "Vert"}, r

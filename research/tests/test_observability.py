"""Test the pipeline-runs tracker context manager.

Uses a temp DuckDB so we don't pollute the project's `data/oanda.duckdb`.
The tracker opens connections via `research.data.duckdb_io`, which
resolves the path through `PATHS.duckdb_path` — we monkeypatch that
single attribute for the test.
"""
from __future__ import annotations

import duckdb
import pytest

from research.observability.tracker import track_run

SCHEMA = """
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


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "obs.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(SCHEMA)
    conn.close()
    # `Paths` is a frozen dataclass — we swap the whole instance out
    # by rebinding the module-level reference inside duckdb_io, which
    # is where `rw_conn()` reads `PATHS.duckdb_path` from.
    from dataclasses import replace
    from research import paths as paths_mod
    from research.data import duckdb_io

    new_paths = replace(paths_mod.PATHS, duckdb_path=db_path)
    monkeypatch.setattr(duckdb_io, "PATHS", new_paths)
    return db_path


def _read_rows(db_path):
    c = duckdb.connect(str(db_path), read_only=True)
    try:
        return c.execute(
            "SELECT run_id, command, instrument, status, elapsed_ms, "
            "ts_finished_ms, error_msg, args_json "
            "FROM pipeline_runs"
        ).fetchall()
    finally:
        c.close()


def test_success_records_finished_row(tmp_db):
    with track_run("label", {"instrument": "EUR_USD", "n_bars": 1000}, instrument="EUR_USD") as run_id:
        assert isinstance(run_id, str) and len(run_id) >= 16

    rows = _read_rows(tmp_db)
    assert len(rows) == 1
    rid, cmd, inst, status, elapsed, finished, err, args = rows[0]
    assert rid == run_id
    assert cmd == "label"
    assert inst == "EUR_USD"
    assert status == "success"
    assert finished is not None
    assert elapsed is not None and elapsed >= 0
    assert err is None
    assert "EUR_USD" in args


def test_failure_records_failed_row_and_reraises(tmp_db):
    with pytest.raises(ValueError, match="boom"):
        with track_run("train.side", {"seed": 42}, instrument="EUR_USD"):
            raise ValueError("boom")

    rows = _read_rows(tmp_db)
    assert len(rows) == 1
    _, cmd, inst, status, elapsed, finished, err, _ = rows[0]
    assert cmd == "train.side"
    assert inst == "EUR_USD"
    assert status == "failed"
    assert finished is not None
    assert elapsed is not None
    assert err is not None and "boom" in err


def test_no_instrument_command(tmp_db):
    with track_run("export.champion", {"model_id": "m1"}):
        pass

    rows = _read_rows(tmp_db)
    assert len(rows) == 1
    _, cmd, inst, status, *_ = rows[0]
    assert cmd == "export.champion"
    assert inst is None
    assert status == "success"


def test_args_with_non_json_values_are_safely_serialized(tmp_db):
    from pathlib import Path

    with track_run("custom", {"path": Path("/tmp/x"), "n": 5}):
        pass

    rows = _read_rows(tmp_db)
    assert len(rows) == 1
    args = rows[0][7]
    # path is stringified via default=str
    assert "/tmp/x" in args
    assert "\"n\": 5" in args

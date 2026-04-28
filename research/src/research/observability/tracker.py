"""Pipeline-run observability.

Every `python -m research <cmd>` invocation should be wrapped in
`with track_run(command, args, instrument=…): ...` so the dashboard's
"Pipeline runs" panel can show what ran, when, how long it took, and
whether it succeeded.

The context manager:

    1. Inserts a `pipeline_runs` row at entry with `status='running'`.
    2. On clean exit, updates the row with `status='success'`,
       `ts_finished_ms`, `elapsed_ms`.
    3. On exception, updates with `status='failed'` + the exception
       message, then re-raises so the CLI exit code stays accurate.

Implementation notes:

* Each call opens its own short-lived `rw_conn`, mirroring the rest of
  the research layer. The live Rust writer holds the long-lived mutex;
  we just need a moment to insert / update.
* `args` is JSON-serialised with `default=str` so non-trivial values
  (Paths, Enums) round-trip without crashing the tracker.
* The tracker is intentionally fail-soft: if the database is missing
  or the schema hasn't been applied, we log a warning and let the
  command run anyway — observability shouldn't block research work.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

import duckdb

from research.data.duckdb_io import rw_conn


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return json.dumps({"_repr": repr(value)})


@contextmanager
def track_run(
    command: str,
    args: dict[str, Any] | None = None,
    *,
    instrument: str | None = None,
) -> Iterator[str]:
    """Wrap a CLI command body so the run lands in `pipeline_runs`.

    Yields the new `run_id` so callers can attach further structured
    logs to it if they want; most callers will just `with track_run(...):`.
    """
    run_id = uuid.uuid4().hex
    started = _now_ms()
    args_json = _safe_json(args or {})

    insert_ok = _try_insert_start(run_id, command, instrument, args_json, started)

    try:
        yield run_id
    except BaseException as exc:
        if insert_ok:
            _try_finalize(
                run_id,
                status="failed",
                error_msg=f"{type(exc).__name__}: {exc}",
                started=started,
            )
        raise
    else:
        if insert_ok:
            _try_finalize(run_id, status="success", error_msg=None, started=started)


def _try_insert_start(
    run_id: str,
    command: str,
    instrument: str | None,
    args_json: str,
    started: int,
) -> bool:
    try:
        with rw_conn() as conn:
            conn.execute(
                "INSERT INTO pipeline_runs "
                "(run_id, command, instrument, args_json, ts_started_ms, status) "
                "VALUES (?, ?, ?, ?, ?, 'running')",
                [run_id, command, instrument, args_json, started],
            )
        return True
    except duckdb.Error as exc:
        # No DB / no schema yet → the live server hasn't run init_schema.
        # Don't block the command; just warn.
        print(f"[tracker] could not record start for {command}: {exc}")
        return False


def _try_finalize(
    run_id: str,
    *,
    status: str,
    error_msg: str | None,
    started: int,
) -> None:
    finished = _now_ms()
    elapsed = finished - started
    try:
        with rw_conn() as conn:
            conn.execute(
                "UPDATE pipeline_runs "
                "SET ts_finished_ms = ?, status = ?, elapsed_ms = ?, error_msg = ? "
                "WHERE run_id = ?",
                [finished, status, elapsed, error_msg, run_id],
            )
    except duckdb.Error as exc:
        print(f"[tracker] could not finalise run {run_id} ({status}): {exc}")

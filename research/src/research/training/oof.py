"""Out-of-fold probability writer. Persists OOF rows to parquet AND
inserts them into `DuckDB.oof_predictions`. Both targets are written
because the trader optimiser reads the parquet (fast columnar IO) and
the dashboard reads the DuckDB rows (live monitoring).
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

import polars as pl

from research.data.duckdb_io import rw_conn
from research.paths import PATHS


OOF_SCHEMA: dict[str, pl.DataType] = {
    "instrument": pl.Utf8,
    "ts_ms": pl.Int64,
    "t1_ms": pl.Int64,
    "fold": pl.Int32,
    "side_label": pl.Int8,
    "meta_label": pl.Int8,
    "p_long": pl.Float64,
    "p_short": pl.Float64,
    "p_take": pl.Float64,
    "calibrated_p": pl.Float64,
    "model_version": pl.Utf8,
}


def write_oof_parquet(df: pl.DataFrame, run_id: str | None = None, instrument: str = "") -> Path:
    """Write OOF rows to artifacts/oof/<instrument>/<run>.parquet."""
    rid = run_id or f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    out = PATHS.artifacts_dir / "oof" / instrument / f"{rid}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return out


def write_oof_duckdb(df: pl.DataFrame) -> int:
    """Insert OOF rows into `DuckDB.oof_predictions`. Returns inserted
    count. Caller is responsible for de-duping by `model_version` if
    re-running."""
    if df.is_empty():
        return 0
    rows = df.select([
        "instrument", "ts_ms", "fold", "p_long", "p_short", "p_take",
        "calibrated_p", "model_version",
    ]).to_dicts()
    with rw_conn() as c:
        for r in rows:
            c.execute(
                """
                INSERT INTO oof_predictions
                  (instrument, ts_ms, fold, p_long, p_short, p_take, calibrated_p, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    r["instrument"],
                    int(r["ts_ms"]),
                    int(r["fold"]),
                    float(r["p_long"]),
                    float(r["p_short"]),
                    float(r["p_take"]),
                    float(r["calibrated_p"]),
                    r["model_version"],
                ],
            )
    return len(rows)


def delete_oof_for_version(model_version: str) -> int:
    """Drop existing rows for a given model_version. Useful when
    re-running training under the same id without producing duplicates."""
    with rw_conn() as c:
        n = c.execute(
            "DELETE FROM oof_predictions WHERE model_version = ?",
            [model_version],
        ).fetchone()
    return int(n[0]) if n is not None else 0

"""Python wrapper around the Rust `label_opt` binary.

The Rust binary owns the full triple-barrier + constrained label
optimisation logic. Python just builds the JSON request, shells out,
parses the response, and inserts the chosen `LabelRow`s into
`DuckDB.labels`.
"""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import polars as pl

from research.data.duckdb_io import rw_conn
from research.paths import PATHS


@dataclass
class LabelOptConfig:
    """Mirror of the Rust `Config` struct in src/bin/label_opt.rs."""
    sigma_span: int = 60
    cusum_h_mult: float = 2.0
    min_gap: int = 1
    pt_atr: float = 2.0
    sl_atr: float = 2.0
    vert_horizon: int = 36
    min_edge: float = 0.0
    min_hold_ms: int = 10_000
    max_hold_ms: int = 600_000
    downside_lambda: float = 4.0
    turnover_cost: float = 0.0
    # Binary-mode minority floor. Operator mandate: minority side ≥ 30%
    # of the chosen subset. The Rust optimiser drops the lowest-scoring
    # majority trades to reach this floor.
    min_minority_frac: float = 0.30


def run_label_opt(bars: pl.DataFrame, cfg: LabelOptConfig | None = None) -> dict:
    """Pipe `bars` through the Rust binary; return the parsed JSON dict.

    `bars` must have columns
    `[ts_ms, open, high, low, close, n_ticks, spread_bp_avg]`. Output
    dict has `n_bars`, `n_events`, `n_raw_labels`, `labels`.
    """
    cfg = cfg or LabelOptConfig()
    binary = PATHS.label_opt_bin
    if not binary.exists():
        raise RuntimeError(
            f"label_opt binary not found at {binary}; "
            f"build it with `cargo build --release -p labeling --bin label_opt`"
        )
    expected = {"ts_ms", "open", "high", "low", "close", "n_ticks", "spread_bp_avg"}
    missing = expected - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")
    payload = {
        "bars": bars.select(sorted(expected)).to_dicts(),
        "cfg": asdict(cfg),
    }
    proc = subprocess.run(
        [str(binary)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"label_opt failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)


def write_labels(
    instrument: str,
    labels_payload: dict,
    label_run_id: Optional[str] = None,
) -> str:
    """Insert the chosen labels into `DuckDB.labels` under a unique
    `label_run_id`. Returns the run-id string. Replaces any existing
    rows with the same (instrument, label_run_id) pair so re-running
    with the same id yields a clean re-write.

    The (instrument, ts_ms, label_run_id) PK lets us keep multiple
    historical runs side-by-side for diff/replay; the *current* run is
    always the most recent one written.
    """
    run_id = label_run_id or f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    rows = labels_payload.get("labels", [])
    if not rows:
        return run_id

    with rw_conn() as c:
        c.execute(
            "DELETE FROM labels WHERE instrument = ? AND label_run_id = ?",
            [instrument, run_id],
        )
        # Build a deterministic oracle_score per row: the per-trade
        # absolute realized return. The Rust optimiser already used a
        # similar score internally; persisting it here makes the
        # dashboard's "ideal score" plot trivial to render.
        for r in rows:
            c.execute(
                """
                INSERT INTO labels
                  (instrument, ts_ms, t1_ms, side, meta_y, realized_r,
                   barrier_hit, oracle_score, label_run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    instrument,
                    int(r["ts_ms"]),
                    int(r["t1_ms"]),
                    int(r["side"]),
                    int(r["meta_y"]),
                    float(r["realized_r"]),
                    str(r["barrier_hit"]),
                    abs(float(r["realized_r"])),
                    run_id,
                ],
            )
    return run_id

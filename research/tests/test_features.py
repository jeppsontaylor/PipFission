"""Phase B feature-engineering tests.

Two responsibilities:

1. Confirm the re-engineered features (`log1p_n_ticks`,
   `force_index_rel`) stay in sane ranges across realistic bar input.
   This is the "no more 9,816 in the dashboard" guard — if a future
   tweak accidentally re-introduces an unscaled column, these tests
   blow up immediately.

2. Confirm the FEATURE_NAMES list still has 24 entries (downstream
   code relies on this) and that the renames replaced the old
   columns rather than appending new ones.

Range tests use 1 000 synthetic bars with realistic n_ticks (0-15 000)
+ a random-walk price; all features must stay bounded.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from research.features import (
    FEATURE_NAMES,
    FORCE_INDEX_REL_WINDOW,
    compute_features,
)


def _synthetic_bars(n: int, seed: int = 0) -> pl.DataFrame:
    """Random-walk OHLC + heavy-tailed n_ticks, no NaNs."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(scale=1e-4, size=n)
    close = 1.10 * np.cumprod(1.0 + rets)
    spread_bp = rng.uniform(0.5, 5.0, size=n)
    # n_ticks drawn from a heavy-tailed distribution so we stress the
    # window-mean denominator with both quiet and burst bars.
    n_ticks = rng.integers(low=0, high=15_000, size=n).astype(np.int64)
    high = close * (1.0 + np.abs(rng.normal(scale=2e-4, size=n)))
    low = close * (1.0 - np.abs(rng.normal(scale=2e-4, size=n)))
    return pl.DataFrame(
        {
            "ts_ms": (np.arange(n) * 10_000).astype(np.int64),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "n_ticks": n_ticks,
            "spread_bp_avg": spread_bp,
        }
    )


def test_feature_names_dimensions_unchanged():
    """Phase B re-engineering renames two columns; total count stays 24
    so the live ONNX champion's input contract doesn't change shape."""
    assert len(FEATURE_NAMES) == 24
    assert "log1p_n_ticks" in FEATURE_NAMES
    assert "force_index_rel" in FEATURE_NAMES
    # Old names must be gone — otherwise we'd have 26 features and
    # break the 24-dim ONNX input.
    assert "n_ticks" not in FEATURE_NAMES
    assert "force_index" not in FEATURE_NAMES


def test_log1p_n_ticks_bounded():
    """ln(1 + n_ticks) for n_ticks ∈ [0, 15 000] sits in [0, 9.62].
    Allow a tiny slack for floats."""
    bars = _synthetic_bars(1_000, seed=1)
    feats = compute_features(bars)
    col = feats["log1p_n_ticks"].to_numpy()
    assert col.min() >= 0.0
    assert col.max() <= 12.0
    # Must not be NaN/inf.
    assert np.isfinite(col).all()


def test_force_index_rel_bounded_in_normal_regime():
    """In the synthetic data above, force_index_rel = (close - prev) ×
    n_ticks / mean_n_ticks_60. For close changes of ~1bp and a tick
    burst at 10x the rolling mean, |force_index_rel| ≲ 0.005, which
    sits comfortably below ±50."""
    bars = _synthetic_bars(1_000, seed=2)
    feats = compute_features(bars)
    col = feats["force_index_rel"].to_numpy()
    assert np.isfinite(col).all()
    assert col.min() >= -50.0
    assert col.max() <= 50.0


def test_force_index_rel_no_zero_division_on_idle_market():
    """Even when n_ticks is zero for many bars, the floored denominator
    keeps `force_index_rel` finite."""
    n = 200
    bars = pl.DataFrame(
        {
            "ts_ms": (np.arange(n) * 10_000).astype(np.int64),
            "open": np.full(n, 1.10),
            "high": np.full(n, 1.10001),
            "low": np.full(n, 1.09999),
            "close": np.full(n, 1.10),
            "n_ticks": np.zeros(n, dtype=np.int64),
            "spread_bp_avg": np.full(n, 1.0),
        }
    )
    feats = compute_features(bars)
    col = feats["force_index_rel"].to_numpy()
    assert np.isfinite(col).all()


def test_force_index_rel_window_constant_matches_implementation():
    """The window length is a named constant the Rust mirror crate
    reads; pin it so the two implementations stay aligned. If you bump
    this in Python, also bump `FORCE_INDEX_REL_WINDOW` in
    `server/crates/bar-features/src/lib.rs`."""
    assert FORCE_INDEX_REL_WINDOW == 60


def test_compute_features_output_shape():
    """Sanity: 24 feature cols + ts_ms = 25 columns, length matches
    input bars."""
    bars = _synthetic_bars(50, seed=3)
    feats = compute_features(bars)
    assert feats.height == 50
    assert set(feats.columns) == {"ts_ms", *FEATURE_NAMES}


# ---- Python⇄Rust parity ----------------------------------------------------
#
# The live engine computes features in `server/crates/bar-features/`; the
# Python research layer computes them here. Both must agree byte-for-byte
# or live inference receives different inputs from training.
#
# `bar_features_dump` is the bridge: stdin JSON bars → stdout JSON
# `{feature_names, values}`. We feed it the SAME synthetic bars the
# Python implementation processes and assert agreement within float
# tolerance.


def _bar_features_dump_path():
    """Locate the `bar_features_dump` binary; skip the test if absent."""
    from research.paths import PATHS

    rel = PATHS.server_dir / "target" / "release" / "bar_features_dump"
    dbg = PATHS.server_dir / "target" / "debug" / "bar_features_dump"
    for p in (rel, dbg):
        if p.exists():
            return p
    return None


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_python_rust_feature_parity(seed):
    """Synthetic bars through Python `compute_features` must match
    Rust `recompute_last` for the LAST bar within float tolerance.

    Catches the silent class of bug where one implementation drifts —
    e.g. Phase B's `force_index_rel` denominator window length, or
    a future feature reorder.
    """
    import json
    import subprocess

    bin_path = _bar_features_dump_path()
    if bin_path is None:
        pytest.skip("bar_features_dump binary not built")

    n = 200
    bars = _synthetic_bars(n, seed=seed)
    py_feats = compute_features(bars)
    py_last = {
        name: float(py_feats[name].to_numpy()[-1]) for name in FEATURE_NAMES
    }

    bars_json = []
    for row in bars.iter_rows(named=True):
        bars_json.append(
            {
                "instrument_id": 0,
                "ts_ms": int(row["ts_ms"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "n_ticks": int(row["n_ticks"]),
                "spread_bp_avg": float(row["spread_bp_avg"]),
            }
        )
    proc = subprocess.run(
        [str(bin_path)],
        input=json.dumps(bars_json),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"bar_features_dump returned {proc.returncode}: {proc.stderr}"
        )
    rust_out = json.loads(proc.stdout)
    rust_names = rust_out["feature_names"]
    rust_values = rust_out["values"]

    # 1. Feature contract aligned.
    assert rust_names == FEATURE_NAMES, (
        "Rust + Python FEATURE_NAMES drift! "
        f"Rust: {rust_names}, Python: {FEATURE_NAMES}"
    )

    # 2. Per-feature numerical agreement. Tolerance is generous —
    # polars' rolling ops use a slightly different reduction order
    # than the Rust loops, and EMA accumulators drift a few ULPs over
    # 200 bars. What we care about is no order-of-magnitude bug.
    drifts = []
    for i, name in enumerate(FEATURE_NAMES):
        py_val = py_last[name]
        rust_val = float(rust_values[i])
        if abs(py_val) < 1e-6 and abs(rust_val) < 1e-6:
            continue
        abs_diff = abs(py_val - rust_val)
        rel_diff = abs_diff / max(abs(py_val), abs(rust_val), 1e-9)
        if abs_diff > 1e-3 and rel_diff > 0.01:
            drifts.append(
                f"{name}: py={py_val:.6f} rust={rust_val:.6f} "
                f"abs_diff={abs_diff:.6e} rel={rel_diff:.4f}"
            )
    if drifts:
        pytest.fail(
            f"Python ⇄ Rust feature parity drift on seed {seed}:\n  "
            + "\n  ".join(drifts)
        )

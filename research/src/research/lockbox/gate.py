"""Single-shot 100-bar lockbox holdout. Run exactly once per
(model_id, params_id). Writes a sealed row to `lockbox_results`;
re-running on the same run_id raises.

The lockbox is the final gate: a champion model + trader params is
allowed to be promoted to live ONLY if its lockbox row exists, is
sealed, and clears the configured thresholds (DSR > 0, max DD ≤
limit, n_trades ≥ floor).
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Optional

import joblib
import numpy as np
import polars as pl

from research.data.duckdb_io import ro_conn, rw_conn
from research.data.extract import extract_bars_10s
from research.features import compute_features
from research.paths import PATHS
from research.stats.deflated_sharpe import deflated_sharpe_from_returns


@dataclass
class LockboxConfig:
    instrument: str
    model_id: str
    params_id: str
    # Number of bars BEFORE the lockbox window (the train + fine-tune
    # region the model has already seen). The lockbox is the next
    # `n_lockbox` bars after this offset.
    n_seen: int = 1100
    n_lockbox: int = 100
    cost_stress: float = 1.0
    # Promotion thresholds.
    min_n_trades: int = 3
    min_dsr: float = 0.50
    max_dd_bp_limit: float = 1500.0


@dataclass
class LockboxResult:
    run_id: str
    sealed: bool
    passed: bool
    reasons: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def is_already_sealed(run_id: str) -> bool:
    with ro_conn() as c:
        row = c.execute(
            "SELECT sealed FROM lockbox_results WHERE run_id = ?",
            [run_id],
        ).fetchone()
    return bool(row[0]) if row else False


def _ewma_sigma(close: np.ndarray, span: int = 60) -> np.ndarray:
    if len(close) < 2 or span < 2:
        return np.zeros(len(close))
    rets = np.diff(np.log(np.clip(close, 1e-12, None)), prepend=np.log(max(close[0], 1e-12)))
    rets[0] = 0.0
    alpha = 2.0 / (span + 1)
    s2 = 0.0
    out = np.zeros(len(close))
    for i in range(1, len(close)):
        r2 = rets[i] ** 2
        s2 = r2 if i == 1 else alpha * r2 + (1 - alpha) * s2
        out[i] = math.sqrt(s2)
    return out


def _load_trader_params(params_id: str) -> dict[str, Any]:
    with ro_conn() as c:
        row = c.execute(
            "SELECT params_json FROM trader_metrics WHERE params_id = ?",
            [params_id],
        ).fetchone()
    if not row:
        raise RuntimeError(f"trader_metrics row missing for params_id={params_id}")
    return json.loads(row[0])


def seal_lockbox(cfg: LockboxConfig, run_id: Optional[str] = None) -> LockboxResult:
    """Run the lockbox evaluation exactly once. Returns a `LockboxResult`
    with pass/fail decision and the reasons. Raises if already sealed.
    """
    rid = run_id or f"lockbox_{cfg.model_id}_{cfg.params_id}"
    if is_already_sealed(rid):
        raise RuntimeError(f"lockbox already sealed for run_id={rid}")

    bars_all = extract_bars_10s(cfg.instrument, n_recent=cfg.n_seen + cfg.n_lockbox)
    if bars_all.height < cfg.n_seen + cfg.n_lockbox:
        raise RuntimeError(
            f"need at least {cfg.n_seen + cfg.n_lockbox} bars for lockbox, "
            f"got {bars_all.height}"
        )
    lockbox_window = bars_all.tail(cfg.n_lockbox)

    champion_pkl = PATHS.artifacts_dir / "models" / cfg.model_id / "champion.pkl"
    if not champion_pkl.exists():
        raise RuntimeError(f"champion model missing at {champion_pkl}")
    blob = joblib.load(champion_pkl)
    model = blob["model"]
    feat_names = blob["feature_names"]

    feats_all = compute_features(bars_all)
    feats_lockbox = feats_all.tail(cfg.n_lockbox)
    X = feats_lockbox.select(feat_names).to_numpy().astype(np.float64)
    p_long = np.clip(model.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    p_short = 1.0 - p_long
    p_take = np.maximum(p_long, p_short)
    sigma = _ewma_sigma(lockbox_window["close"].to_numpy().astype(np.float64))

    params = _load_trader_params(cfg.params_id)
    bars_payload = lockbox_window.select(
        ["ts_ms", "open", "high", "low", "close", "n_ticks", "spread_bp_avg"]
    ).to_dicts()
    probs_payload = [
        {"p_long": float(p_long[i]), "p_short": float(p_short[i]),
         "p_take": float(p_take[i]), "calibrated": float(p_take[i])}
        for i in range(cfg.n_lockbox)
    ]
    req = {
        "bars": bars_payload,
        "probs": probs_payload,
        "sigma": list(sigma.astype(float)),
        "params": params,
        "costs": {
            "commission_bp": 0.5 * cfg.cost_stress,
            "spread_bp": 1.0 * cfg.cost_stress,
            "slippage_bp": 0.5 * cfg.cost_stress,
        },
    }
    proc = subprocess.run(
        [str(PATHS.trader_backtest_bin)],
        input=json.dumps(req),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"trader_backtest failed: {proc.stderr.strip()}")
    resp = json.loads(proc.stdout)
    summary = resp["report"]["summary"]
    ledger = resp["report"]["ledger"]

    # Compute DSR on the lockbox per-trade returns. n_trials ≈ 1 since
    # we're not searching here; conservative DSR guards against tail
    # luck.
    per_trade = np.array([t["net_r"] for t in ledger], dtype=np.float64)
    dsr = deflated_sharpe_from_returns(per_trade, n_trials=1) if per_trade.size > 0 else 0.0

    reasons: list[str] = []
    n_trades = int(summary.get("n_trades", 0))
    if n_trades < cfg.min_n_trades:
        reasons.append(f"n_trades {n_trades} < min {cfg.min_n_trades}")
    if dsr < cfg.min_dsr:
        reasons.append(f"DSR {dsr:.3f} < min {cfg.min_dsr:.3f}")
    if summary.get("max_drawdown_bp", 0.0) > cfg.max_dd_bp_limit:
        reasons.append(
            f"max_dd_bp {summary.get('max_drawdown_bp'):.1f} > limit {cfg.max_dd_bp_limit:.1f}"
        )
    passed = not reasons

    sealed_summary = dict(summary)
    sealed_summary["dsr"] = float(dsr)
    sealed_summary["pass"] = passed
    sealed_summary["reasons"] = reasons
    sealed_summary["n_lockbox_bars"] = cfg.n_lockbox
    sealed_summary["cost_stress"] = cfg.cost_stress

    with rw_conn() as c:
        c.execute(
            """
            INSERT INTO lockbox_results
              (run_id, ts_ms, model_id, params_id, summary_json, sealed)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO NOTHING
            """,
            [
                rid,
                int(time.time() * 1000),
                cfg.model_id,
                cfg.params_id,
                json.dumps(sealed_summary),
                True,
            ],
        )
    return LockboxResult(run_id=rid, sealed=True, passed=passed, reasons=reasons, summary=sealed_summary)

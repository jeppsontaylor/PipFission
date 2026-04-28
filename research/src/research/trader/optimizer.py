"""Trader fine-tuner. NSGA-II (Optuna multi-objective) over TraderParams,
scored on the **next 100 bars** the classifier has never seen.

Pipeline per trial:
  1. Sample TraderParams from the bounds exposed by the Rust binary.
  2. Build a `trader_backtest` JSON request from:
       - bars: the fine-tune window (bars [n_train .. n_train+100)).
       - probs: predicted by the refit champion model on those bars.
       - sigma: per-bar EWMA σ from the labeling crate's volatility helper.
       - costs: caller-provided (default = realistic).
       - params: this trial's TraderParams.
  3. Pipe to the binary, parse the Report.
  4. Return a multi-objective vector:
       [-fine_tune_sortino, +max_drawdown_bp, +turnover_per_day, +instability]
     (Optuna minimises every objective.)

The driver also runs a parallel "in-sample" backtest on bars
[0..n_train) using the OOF probs from `oof_predictions`. Both score sets
are written to `trader_metrics`.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import optuna
import polars as pl

from research.data.duckdb_io import ro_conn, rw_conn
from research.data.extract import extract_bars_10s
from research.features import compute_features, FEATURE_NAMES
from research.paths import PATHS


@dataclass
class TraderFineTuneConfig:
    instrument: str
    model_id: str
    n_train: int = 1000
    n_fine_tune: int = 100
    n_trials: int = 50
    seed: int = 7
    # Cost stress factor. 1.0 = realistic, 2.0 = stress-test.
    cost_stress: float = 1.0


def load_param_bounds() -> list[dict[str, Any]]:
    """Read TraderParams bounds from the Rust binary so the optimiser
    never drifts out of legal trader configurations."""
    proc = subprocess.run(
        [str(PATHS.trader_backtest_bin), "--print-bounds"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _sample_params(trial: optuna.Trial, bounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Sample a full TraderParams dict from the Optuna trial."""
    out: dict[str, Any] = {}
    for b in bounds:
        if b["is_int"]:
            out[b["name"]] = trial.suggest_int(b["name"], int(b["lo"]), int(b["hi"]))
        else:
            out[b["name"]] = trial.suggest_float(b["name"], b["lo"], b["hi"])
    return out


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


def _build_request(
    bars_df: pl.DataFrame,
    probs: np.ndarray,
    sigma: np.ndarray,
    params: dict[str, Any],
    cost_stress: float,
) -> dict[str, Any]:
    """Assemble the JSON request for trader_backtest. `probs` is a
    P(side=+1) series; we map it to (p_long, p_short, p_take, calibrated).
    """
    bars_payload = bars_df.select(
        ["ts_ms", "open", "high", "low", "close", "n_ticks", "spread_bp_avg"]
    ).to_dicts()
    p_long = np.clip(probs, 1e-9, 1 - 1e-9)
    p_short = 1.0 - p_long
    p_take = np.maximum(p_long, p_short)
    calib = p_take.copy()
    probs_payload = [
        {
            "p_long": float(p_long[i]),
            "p_short": float(p_short[i]),
            "p_take": float(p_take[i]),
            "calibrated": float(calib[i]),
        }
        for i in range(len(p_long))
    ]
    return {
        "bars": bars_payload,
        "probs": probs_payload,
        "sigma": list(sigma.astype(float)),
        "params": params,
        "costs": {
            "commission_bp": 0.5 * cost_stress,
            "spread_bp": 1.0 * cost_stress,
            "slippage_bp": 0.5 * cost_stress,
        },
    }


def _run_backtest(req: dict[str, Any]) -> dict[str, Any]:
    proc = subprocess.run(
        [str(PATHS.trader_backtest_bin)],
        input=json.dumps(req),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"trader_backtest failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def _load_oof(instrument: str, model_version: str) -> pl.DataFrame:
    with ro_conn() as c:
        return c.execute(
            "SELECT instrument, ts_ms, fold, p_long, p_short, p_take, calibrated_p, model_version "
            "FROM oof_predictions WHERE instrument = ? AND model_version = ? "
            "ORDER BY ts_ms ASC",
            [instrument, model_version],
        ).pl()


def _persist_trial(
    study_id: str, trial_id: int, params: dict[str, Any], summary: dict[str, Any]
) -> None:
    with rw_conn() as c:
        c.execute(
            """
            INSERT INTO optimizer_trials
              (study_id, trial_id, ts_ms, params_json, score, sortino, max_dd_bp,
               turnover, pareto_rank)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                study_id,
                int(trial_id),
                int(time.time() * 1000),
                json.dumps(params),
                # Composite scalar score = sortino - 0.5 * dd_bp/1000 - 0.25 * turnover.
                float(
                    summary.get("sortino", 0.0)
                    - 0.5 * (summary.get("max_drawdown_bp", 0.0) / 1000.0)
                    - 0.25 * summary.get("turnover_per_day", 0.0)
                ),
                float(summary.get("sortino", 0.0)),
                float(summary.get("max_drawdown_bp", 0.0)),
                float(summary.get("turnover_per_day", 0.0)),
                0,  # pareto_rank: filled in after the study completes; 0 = unranked.
            ],
        )


def fine_tune_trader(cfg: TraderFineTuneConfig) -> dict[str, Any]:
    """Run the multi-objective trader optimiser and persist results.

    Returns a summary dict with `best_params_id`, `n_trials`, the
    in-sample + fine-tune metrics for the chosen Pareto-optimal point,
    and the path to a JSON report.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bounds = load_param_bounds()

    # Pull the model + bars + OOF.
    bars = extract_bars_10s(cfg.instrument, n_recent=cfg.n_train + cfg.n_fine_tune)
    if bars.height < cfg.n_train + 30:
        raise RuntimeError(
            f"need at least {cfg.n_train + 30} bars for fine-tune, got {bars.height}"
        )
    train_window = bars.head(cfg.n_train)
    fine_tune_window = bars.tail(min(cfg.n_fine_tune, max(0, bars.height - cfg.n_train)))
    if fine_tune_window.height == 0:
        raise RuntimeError("fine-tune window is empty — extend bars or shrink n_train")

    # Probs:
    #  * train window: OOF probs from `oof_predictions`.
    #  * fine-tune window: the refit champion's predictions.
    oof = _load_oof(cfg.instrument, cfg.model_id)
    if oof.is_empty():
        raise RuntimeError(f"no OOF for model_version={cfg.model_id}")

    # Align OOF to train_window. Bars without an OOF entry get a neutral
    # 0.5 probability (the trader will skip them via thresholds).
    # CPCV emits one OOF row per (ts_ms, fold) pair; collapse to one
    # row per ts_ms by averaging — the dense series is what the trader
    # actually consumes per bar.
    oof_p = (
        oof.group_by("ts_ms")
        .agg(pl.col("p_long").mean().alias("_oof_p"))
    )
    train_aligned = train_window.join(oof_p, on="ts_ms", how="left").sort("ts_ms")
    train_p = train_aligned["_oof_p"].fill_null(0.5).to_numpy().astype(np.float64)
    assert len(train_p) == train_window.height, (
        f"OOF alignment produced {len(train_p)} probs for {train_window.height} bars"
    )

    champion_pkl = PATHS.artifacts_dir / "models" / cfg.model_id / "champion.pkl"
    if not champion_pkl.exists():
        raise RuntimeError(f"champion model not found at {champion_pkl}")
    blob = joblib.load(champion_pkl)
    champion_model = blob["model"]
    feat_names = blob["feature_names"]

    # Compute features for ALL bars (the on-the-fly recompute the live
    # engine does at inference time). We need them for the fine-tune
    # window — bar 1000..1099 — to ask the refit model for predictions.
    feats_all = compute_features(bars)
    feats_ft = feats_all.tail(fine_tune_window.height)
    X_ft = feats_ft.select(feat_names).to_numpy().astype(np.float64)
    ft_p = champion_model.predict_proba(X_ft)[:, 1]

    sigma_train = _ewma_sigma(train_window["close"].to_numpy().astype(np.float64))
    sigma_ft = _ewma_sigma(fine_tune_window["close"].to_numpy().astype(np.float64))

    study_id = f"trader_{cfg.model_id}_{int(time.time())}"

    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        params = _sample_params(trial, bounds)
        # Fine-tune-window backtest is the headline objective.
        ft_req = _build_request(fine_tune_window, ft_p, sigma_ft, params, cfg.cost_stress)
        ft_resp = _run_backtest(ft_req)
        ft_summary = ft_resp["report"]["summary"]
        # Don't persist trial here — wait until we also have in-sample.
        in_req = _build_request(train_window, train_p, sigma_train, params, cfg.cost_stress)
        in_resp = _run_backtest(in_req)
        in_summary = in_resp["report"]["summary"]
        trial.set_user_attr("in_sample", in_summary)
        trial.set_user_attr("fine_tune", ft_summary)
        trial.set_user_attr("params", params)
        _persist_trial(study_id, trial.number, params, ft_summary)
        # Optuna minimises every objective by default. Negate Sortino so
        # higher Sortino → smaller (better) value.
        return (
            -float(ft_summary.get("sortino", 0.0)),
            float(ft_summary.get("max_drawdown_bp", 1e9)),
            float(ft_summary.get("turnover_per_day", 0.0)),
        )

    sampler = optuna.samplers.NSGAIISampler(seed=cfg.seed)
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"], sampler=sampler)
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)

    # Pick the Pareto-front point with the highest Sortino that also
    # cleared a min-trade-count constraint. If none clear the constraint,
    # fall back to the highest-Sortino front member.
    front = study.best_trials
    if not front:
        raise RuntimeError("no Pareto-front trials — check optimiser settings")
    constrained = [
        t for t in front if t.user_attrs.get("fine_tune", {}).get("n_trades", 0) >= 3
    ]
    pool = constrained if constrained else front
    chosen = max(pool, key=lambda t: t.user_attrs["fine_tune"].get("sortino", -1e9))

    chosen_params = chosen.user_attrs["params"]
    in_summary = chosen.user_attrs["in_sample"]
    ft_summary = chosen.user_attrs["fine_tune"]

    params_id = f"params_{cfg.model_id}_{uuid.uuid4().hex[:8]}"
    with rw_conn() as c:
        c.execute("DELETE FROM trader_metrics WHERE params_id = ?", [params_id])
        c.execute(
            """
            INSERT INTO trader_metrics
              (params_id, model_id, ts_ms, in_sample_sharpe, in_sample_sortino,
               fine_tune_sharpe, fine_tune_sortino, max_dd_bp, turnover_per_day,
               hit_rate, profit_factor, n_trades, params_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                params_id,
                cfg.model_id,
                int(time.time() * 1000),
                float(in_summary.get("sharpe", 0.0)),
                float(in_summary.get("sortino", 0.0)),
                float(ft_summary.get("sharpe", 0.0)),
                float(ft_summary.get("sortino", 0.0)),
                float(ft_summary.get("max_drawdown_bp", 0.0)),
                float(ft_summary.get("turnover_per_day", 0.0)),
                float(ft_summary.get("hit_rate", 0.0)),
                float(ft_summary.get("profit_factor", 0.0)),
                int(ft_summary.get("n_trades", 0)),
                json.dumps(chosen_params),
            ],
        )

    report_dir = PATHS.artifacts_dir / "reports" / study_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "study_id": study_id,
        "model_id": cfg.model_id,
        "params_id": params_id,
        "n_trials": cfg.n_trials,
        "n_pareto": len(front),
        "chosen_params": chosen_params,
        "in_sample": in_summary,
        "fine_tune": ft_summary,
    }
    (report_dir / "summary.json").write_text(json.dumps(report, indent=2))
    return report

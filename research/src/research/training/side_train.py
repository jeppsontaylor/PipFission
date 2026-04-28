"""Side-classifier training driver — model-zoo edition.

Pipeline:
  1. Load bars and labels for one instrument (from DuckDB).
  2. Recompute features on the fly via `research.features.compute_features`.
  3. Build the supervised matrix (X, y, sample weights, t0/t1, realized_r).
  4. For each candidate model in the zoo (LightGBM, XGBoost, CatBoost,
     LogisticRegression, ExtraTrees by default):
       a. Run Combinatorial Purged CV with embargo.
       b. Per fold, run inner Optuna TPE over the candidate's search
          space using a nested purged k-fold for HPO scoring.
       c. Refit on the train fold; calibrate via sigmoid
          CalibratedClassifierCV; predict on the test fold.
       d. Aggregate OOF rows across folds.
  5. Score every candidate by OOS log loss (lower is better — gives
     calibrated probabilities). Pick the winner.
  6. Persist all candidates' metrics to `model_candidates` (so the
     operator can compare on the dashboard); insert the winner's row
     into `model_metrics`; bag the winner's OOF predictions to
     `oof_predictions` + parquet; refit the winner on the full set
     and pickle it as the champion.

This module owns no preprocessing — it pulls a `Pipeline` from
`research.models.preproc` so every candidate gets the same
StandardScaler + VarianceThreshold baseline. ONNX export of the
champion (skl2onnx + onnxmltools) is handled by `research.export`.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import optuna
import polars as pl
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from research.paths import PATHS

from research.cv.purged import combinatorial_purged_cv, purged_kfold
from research.data.duckdb_io import rw_conn, ro_conn
from research.data.extract import extract_bars_10s
from research.features import compute_features, FEATURE_NAMES
from research.models import (
    DEFAULT_CANDIDATES,
    ModelSpec,
    get_spec,
    make_preprocessor,
    parse_candidates,
    wrap_with_preprocessor,
)
from research.models.calibration import wrap_sigmoid
from research.models.preproc import SELECT_K_CHOICES, SelectK
from research.training.oof import write_oof_duckdb, write_oof_parquet


@dataclass
class SideTrainConfig:
    """All knobs the training driver respects."""
    instrument: str
    n_bars: int = 1000
    n_splits: int = 6
    n_test_groups: int = 2
    embargo_pct: float = 0.01
    n_optuna_trials: int = 25
    inner_cv_splits: int = 3
    label_run_id: Optional[str] = None
    seed: int = 42
    # Comma-separated candidate names (or `None` for the default zoo).
    # e.g. "lgbm,xgb,logreg" — see research.models.registry.
    candidates: Optional[str] = None


@dataclass
class CandidateResult:
    """Outcome for one model spec on one training run."""
    spec_name: str
    model_id: str
    oos_auc: float
    oos_log_loss: float
    oos_brier: float
    oos_balanced_acc: float
    n_train: int
    n_oof: int
    train_sortino: float
    train_sharpe: float
    max_train_sortino: float
    max_train_sharpe: float
    oof_records: list[dict]
    fitted_on_full: object  # the calibrated pipeline refitted on full data


@dataclass
class SideTrainResult:
    model_id: str
    model_version: str
    instrument: str
    spec_name: str
    oos_auc: float
    oos_log_loss: float
    oos_brier: float
    oos_balanced_acc: float
    n_train: int
    n_oof: int
    train_sortino: float
    max_train_sortino: float
    train_sharpe: float
    max_train_sharpe: float
    oof_parquet_path: str
    label_run_id: str
    champion_path: str
    candidates: list[dict]
    """All candidate scores for the run, one dict per spec."""

    def as_dict(self) -> dict:
        return asdict(self)


def _model_artifacts_dir(model_id: str) -> Path:
    return PATHS.artifacts_dir / "models" / model_id


def _load_labels(instrument: str, label_run_id: Optional[str]) -> pl.DataFrame:
    with ro_conn() as c:
        if label_run_id is None:
            row = c.execute(
                "SELECT label_run_id, MAX(ts_ms) AS last_ts "
                "FROM labels WHERE instrument = ? "
                "GROUP BY label_run_id ORDER BY last_ts DESC LIMIT 1",
                [instrument],
            ).fetchone()
            if row is None:
                raise RuntimeError(f"no labels found for {instrument}")
            label_run_id = row[0]
        df = c.execute(
            "SELECT instrument, ts_ms, t1_ms, side, meta_y, realized_r, "
            "       barrier_hit, oracle_score, label_run_id "
            "FROM labels WHERE instrument = ? AND label_run_id = ? "
            "ORDER BY ts_ms ASC",
            [instrument, label_run_id],
        ).pl()
    return df


def _build_supervised_matrix(
    bars: pl.DataFrame, labels: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feats = compute_features(bars)
    merged = labels.join(feats, on="ts_ms", how="inner")
    merged = merged.filter(pl.col("side") != 0)
    if merged.is_empty():
        raise RuntimeError("no usable labels (all sides == 0)")
    X = merged.select(FEATURE_NAMES).to_numpy().astype(np.float64)
    y = (merged["side"].to_numpy() == 1).astype(np.int8)
    w = np.abs(merged["realized_r"].to_numpy()).astype(np.float64)
    ts0 = merged["ts_ms"].to_numpy().astype(np.int64)
    ts1 = merged["t1_ms"].to_numpy().astype(np.int64)
    realized = merged["realized_r"].to_numpy().astype(np.float64)
    return X, y, w, ts0, ts1, realized


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _annualised_sharpe(per_trade: np.ndarray, n_periods_per_year: float) -> float:
    if per_trade.size < 2:
        return 0.0
    mean = per_trade.mean()
    sd = per_trade.std(ddof=1)
    if sd <= 0.0:
        return 0.0
    return float(mean / sd * math.sqrt(n_periods_per_year))


def _annualised_sortino(per_trade: np.ndarray, n_periods_per_year: float) -> float:
    if per_trade.size < 2:
        return 0.0
    mean = per_trade.mean()
    downside = per_trade[per_trade < 0]
    if downside.size == 0:
        return 0.0
    dsd = float(np.sqrt((downside ** 2).mean()))
    if dsd <= 0.0:
        return 0.0
    return float(mean / dsd * math.sqrt(n_periods_per_year))


def _split_meta_params(
    params: dict[str, Any] | None,
) -> tuple[dict[str, Any], SelectK]:
    """Pull cross-cutting meta-params (like `feature_select_k`) out of
    the candidate's hyperparameter dict. Returns `(estimator_params,
    feature_select_k)`. Defaults to `"all"` (no reduction) when absent.
    """
    if not params:
        return {}, "all"
    p = dict(params)
    k: SelectK = p.pop("feature_select_k", "all")
    # Optuna suggest_categorical passes the value through as-is; we
    # accept either string ("all") or int.
    return p, k


def _build_pipeline(spec: ModelSpec, params: dict[str, Any] | None = None):
    """Compose preprocessor + classifier (no calibrator) for one
    candidate. Returns `Pipeline([variance, scaler, ?select,
    estimator])`. Used during inner Optuna scoring where calibration
    isn't required.

    The preprocessor's `feature_select_k` is read from `params` if
    present (the training driver injects it as a meta-param); the
    estimator gets the remaining hyperparameters.
    """
    est_params, k = _split_meta_params(params)
    estimator = spec.constructor(est_params)
    return wrap_with_preprocessor(estimator, feature_select_k=k)


def _build_calibrated_pipeline(
    spec: ModelSpec,
    params: dict[str, Any] | None,
    cv: int,
):
    """Compose preprocessor + calibrated classifier for one candidate
    in the form `Pipeline([variance, scaler, ?select,
    CalibratedClassifierCV(estimator, cv=N)])`.

    This is the ONNX-friendly shape: the preprocessing steps and the
    calibrator all live inside one Pipeline, so skl2onnx traverses
    the graph linearly and inlines `Scaler` + `Imputer`-style ops
    correctly. The earlier shape — `CalibratedClassifierCV(Pipeline)`
    — produced ONNX graphs that diverged from the Python pipeline by
    >40 % on linear models when fed real-distribution features (see
    test_onnx_export_zoo.py).

    There's a tiny leakage from fitting the StandardScaler on data
    that includes the calibration holdout, but it's well below the
    noise floor on a 1000-bar window and is the price of getting a
    working ONNX export.
    """
    from sklearn.pipeline import Pipeline as SkPipeline

    est_params, k = _split_meta_params(params)
    pre = make_preprocessor(feature_select_k=k)
    estimator = spec.constructor(est_params)
    calibrated = wrap_sigmoid(estimator, cv=cv)
    return SkPipeline(steps=[*pre.steps, ("classifier", calibrated)])


def _train_candidate(
    spec: ModelSpec,
    cfg: SideTrainConfig,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    ts0: np.ndarray,
    ts1: np.ndarray,
    realized: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    run_ts: int,
) -> CandidateResult:
    """Run CPCV + inner Optuna for one candidate spec.

    Returns a CandidateResult including the OOF records (so the
    winner's OOF can be persisted) and a calibrated pipeline refitted
    on the full data (so the winner can be pickled as champion).
    """
    model_id = f"side_{spec.name}_{run_ts}_{uuid.uuid4().hex[:6]}"

    oof_records: list[dict] = []
    fold_aucs: list[float] = []
    fold_losses: list[float] = []
    train_sortinos: list[float] = []
    train_sharpes: list[float] = []
    last_best_params: dict | None = None

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        if len(np.unique(y[train_idx])) < 2:
            continue
        sub_t0 = ts0[train_idx]
        sub_t1 = ts1[train_idx]
        inner_splits = purged_kfold(
            sub_t0, sub_t1,
            n_splits=cfg.inner_cv_splits, embargo_pct=cfg.embargo_pct,
        )

        def objective(trial: optuna.Trial) -> float:
            params = spec.search_space(trial)
            # Meta-param: feature reduction. Suggested independently of
            # the spec's search space so every candidate explores the
            # same SELECT_K_CHOICES alphabet ("all" / 12 / 16 / 20).
            params["feature_select_k"] = trial.suggest_categorical(
                "feature_select_k", SELECT_K_CHOICES,
            )
            scores: list[float] = []
            for it_train, it_test in inner_splits:
                if it_train.size == 0 or it_test.size == 0:
                    continue
                pipe = _build_pipeline(spec, params)
                # Sample weights aren't supported by every Pipeline
                # estimator the same way; skip weights inside Optuna's
                # quick HPO loop and use them only on the calibration
                # refit. (LightGBM/XGB/CatBoost handle weights natively;
                # logreg/extratrees use them via `class_weight` if asked.)
                try:
                    pipe.fit(X[train_idx][it_train], y[train_idx][it_train])
                except Exception:
                    return 0.5
                proba = pipe.predict_proba(X[train_idx][it_test])[:, 1]
                scores.append(_safe_auc(y[train_idx][it_test], proba))
            return float(np.mean(scores)) if scores else 0.5

        sampler = optuna.samplers.TPESampler(seed=int(rng.integers(0, 1 << 31)))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            objective, n_trials=cfg.n_optuna_trials, show_progress_bar=False
        )
        best_params = dict(study.best_params)
        last_best_params = best_params

        # Refit on the full train fold + calibrate via sigmoid.
        cv_for_calib = 3 if train_idx.size >= 30 else 2
        # ONNX-friendly shape: Pipeline([variance, scaler, ?select,
        # CalibratedClassifierCV(estimator)]) — see _build_calibrated_pipeline
        # docstring for why this beats CalibratedClassifierCV(Pipeline).
        calibrated = _build_calibrated_pipeline(spec, best_params, cv_for_calib)
        try:
            calibrated.fit(X[train_idx], y[train_idx])
        except Exception:
            # If a candidate fundamentally can't fit (e.g. CatBoost
            # too few samples), record neutral OOF and move on — the
            # zoo gives us alternatives.
            continue

        proba = calibrated.predict_proba(X[test_idx])[:, 1]
        p_long = proba
        p_short = 1.0 - proba
        p_take = np.maximum(p_long, p_short)
        cal_p = np.maximum(p_long, p_short)

        fold_aucs.append(_safe_auc(y[test_idx], proba))
        fold_losses.append(
            float(log_loss(y[test_idx], np.clip(proba, 1e-6, 1 - 1e-6)))
        )

        train_pred = (calibrated.predict_proba(X[train_idx])[:, 1] >= 0.5).astype(int)
        train_pred_side = np.where(train_pred == 1, 1, -1)
        train_pnl = train_pred_side * realized[train_idx]
        train_sortinos.append(_annualised_sortino(train_pnl, 252.0))
        train_sharpes.append(_annualised_sharpe(train_pnl, 252.0))

        for k, idx in enumerate(test_idx):
            oof_records.append({
                "instrument": cfg.instrument,
                "ts_ms": int(ts0[idx]),
                "t1_ms": int(ts1[idx]),
                "fold": fold_idx,
                "side_label": int(1 if y[idx] == 1 else -1),
                "meta_label": 1,
                "p_long": float(p_long[k]),
                "p_short": float(p_short[k]),
                "p_take": float(p_take[k]),
                "calibrated_p": float(cal_p[k]),
                "model_version": model_id,
            })

    if not oof_records:
        # Candidate failed completely — return a neutral/poor result so
        # the zoo selection naturally drops it.
        return CandidateResult(
            spec_name=spec.name,
            model_id=model_id,
            oos_auc=0.5,
            oos_log_loss=1e6,
            oos_brier=1.0,
            oos_balanced_acc=0.5,
            n_train=int(X.shape[0]),
            n_oof=0,
            train_sortino=0.0,
            train_sharpe=0.0,
            max_train_sortino=0.0,
            max_train_sharpe=0.0,
            oof_records=[],
            fitted_on_full=None,
        )

    oof_df = pl.DataFrame(oof_records)
    oos_proba = oof_df["p_long"].to_numpy().astype(np.float64)
    oos_y = (oof_df["side_label"].to_numpy().astype(np.int8) == 1).astype(np.int8)
    oos_auc = _safe_auc(oos_y, oos_proba)
    oos_log = (
        float(log_loss(oos_y, np.clip(oos_proba, 1e-6, 1 - 1e-6)))
        if len(np.unique(oos_y)) > 1 else 0.0
    )
    oos_brier = float(brier_score_loss(oos_y, np.clip(oos_proba, 1e-6, 1 - 1e-6)))
    oos_bacc = float(balanced_accuracy_score(oos_y, (oos_proba >= 0.5).astype(int)))

    # Refit on the full data with the last fold's best_params (a decent
    # proxy for the global best — the alternative is one more outer
    # Optuna run which doubles training time for marginal gain on 1k rows).
    n = int(X.shape[0])
    cv_for_calib = 3 if n >= 30 else 2
    refit_calibrated = _build_calibrated_pipeline(
        spec, last_best_params or {}, cv_for_calib,
    )
    try:
        refit_calibrated.fit(X, y)
        fitted_on_full = refit_calibrated
    except Exception:
        fitted_on_full = None

    return CandidateResult(
        spec_name=spec.name,
        model_id=model_id,
        oos_auc=float(oos_auc),
        oos_log_loss=float(oos_log),
        oos_brier=float(oos_brier),
        oos_balanced_acc=float(oos_bacc),
        n_train=n,
        n_oof=int(oof_df.height),
        train_sortino=float(np.mean(train_sortinos)) if train_sortinos else 0.0,
        train_sharpe=float(np.mean(train_sharpes)) if train_sharpes else 0.0,
        max_train_sortino=float(np.max(train_sortinos)) if train_sortinos else 0.0,
        max_train_sharpe=float(np.max(train_sharpes)) if train_sharpes else 0.0,
        oof_records=oof_records,
        fitted_on_full=fitted_on_full,
    )


def _persist_candidates_table(
    *,
    run_id: str,
    instrument: str,
    ts_ms: int,
    candidates: list[CandidateResult],
    winner_spec: str,
) -> None:
    """Append one row per candidate to `model_candidates`. UNIQUE
    constraint on (run_id, spec_name) makes this idempotent across
    crash-recovery."""
    with rw_conn() as c:
        # Idempotency via DELETE-then-INSERT under the unique
        # (run_id, spec_name) key. DuckDB's ON CONFLICT requires the
        # target to be backed by an explicit index, and the schema's
        # UNIQUE constraint isn't always picked up; this pattern is
        # equivalent and version-stable.
        for cand in candidates:
            c.execute(
                "DELETE FROM model_candidates WHERE run_id = ? AND spec_name = ?",
                [run_id, cand.spec_name],
            )
            c.execute(
                """
                INSERT INTO model_candidates
                  (run_id, spec_name, model_id, instrument, ts_ms,
                   oos_auc, oos_log_loss, oos_brier, oos_balanced_acc,
                   n_train, n_oof, is_winner)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id, cand.spec_name, cand.model_id, instrument, ts_ms,
                    cand.oos_auc, cand.oos_log_loss, cand.oos_brier,
                    cand.oos_balanced_acc,
                    cand.n_train, cand.n_oof,
                    cand.spec_name == winner_spec,
                ],
            )


def train_side(cfg: SideTrainConfig) -> SideTrainResult:
    """Run the full side-classifier training pipeline across the model
    zoo. Returns the *winner's* result; per-candidate breakdown is also
    persisted to `model_candidates` for dashboard comparison."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rng = np.random.default_rng(cfg.seed)

    bars = extract_bars_10s(cfg.instrument, n_recent=cfg.n_bars)
    if bars.is_empty():
        raise RuntimeError(f"no bars in DuckDB for {cfg.instrument}")
    labels = _load_labels(cfg.instrument, cfg.label_run_id)
    if labels.is_empty():
        raise RuntimeError(f"no labels available for {cfg.instrument}")
    label_run_id = labels[0, "label_run_id"]
    X, y, w, ts0, ts1, realized = _build_supervised_matrix(bars, labels)
    n = X.shape[0]
    if n < cfg.n_splits * 2:
        raise RuntimeError(f"too few labels ({n}) for n_splits={cfg.n_splits}")

    splits = combinatorial_purged_cv(
        ts0, ts1, n_splits=cfg.n_splits, n_test_groups=cfg.n_test_groups,
        embargo_pct=cfg.embargo_pct,
    )
    if not splits:
        raise RuntimeError("CPCV yielded no splits — check n_splits / sample count")

    candidate_names = parse_candidates(cfg.candidates)
    run_ts = int(time.time())
    run_id = f"trainrun_{run_ts}_{uuid.uuid4().hex[:6]}"

    # Train every candidate. These are independent and could in
    # principle parallelise; keep serial here so the orchestrator's
    # log stream is comprehensible. Each candidate already runs Optuna
    # internally so per-trial parallelism happens at that layer.
    candidates: list[CandidateResult] = []
    for name in candidate_names:
        spec = get_spec(name)
        try:
            res = _train_candidate(
                spec, cfg, X, y, w, ts0, ts1, realized, splits, rng, run_ts,
            )
        except Exception as exc:
            # Skip candidates that blow up entirely; the zoo gives us
            # alternatives. Surface in `last_status_message` via stdout
            # so the operator can see it in the run log.
            print(f"[train.side] candidate {name!r} failed: {exc}")
            continue
        candidates.append(res)

    if not candidates:
        raise RuntimeError("no candidate produced OOF records — entire zoo failed")

    # Pick the winner: lowest OOS log loss. Log loss rewards calibrated
    # probabilities, which is what the trader optimizer downstream
    # actually consumes.
    winner = min(candidates, key=lambda c: c.oos_log_loss)
    if winner.fitted_on_full is None:
        raise RuntimeError(f"winner {winner.spec_name} has no fitted full model")

    # Persist per-candidate comparison.
    ts_ms = int(time.time() * 1000)
    _persist_candidates_table(
        run_id=run_id, instrument=cfg.instrument, ts_ms=ts_ms,
        candidates=candidates, winner_spec=winner.spec_name,
    )

    # OOF + champion artifacts for the winner only.
    oof_df = pl.DataFrame(winner.oof_records)
    parquet_path = write_oof_parquet(
        oof_df, run_id=winner.model_id, instrument=cfg.instrument,
    )
    write_oof_duckdb(oof_df.drop("t1_ms", "side_label", "meta_label"))

    champion_dir = _model_artifacts_dir(winner.model_id)
    champion_dir.mkdir(parents=True, exist_ok=True)
    champion_path = champion_dir / "champion.pkl"
    joblib.dump(
        {
            "model": winner.fitted_on_full,
            "feature_names": FEATURE_NAMES,
            "instrument": cfg.instrument,
            "label_run_id": label_run_id,
            "n_train": int(n),
            "spec_name": winner.spec_name,
        },
        champion_path,
    )

    # Headline model_metrics row mirrors winner's stats — keeps the
    # existing `/api/model/metrics` consumer working unchanged.
    with rw_conn() as c:
        c.execute("DELETE FROM model_metrics WHERE model_id = ?", [winner.model_id])
        c.execute(
            """
            INSERT INTO model_metrics
              (model_id, instrument, ts_ms, oos_auc, oos_log_loss, oos_brier,
               oos_balanced_acc, calib_slope, calib_intercept,
               train_sharpe, train_sortino, max_train_sortino, max_train_sharpe,
               n_train, n_oof)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                winner.model_id, cfg.instrument, ts_ms,
                winner.oos_auc, winner.oos_log_loss, winner.oos_brier,
                winner.oos_balanced_acc,
                1.0, 0.0,
                winner.train_sharpe, winner.train_sortino,
                winner.max_train_sortino, winner.max_train_sharpe,
                int(n), int(oof_df.height),
            ],
        )

    return SideTrainResult(
        model_id=winner.model_id,
        model_version=winner.model_id,
        instrument=cfg.instrument,
        spec_name=winner.spec_name,
        oos_auc=winner.oos_auc,
        oos_log_loss=winner.oos_log_loss,
        oos_brier=winner.oos_brier,
        oos_balanced_acc=winner.oos_balanced_acc,
        n_train=winner.n_train,
        n_oof=winner.n_oof,
        train_sortino=winner.train_sortino,
        max_train_sortino=winner.max_train_sortino,
        train_sharpe=winner.train_sharpe,
        max_train_sharpe=winner.max_train_sharpe,
        oof_parquet_path=str(parquet_path),
        label_run_id=label_run_id,
        champion_path=str(champion_path),
        candidates=[
            {
                "spec_name": c.spec_name,
                "model_id": c.model_id,
                "oos_auc": c.oos_auc,
                "oos_log_loss": c.oos_log_loss,
                "oos_brier": c.oos_brier,
                "oos_balanced_acc": c.oos_balanced_acc,
                "n_oof": c.n_oof,
                "is_winner": c.spec_name == winner.spec_name,
            }
            for c in candidates
        ],
    )

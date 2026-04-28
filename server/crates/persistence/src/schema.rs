//! DuckDB DDL. Time stored as BIGINT ms-since-epoch on every table —
//! cheap integer comparisons for the per-instrument FIFO cap and time
//! range queries.
//!
//! Design rules locked by the user:
//!   - HARD 10_000 rows per instrument cap on every per-instrument
//!     time-series table; never exceeded.
//!   - No persisted derivative features; downstream consumers recompute
//!     from `price_ticks` / `bars_10s`.
//!   - Stats / audit tables (`paper_fills`, `model_metrics`,
//!     `trader_metrics`, `optimizer_trials`, `lockbox_results`,
//!     `model_artifacts`, `trade_ledger`, `fitness`) are append-only
//!     with no auto-shedding.

/// PRAGMAs / SET statements run on every open. DuckDB is mostly zero-config:
/// we just bound the WAL to keep checkpoint pressure modest.
pub const PRAGMAS: &str = "
    PRAGMA threads = 4;
    PRAGMA memory_limit = '512MB';
";

pub const SCHEMA: &[&str] = &[
    // ----- Raw market data (10k cap, the only persisted prices) ------------
    "CREATE TABLE IF NOT EXISTS price_ticks (
        instrument  VARCHAR  NOT NULL,
        ts_ms       BIGINT   NOT NULL,
        bid         DOUBLE   NOT NULL,
        ask         DOUBLE   NOT NULL,
        mid         DOUBLE   NOT NULL,
        spread_bp   DOUBLE   NOT NULL,
        status      VARCHAR
    )",
    "CREATE INDEX IF NOT EXISTS idx_price_ticks_instr_ts
        ON price_ticks(instrument, ts_ms)",

    // 10-second OHLCV bars (10k cap). Trading timeframe for the new pipeline.
    "CREATE TABLE IF NOT EXISTS bars_10s (
        instrument     VARCHAR  NOT NULL,
        ts_ms          BIGINT   NOT NULL,
        open           DOUBLE   NOT NULL,
        high           DOUBLE   NOT NULL,
        low            DOUBLE   NOT NULL,
        close          DOUBLE   NOT NULL,
        n_ticks        BIGINT   NOT NULL,
        spread_bp_avg  DOUBLE   NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS idx_bars_10s_instr_ts
        ON bars_10s(instrument, ts_ms)",

    // ----- ML pipeline tables (10k cap on per-instrument rows) -------------
    // Ideal buy/sell entry points produced by the label optimizer over the
    // most recent 1000-bar window. Each retrain rewrites these for that
    // (instrument, label_run_id) pair.
    "CREATE TABLE IF NOT EXISTS labels (
        instrument     VARCHAR  NOT NULL,
        ts_ms          BIGINT   NOT NULL,
        t1_ms          BIGINT   NOT NULL,
        side           TINYINT  NOT NULL,
        meta_y         TINYINT  NOT NULL,
        realized_r     DOUBLE   NOT NULL,
        barrier_hit    VARCHAR  NOT NULL,
        oracle_score   DOUBLE   NOT NULL,
        label_run_id   VARCHAR  NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS idx_labels_instr_ts
        ON labels(instrument, ts_ms)",
    "CREATE INDEX IF NOT EXISTS idx_labels_run
        ON labels(instrument, label_run_id)",

    // Out-of-fold probabilities — one row per (instrument, ts_ms, fold,
    // model_version). The trader fine-tuner reads these to score
    // TraderParams without ever re-fitting the classifier.
    "CREATE TABLE IF NOT EXISTS oof_predictions (
        instrument      VARCHAR  NOT NULL,
        ts_ms           BIGINT   NOT NULL,
        fold            INTEGER  NOT NULL,
        p_long          DOUBLE   NOT NULL,
        p_short         DOUBLE   NOT NULL,
        p_take          DOUBLE   NOT NULL,
        calibrated_p    DOUBLE   NOT NULL,
        model_version   VARCHAR  NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS idx_oof_instr_ts
        ON oof_predictions(instrument, ts_ms)",

    // Live signals from the deployed champion (one per closed bar).
    "CREATE TABLE IF NOT EXISTS signals (
        instrument     VARCHAR  NOT NULL,
        ts_ms          BIGINT   NOT NULL,
        direction      VARCHAR  NOT NULL,
        confidence     DOUBLE   NOT NULL,
        prob_long      DOUBLE   NOT NULL,
        prob_flat      DOUBLE   NOT NULL,
        prob_short     DOUBLE   NOT NULL,
        prob_take      DOUBLE   NOT NULL,
        model_id       VARCHAR  NOT NULL,
        model_version  BIGINT   NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS idx_signals_instr_ts
        ON signals(instrument, ts_ms)",

    // Champion-model live signals (separate table from `signals` so the
    // legacy logreg path and the new ONNX path can be charted side-by-side
    // without conflating model_ids). One row per closed 10s bar per
    // instrument; capped at 10k per instrument like the other live tables.
    "CREATE TABLE IF NOT EXISTS champion_signals (
        instrument    VARCHAR  NOT NULL,
        ts_ms         BIGINT   NOT NULL,
        p_long        DOUBLE   NOT NULL,
        p_short       DOUBLE   NOT NULL,
        p_take        DOUBLE   NOT NULL,
        calibrated    DOUBLE   NOT NULL,
        model_id      VARCHAR  NOT NULL,
        kind          VARCHAR  NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS idx_champion_signals_instr_ts
        ON champion_signals(instrument, ts_ms)",

    // ----- Audit + provenance (append-only, NEVER shed) --------------------
    "CREATE TABLE IF NOT EXISTS paper_fills (
        order_id    VARCHAR  NOT NULL,
        instrument  VARCHAR  NOT NULL,
        ts_ms       BIGINT   NOT NULL,
        units       BIGINT   NOT NULL,
        price       DOUBLE   NOT NULL,
        fee         DOUBLE   NOT NULL,
        mode        VARCHAR  NOT NULL,
        UNIQUE(order_id)
    )",
    "CREATE INDEX IF NOT EXISTS idx_fills_instr_ts
        ON paper_fills(instrument, ts_ms)",

    // Closed trades from the deployed trader / backtester. One row per round
    // trip. Phase D enriches the row with the entry/exit signal +
    // params context that the trader saw when it made the decisions —
    // analysts no longer need to re-join `champion_signals` (which is
    // shed after ~28h on the per-instrument cap).
    "CREATE TABLE IF NOT EXISTS trade_ledger (
        run_id            VARCHAR  NOT NULL,
        instrument        VARCHAR  NOT NULL,
        ts_in_ms          BIGINT   NOT NULL,
        ts_out_ms         BIGINT   NOT NULL,
        side              TINYINT  NOT NULL,
        qty               DOUBLE   NOT NULL,
        entry_px          DOUBLE   NOT NULL,
        exit_px           DOUBLE   NOT NULL,
        fee               DOUBLE   NOT NULL,
        slip              DOUBLE   NOT NULL,
        realized_r        DOUBLE   NOT NULL,
        exit_reason       VARCHAR  NOT NULL,
        -- Phase D enrichment (nullable; existing rows kept their NULLs).
        model_id          VARCHAR,
        params_id         VARCHAR,
        entry_p_long      DOUBLE,
        entry_p_short     DOUBLE,
        entry_calibrated  DOUBLE,
        entry_spread_bp   DOUBLE,
        entry_atr_14      DOUBLE,
        exit_p_long       DOUBLE,
        exit_p_short      DOUBLE,
        decision_chain    VARCHAR,
        snapshot_path     VARCHAR
    )",
    "CREATE INDEX IF NOT EXISTS idx_ledger_instr_ts
        ON trade_ledger(instrument, ts_out_ms)",
    "CREATE INDEX IF NOT EXISTS idx_ledger_model
        ON trade_ledger(model_id)",

    // Per-(model_id) classifier metrics — append-only. One row per training
    // run; the latest row for a given instrument is the current champion.
    "CREATE TABLE IF NOT EXISTS model_metrics (
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
    )",

    // Trader-parameter performance — one row per (params_id) on a given
    // training+fine-tune cycle. The dashboard surfaces both the in-sample
    // figures and the unseen-100 fine-tune figures so the user can see
    // whether the trader generalised or overfit.
    "CREATE TABLE IF NOT EXISTS trader_metrics (
        params_id              VARCHAR  NOT NULL,
        model_id               VARCHAR  NOT NULL,
        ts_ms                  BIGINT   NOT NULL,
        in_sample_sharpe       DOUBLE   NOT NULL,
        in_sample_sortino      DOUBLE   NOT NULL,
        fine_tune_sharpe       DOUBLE   NOT NULL,
        fine_tune_sortino      DOUBLE   NOT NULL,
        max_dd_bp              DOUBLE   NOT NULL,
        turnover_per_day       DOUBLE   NOT NULL,
        hit_rate               DOUBLE   NOT NULL,
        profit_factor          DOUBLE   NOT NULL,
        n_trades               BIGINT   NOT NULL,
        params_json            VARCHAR  NOT NULL,
        UNIQUE(params_id)
    )",

    // Every Optuna trial the trader optimizer evaluates. Useful for the
    // dashboard's optimizer-trial leaderboard and for diagnosing collapse.
    "CREATE TABLE IF NOT EXISTS optimizer_trials (
        study_id        VARCHAR  NOT NULL,
        trial_id        BIGINT   NOT NULL,
        ts_ms           BIGINT   NOT NULL,
        params_json     VARCHAR  NOT NULL,
        score           DOUBLE   NOT NULL,
        sortino         DOUBLE   NOT NULL,
        max_dd_bp       DOUBLE   NOT NULL,
        turnover        DOUBLE   NOT NULL,
        pareto_rank     INTEGER  NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS idx_trials_study_id
        ON optimizer_trials(study_id, trial_id)",

    // Single-shot 100-bar lockbox results. Sealed=true means evaluated and
    // is never re-evaluated for that run_id — write-once.
    "CREATE TABLE IF NOT EXISTS lockbox_results (
        run_id        VARCHAR  NOT NULL,
        ts_ms         BIGINT   NOT NULL,
        model_id      VARCHAR  NOT NULL,
        params_id     VARCHAR  NOT NULL,
        summary_json  VARCHAR  NOT NULL,
        sealed        BOOLEAN  NOT NULL,
        UNIQUE(run_id)
    )",

    // Champion ONNX bytes (and any fallback flavours). The hot-swap loader
    // can either read from the filesystem cache (artifacts/models/) or from
    // this table for cold starts.
    "CREATE TABLE IF NOT EXISTS model_artifacts (
        model_id     VARCHAR  NOT NULL,
        ts_ms        BIGINT   NOT NULL,
        kind         VARCHAR  NOT NULL,
        version      VARCHAR  NOT NULL,
        onnx_blob    BLOB     NOT NULL,
        sha256       VARCHAR  NOT NULL,
        calib_json   VARCHAR  NOT NULL,
        UNIQUE(model_id)
    )",

    // ----- Deployment quality gate ----------------------------------------
    // Every train.side run finishes with an evaluation against
    // configurable performance floors (min OOS AUC, max OOS log loss,
    // min fine-tune Sortino, max drawdown, etc.). If the would-be
    // champion fails ANY floor, `passed_gate=false` and the export
    // step is skipped — the prior champion stays live. This row records
    // the decision so the dashboard can show "would-be champion was
    // blocked because oos_auc 0.51 < floor 0.55".
    "CREATE TABLE IF NOT EXISTS model_deployment_gate (
        model_id              VARCHAR  NOT NULL,
        instrument            VARCHAR  NOT NULL,
        ts_ms                 BIGINT   NOT NULL,
        oos_auc               DOUBLE   NOT NULL,
        oos_log_loss          DOUBLE   NOT NULL,
        oos_brier             DOUBLE   NOT NULL,
        oos_balanced_acc      DOUBLE   NOT NULL,
        fine_tune_sortino     DOUBLE   NOT NULL,
        fine_tune_max_dd_bp   DOUBLE   NOT NULL,
        passed_gate           BOOLEAN  NOT NULL,
        blocked_reasons       VARCHAR  NOT NULL,
        gate_thresholds_json  VARCHAR  NOT NULL,
        UNIQUE(model_id)
    )",
    "CREATE INDEX IF NOT EXISTS idx_deployment_gate_instr
        ON model_deployment_gate(instrument, ts_ms)",

    // ----- Model zoo: per-candidate scores per training run ---------------
    // Every train.side run trains every model in the zoo (LGBM, XGB,
    // CatBoost, LogReg, ExtraTrees by default) and records each
    // candidate's OOS scores here. The winner (lowest OOS log loss)
    // becomes the champion that gets exported to ONNX. Append-only;
    // grows linearly with retrain count, naturally bounded by
    // `pipeline_runs` retention pressure (operator can prune by run_id
    // when needed). The dashboard reads this to show the zoo
    // comparison per retrain.
    "CREATE TABLE IF NOT EXISTS model_candidates (
        run_id            VARCHAR  NOT NULL,
        spec_name         VARCHAR  NOT NULL,
        model_id          VARCHAR  NOT NULL,
        instrument        VARCHAR  NOT NULL,
        ts_ms             BIGINT   NOT NULL,
        oos_auc           DOUBLE   NOT NULL,
        oos_log_loss      DOUBLE   NOT NULL,
        oos_brier         DOUBLE   NOT NULL,
        oos_balanced_acc  DOUBLE   NOT NULL,
        n_train           BIGINT   NOT NULL,
        n_oof             BIGINT   NOT NULL,
        is_winner         BOOLEAN  NOT NULL,
        UNIQUE(run_id, spec_name)
    )",
    "CREATE INDEX IF NOT EXISTS idx_candidates_run
        ON model_candidates(run_id, ts_ms)",
    "CREATE INDEX IF NOT EXISTS idx_candidates_instr
        ON model_candidates(instrument, ts_ms)",

    // ----- Operator visibility into the Python research pipeline ----------
    // One row per `python -m research <cmd>` invocation. Written by the
    // research.observability.tracker context manager — start row at
    // entry, finish row at exit (success or failure). The dashboard
    // surfaces this as the "retrain history" panel so the operator can
    // see what ran, when, how long it took, and whether it succeeded.
    "CREATE TABLE IF NOT EXISTS pipeline_runs (
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
    )",
    "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started
        ON pipeline_runs(ts_started_ms)",

    // ----- Existing tables kept for backwards compatibility ----------------
    // Legacy logreg-fitness rows from the existing strategy. The crate
    // continues to write here while it remains the fallback predictor.
    "CREATE TABLE IF NOT EXISTS fitness (
        instrument         VARCHAR  NOT NULL,
        model_id           VARCHAR  NOT NULL,
        model_version      BIGINT   NOT NULL,
        trained_at_ms      BIGINT   NOT NULL,
        train_samples      BIGINT   NOT NULL,
        train_accuracy     DOUBLE   NOT NULL,
        train_log_loss     DOUBLE   NOT NULL,
        train_sharpe       DOUBLE   NOT NULL,
        train_dist_long    BIGINT   NOT NULL,
        train_dist_flat    BIGINT   NOT NULL,
        train_dist_short   BIGINT   NOT NULL,
        oos_samples        BIGINT   NOT NULL,
        oos_accuracy       DOUBLE   NOT NULL,
        oos_log_loss       DOUBLE   NOT NULL,
        oos_sharpe         DOUBLE   NOT NULL,
        oos_dist_long      BIGINT   NOT NULL,
        oos_dist_flat      BIGINT   NOT NULL,
        oos_dist_short     BIGINT   NOT NULL,
        samples_seen       BIGINT   NOT NULL,
        train_window_lo    BIGINT   NOT NULL,
        train_window_hi    BIGINT   NOT NULL,
        oos_window_lo      BIGINT   NOT NULL,
        oos_window_hi      BIGINT   NOT NULL,
        UNIQUE(model_id, model_version)
    )",

    // Per-poll OANDA account snapshots.
    "CREATE TABLE IF NOT EXISTS account_snapshots (
        ts_ms           BIGINT   NOT NULL,
        nav             DOUBLE   NOT NULL,
        balance         DOUBLE   NOT NULL,
        unrealized_pl   DOUBLE   NOT NULL,
        realized_pl     DOUBLE   NOT NULL,
        margin_used     DOUBLE   NOT NULL,
        margin_avail    DOUBLE   NOT NULL,
        currency        VARCHAR  NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS idx_acct_ts
        ON account_snapshots(ts_ms)",
];

/// Idempotent ALTER-TABLE migrations. Run AFTER every `CREATE TABLE
/// IF NOT EXISTS` in `SCHEMA` so existing tables pick up any columns
/// added after their first creation. Each entry uses
/// `ADD COLUMN IF NOT EXISTS` so re-running on an already-migrated
/// DB is a no-op.
///
/// **Operational discipline (do not violate):**
/// On a schema change, ADD a migration here. NEVER drop and
/// recreate a time-series table (`price_ticks`, `bars_10s`, `labels`,
/// `oof_predictions`, `signals`, `champion_signals`) — those carry
/// the operator's collected market data and recreating them throws
/// it away. Audit tables (`trade_ledger`, `paper_fills`,
/// `model_metrics`, …) likewise carry the trade history. Always
/// migrate.
pub const MIGRATIONS: &[&str] = &[
    // Phase D enrichment: trade_ledger gains entry/exit signal +
    // params + forensic snapshot pointer. Columns are nullable; old
    // rows simply have NULLs in the new fields.
    "ALTER TABLE trade_ledger ADD COLUMN IF NOT EXISTS model_id          VARCHAR",
    "ALTER TABLE trade_ledger ADD COLUMN IF NOT EXISTS params_id         VARCHAR",
    "ALTER TABLE trade_ledger ADD COLUMN IF NOT EXISTS entry_p_long      DOUBLE",
    "ALTER TABLE trade_ledger ADD COLUMN IF NOT EXISTS entry_p_short     DOUBLE",
    "ALTER TABLE trade_ledger ADD COLUMN IF NOT EXISTS entry_calibrated  DOUBLE",
    "ALTER TABLE trade_ledger ADD COLUMN IF NOT EXISTS entry_spread_bp   DOUBLE",
    "ALTER TABLE trade_ledger ADD COLUMN IF NOT EXISTS entry_atr_14      DOUBLE",
    "ALTER TABLE trade_ledger ADD COLUMN IF NOT EXISTS exit_p_long       DOUBLE",
    "ALTER TABLE trade_ledger ADD COLUMN IF NOT EXISTS exit_p_short      DOUBLE",
    "ALTER TABLE trade_ledger ADD COLUMN IF NOT EXISTS decision_chain    VARCHAR",
    "ALTER TABLE trade_ledger ADD COLUMN IF NOT EXISTS snapshot_path     VARCHAR",

    // Binary burn-down (2026-04-26): the labelling pipeline no longer emits
    // side=0 rows, but pre-burn-down rows in this DB still have them. Force
    // a sign so the binary classifier sees only ±1 labels. `realized_r >= 0`
    // → +1; strictly negative → -1. Idempotent: after the first pass no
    // rows match `side = 0`, so this is a no-op on subsequent restarts.
    // Time-series labels are sliced by the 10k retention cap so this self-
    // heals over time, but leaving it here keeps the invariant explicit.
    "UPDATE labels
        SET side = CASE WHEN realized_r >= 0 THEN 1 ELSE -1 END
        WHERE side = 0",
];

/// Tables bounded per-instrument by `MAX_ROWS_PER_INSTRUMENT`. Every
/// retention sweep enforces this HARD cap. `paper_fills` is intentionally
/// not in the list — fills are an audit trail, not a sliding window.
pub const PER_INSTRUMENT_TABLES: &[&str] = &[
    "price_ticks",
    "bars_10s",
    "labels",
    "oof_predictions",
    "signals",
    "champion_signals",
];

/// HARD cap. The retention sweep guarantees no per-instrument table ever
/// holds more than this many rows for a single instrument. User-locked.
pub const MAX_ROWS_PER_INSTRUMENT: i64 = 10_000;

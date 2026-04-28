# trade_logs/ — agent-readable trading record

Lightweight previews of what the live trading system did, per release version,
per instrument. Designed for an LLM agent (or human reviewer) to study without
needing to spin up the full stack or query DuckDB.

**Heavy data lives elsewhere** (gitignored): full per-trade JSON snapshots are
at `data/trades/<run_id>/<ts>.json`, raw ticks are in `data/oanda.duckdb`,
parquet archives are at `data/archive/`.

## Layout

```
trade_logs/
├── README.md                       (this file)
├── v1.0.1/                         (release version — bumps create a new folder)
│   ├── training_log.jsonl          (one line per training run, all instruments)
│   ├── deployment_gate.jsonl       (one line per gate decision)
│   ├── champion_changes.jsonl      (one line per ONNX hot-swap)
│   ├── EUR_USD/
│   │   ├── trades.jsonl            (one line per closed round-trip)
│   │   └── decisions.jsonl         (one line per OPEN/SKIP decision per bar)
│   ├── USD_JPY/...
│   └── ...
└── v1.0.2/                         (when version bumps, new files land here)
```

Each `*.jsonl` file is capped at **2000 most-recent lines**. Older entries are
dropped from the preview but still exist in DuckDB (`trade_ledger`,
`model_metrics`, `model_deployment_gate`, `model_candidates`) and on disk in
`data/trades/`. Treat these files as a sliding window into the system's
behaviour, not the source of truth.

## File schemas

All files are newline-delimited JSON (one record per line). Fields that don't
apply for a given record are omitted (not set to null).

### `<v>/<ticker>/trades.jsonl`

One record per **closed** trade. Written by the live-trader at exit.

| field | type | meaning |
| --- | --- | --- |
| `v` | string | release version (matches folder) |
| `instrument` | string | e.g. `EUR_USD` |
| `run_id` | string | trader runtime id |
| `ts_in_ms` / `ts_out_ms` | i64 | entry / exit timestamps (epoch ms) |
| `side` | i8 | +1 long, −1 short |
| `qty` | f64 | absolute units |
| `entry_px` / `exit_px` | f64 | fill prices |
| `fee` / `slip` | f64 | costs |
| `realized_r` | f64 | net return after costs (fractional) |
| `net_pnl` | f64 | dollar P/L (qty × side × (exit−entry) − fee − slip) |
| `exit_reason` | string | `take_profit`, `stop_loss`, `max_hold`, `reverse`, `manual` |
| `model_id` | string | champion that produced the entry signal |
| `model_kind` | string | `lgbm-onnx`, `xgb-onnx`, `logreg`, … |
| `params_id` | string | trader-params snapshot used at the time |
| `n_features` | i32 | feature-vector width (24 today) |
| `model_oos_auc` | f64 | training-time OOS AUC for `model_id` |
| `model_oos_log_loss` | f64 | OOS log loss for `model_id` |
| `entry_p_long` / `entry_p_short` / `entry_calibrated` | f64 | model probabilities at entry bar |
| `entry_spread_bp` / `entry_atr_14` | f64 | market state at entry |
| `exit_p_long` / `exit_p_short` | f64 | probabilities at the exit bar |
| `decision_chain` | string[] | which rules fired (`"entry:prob_long>=0.62"`, `"exit:take_profit"`) |
| `trader_params` | object | the live `TraderParams` at entry (long_threshold, min_hold_bars, …) |
| `snapshot_path` | string | pointer to the heavy JSON in `data/trades/` (gitignored) |
| `git_sha` | string | repo SHA the api-server was built from |

### `<v>/<ticker>/decisions.jsonl`

One record per **bar** for which the trader produced any decision (OPEN / SKIP /
CLOSE). Lighter than `trades.jsonl` — focused on *why a trade was or was not
taken*. Use this to see how the policy reacts to the probability stream.

| field | type | meaning |
| --- | --- | --- |
| `v` | string | release version |
| `instrument` | string | |
| `ts_ms` | i64 | bar timestamp |
| `action` | string | `OPEN_LONG`, `OPEN_SHORT`, `CLOSE`, `SKIP_THRESHOLD`, `SKIP_COOLDOWN`, `SKIP_HOLD`, `SKIP_SPREAD`, `SKIP_RISK` |
| `p_long` / `p_short` / `calibrated` | f64 | model output |
| `why` | string | short reason text (`"prob_long<long_threshold"`, `"in_cooldown(8/12)"`, …) |
| `model_id` | string | active champion |

### `<v>/training_log.jsonl`

One record per training run, all instruments share this file. Written by the
Python research pipeline.

| field | type | meaning |
| --- | --- | --- |
| `v` | string | |
| `ts_ms` | i64 | finish time |
| `instrument` | string | |
| `run_id` | string | pipeline run id |
| `model_id` | string | new champion id (or candidate id) |
| `model_kind` | string | `lgbm`, `xgb`, …, `lgbm-onnx`, … |
| `n_features` | i32 | |
| `n_train` / `n_oof` | i64 | sample counts |
| `oos_auc` / `oos_log_loss` / `oos_brier` / `oos_balanced_acc` | f64 | classifier metrics |
| `train_sortino` / `fine_tune_sortino` / `fine_tune_max_dd_bp` | f64 | trader metrics |
| `passed_gate` | bool | did this candidate pass the deployment quality gate? |
| `blocked_reasons` | string | comma-separated floor failures, empty if passed |
| `candidates` | object[] | `[{spec, oos_log_loss, is_winner}]` for the entire zoo run |
| `trader_params` | object | params chosen by the fine-tuner |
| `git_sha` | string | |
| `duration_secs` | f64 | wall time of the pipeline run |

### `<v>/deployment_gate.jsonl`

One record per gate evaluation. Records WHY the gate accepted / rejected.

| field | type | meaning |
| --- | --- | --- |
| `v` | string | |
| `ts_ms` | i64 | |
| `instrument` | string | |
| `model_id` | string | candidate that was evaluated |
| `passed` | bool | |
| `blocked` | string[] | per-floor reason text when not passed |
| `thresholds` | object | floors at the time of evaluation |

### `<v>/champion_changes.jsonl`

One record per ONNX hot-swap event in the live api-server.

| field | type | meaning |
| --- | --- | --- |
| `v` | string | |
| `ts_ms` | i64 | |
| `old_model_id` | string | what was serving |
| `new_model_id` | string | what's serving now |
| `kind` | string | `onnx`, `logreg-fallback` |
| `sha256` | string | of the ONNX blob |
| `n_features` | i32 | |

## How to query

```bash
# count trades by exit reason (current version)
jq -s 'group_by(.exit_reason) | map({reason: .[0].exit_reason, n: length})' \
  trade_logs/v1.0.1/EUR_USD/trades.jsonl

# total realized_r per instrument
for f in trade_logs/v1.0.1/*/trades.jsonl; do
  inst=$(basename $(dirname "$f"))
  total=$(jq -s 'map(.realized_r) | add // 0' "$f")
  echo "$inst $total"
done

# all training runs that failed the gate
jq 'select(.passed_gate==false)' trade_logs/v1.0.1/training_log.jsonl
```

DuckDB can read JSONL natively:

```sql
-- via duckdb cli (NB: api-server holds the lock; use a separate connection)
SELECT instrument, COUNT(*) AS n_trades, SUM(realized_r) AS cum_r
FROM read_json_auto('trade_logs/v1.0.1/*/trades.jsonl')
GROUP BY instrument
ORDER BY cum_r DESC;
```

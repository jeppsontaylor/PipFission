# `trade_logs/` — Agent-readable trading record

This directory is a **lightweight, git-checked-in preview** of what the live
PipFission system does, designed for an LLM agent (or human reviewer) to
audit behaviour without spinning up the full stack.

> **TL;DR**:
> - One JSONL file per file type, capped at the 2000 most-recent records.
> - Folder per release version (`v1.0.1/`); bumping
>   `server/Cargo.toml`'s workspace `version` rolls future writes into
>   the next folder.
> - Heavy data (full DuckDB, raw ticks, per-trade JSON snapshots, parquet
>   archives) lives in `data/` (gitignored).
> - Five record types: **features** (per bar), **decisions** (per bar),
>   **trades** (per closed round-trip), **training_log** (per pipeline
>   run), **deployment_gate** + **champion_changes** (system events).

---

## Layout

```
trade_logs/
├── README.md                       (this file)
├── v1.0.1/                         (release-version segmentation)
│   ├── training_log.jsonl          one line / pipeline run
│   ├── deployment_gate.jsonl       one line / gate decision
│   ├── champion_changes.jsonl      one line / ONNX hot-swap
│   ├── EUR_USD/                    one folder per instrument
│   │   ├── features.jsonl          one line / closed 10s bar (raw 24-dim feature vector + champion output)
│   │   ├── decisions.jsonl         one line / closed bar (action + reason)
│   │   └── trades.jsonl            one line / closed round-trip
│   ├── USD_JPY/...
│   └── ...
└── v1.0.2/                         (when version bumps; old folders frozen)
```

Each `*.jsonl` is capped at **2000 most-recent lines** with atomic
roll-off. Older lines are dropped from the preview but still persist
in DuckDB (`trade_ledger`, `model_metrics`, `model_deployment_gate`,
`model_candidates`, `signals`, `champion_signals`, `oof_predictions`)
and on disk under `data/trades/`. Treat these files as a **sliding
window**, not source of truth.

---

## How an agent should review

Recommended sweep order:

1. **STATUS.md** at the repo root — written after each session,
   summarises live state, current champion, what worked, what's next.
2. **`training_log.jsonl`** — see what models have been tried, their
   OOS metrics, the zoo comparison, and which one became champion.
3. **`deployment_gate.jsonl`** — see why a candidate was accepted or
   blocked from going live.
4. **`<ticker>/trades.jsonl`** — actual closed round-trips with full
   model + params context. Look at `realized_r`, `exit_reason`,
   `entry_p_long`/`entry_p_short` to grade the system.
5. **`<ticker>/decisions.jsonl`** — every bar's action. Most rows are
   `skip` with a reason. Use this to study WHEN the trader chose not
   to trade and why (cooldown, below-threshold, spread-too-wide,
   risk-paused, etc.).
6. **`<ticker>/features.jsonl`** — the 24-dim feature vector at each
   bar plus the champion's output (`p_long`, `p_short`,
   `calibrated`). Use this to study WHAT the model saw and how it
   reacted. Pair with decisions/trades to attribute outcomes to
   features.
7. **`champion_changes.jsonl`** — when did the live model swap, and
   to what?

---

## File schemas

All files are **newline-delimited JSON** (one record per line).
Optional fields are omitted, not set to null, except where
`null` itself carries meaning (e.g. `realized_r` on an `open` /
`skip` decision).

### `<v>/<ticker>/features.jsonl` (NEW in v1.0.1)

One record per closed 10s bar for which the inference engine produced
a `ChampionSignal`. **The most useful file for an outside reviewer** —
shows the raw model input + output and lets you reconstruct decision
attribution.

| Field | Type | Meaning |
| --- | --- | --- |
| `v` | string | release version |
| `instrument` | string | e.g. `EUR_USD` |
| `ts_ms` | i64 | bar close timestamp (epoch ms) |
| `close` / `n_ticks` / `spread_bp_avg` | f64/u32/f64 | raw bar context |
| `model_id` | string | active champion |
| `kind` | string | `onnx` / `fallback` |
| `p_long` / `p_short` / `calibrated` | f64 | softmax + Platt-calibrated probs |
| `features` | object | `{name: value}` for all 24 features |

The 24 feature names (from `bar-features::FEATURE_NAMES`):

```
log_ret_1, log_ret_5, log_ret_20, log_ret_60,
sma_dev_30, sma_dev_120, vol_30, vol_120, vol_300,
rsi_14, atr_14, bb_upper_dev, bb_lower_dev,
macd, macd_signal, macd_hist,
range_60, drawdown_300,
log1p_n_ticks, force_index_rel, spread_bp, spread_z,
minute_sin, minute_cos
```

**Notes for agents**:
- Values rounded to 5 decimals to keep lines short. The live engine
  uses full precision; this is a preview.
- `force_index_rel = force_index / rolling_60_n_ticks_mean` (Phase B
  re-engineering — old `force_index` field could hit 9 800).
- `log1p_n_ticks = log(1 + n_ticks)` ranges 0–12.
- Features computed via `bar_features::recompute_last` which is the
  same code path the model trains on, so agent analysis uses the same
  inputs the model sees.

### `<v>/<ticker>/decisions.jsonl`

One record per bar emitting any trader decision (open / close / skip).
Compact — focused on *why* the system chose to act or not.

| Field | Type | Meaning |
| --- | --- | --- |
| `v` | string | |
| `instrument` | string | |
| `ts_ms` | i64 | bar timestamp |
| `action` | string | `open_long`, `open_short`, `close`, `skip` |
| `reason` | string | `signal`, `take_profit`, `stop_loss`, `max_hold`, `min_hold`, `reverse`, `cooldown`, `below_threshold`, `spread_too_wide`, `stale_data`, `daily_loss_kill`, `drawdown_pause`, `trailing_stop` |
| `price` | f64 | bar close (for opens) or fill price (for closes) |
| `realized_r` | f64 \| null | populated on close events |
| `model_id` | string | active champion |
| `params_id` | string | active TraderParams snapshot |

### `<v>/<ticker>/trades.jsonl`

One record per **closed** round-trip. Written at the exit bar.

| Field | Type | Meaning |
| --- | --- | --- |
| `v` | string | |
| `instrument` / `run_id` | string | |
| `ts_in_ms` / `ts_out_ms` | i64 | entry / exit (ms) |
| `side` | i8 | +1 long, −1 short |
| `qty` | f64 | absolute units |
| `entry_px` / `exit_px` / `fee` / `slip` | f64 | fills + costs |
| `realized_r` | f64 | net return after costs (fractional) |
| `net_pnl` | f64 | dollar P/L |
| `exit_reason` | string | `take_profit`, `stop_loss`, `max_hold`, `reverse`, `manual` |
| `model_id` / `model_kind` | string | classifier produced the entry signal |
| `params_id` | string | TraderParams snapshot used |
| `n_features` | i32 | feature width (24 today) |
| `model_oos_auc` / `model_oos_log_loss` | f64 | training-time OOS metrics for `model_id` |
| `entry_p_long` / `entry_p_short` / `entry_calibrated` | f64 | model probs at entry bar |
| `entry_spread_bp` / `entry_atr_14` | f64 | market state at entry |
| `exit_p_long` / `exit_p_short` | f64 | model probs at exit bar |
| `decision_chain` | string[] | rules that fired (e.g. `["entry:signal", "exit:take_profit"]`) |
| `trader_params` | object | full `TraderParams` at entry |
| `snapshot_path` | string | pointer to heavy `data/trades/<run_id>/<ts>.json` (gitignored) |

### `<v>/training_log.jsonl`

One record per **pipeline run** (`python -m research pipeline run`).
Aggregated across all instruments — the `instrument` field tells you
which one each row corresponds to.

| Field | Type | Meaning |
| --- | --- | --- |
| `v` | string | |
| `ts_ms` | i64 | finish time |
| `instrument` | string | |
| `run_id` | string | pipeline run id |
| `model_id` / `model_kind` | string | new champion (or candidate) |
| `n_features` | i32 | |
| `n_train` / `n_oof` | i64 | sample counts |
| `oos_auc` / `oos_log_loss` / `oos_brier` / `oos_balanced_acc` | f64 | classifier OOS metrics |
| `train_sortino` / `fine_tune_sortino` / `fine_tune_max_dd_bp` | f64 | trader metrics |
| `passed_gate` | bool | did this candidate pass deployment gate? |
| `blocked_reasons` | string | comma-separated floor failures |
| `lockbox_passed` | bool | |
| `candidates` | object[] | `[{spec, oos_log_loss, oos_auc, is_winner}]` for the entire zoo run |
| `params_id` | string | |
| `duration_secs` | f64 | wall time |

The **zoo** typically includes: `lgbm`, `xgb`, `catboost`, `logreg`,
`extratrees`, `mlp`, `histgb` (7 candidates per run; the lowest OOS
log loss wins). Tree-based models often dominate; logreg / mlp give
diversity.

### `<v>/deployment_gate.jsonl`

One record per gate evaluation (always paired with a `training_log`
row for the same `model_id`).

| Field | Type | Meaning |
| --- | --- | --- |
| `v` / `ts_ms` / `instrument` / `model_id` | | |
| `passed` | bool | |
| `blocked` | string[] | per-floor reason text when not passed |
| `thresholds` | object | floors at evaluation time (env-overridable) |

Default thresholds (env vars in parens):

```
min_oos_auc            0.55  (MIN_OOS_AUC)
max_oos_log_loss       0.70  (MAX_OOS_LOG_LOSS)
min_oos_balanced_acc   0.52  (MIN_OOS_BALANCED_ACC)
min_fine_tune_sortino  0.30  (MIN_FINE_TUNE_SORTINO)
max_fine_tune_dd_bp    1500  (MAX_FINE_TUNE_DD_BP)
require_lockbox_pass   true  (REQUIRE_LOCKBOX_PASS)
```

### `<v>/champion_changes.jsonl`

One record per ONNX hot-swap event in the live api-server.

| Field | Type | Meaning |
| --- | --- | --- |
| `v` / `ts_ms` / `kind` / `n_features` | | |
| `new_model_id` | string | what's serving now |
| `sha256` | string | SHA-256 of the ONNX blob |

---

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

# average p_long when the trader opened a long position
jq -s 'map(select(.action=="open_long") | .price) | add/length' \
  trade_logs/v1.0.1/EUR_USD/decisions.jsonl

# distribution of feature `rsi_14` over the last 2k bars
jq -r '.features.rsi_14' trade_logs/v1.0.1/EUR_USD/features.jsonl | sort -n
```

DuckDB reads JSONL natively (use a separate process — the live
api-server holds the lock on `data/oanda.duckdb`):

```sql
-- summary across all instruments
SELECT instrument, COUNT(*) AS n_trades,
       SUM(realized_r) AS cum_r,
       SUM(CASE WHEN realized_r > 0 THEN 1 ELSE 0 END) AS wins,
       SUM(CASE WHEN realized_r < 0 THEN 1 ELSE 0 END) AS losses
FROM read_json_auto('trade_logs/v1.0.1/*/trades.jsonl')
GROUP BY instrument
ORDER BY cum_r DESC;

-- correlation: feature vs subsequent return
WITH bars AS (
  SELECT instrument, ts_ms, close, features.rsi_14 AS rsi
  FROM read_json_auto('trade_logs/v1.0.1/EUR_USD/features.jsonl')
)
SELECT corr(rsi, lead_close - close) AS rsi_to_next_close
FROM (SELECT *, LEAD(close) OVER (ORDER BY ts_ms) AS lead_close FROM bars);
```

---

## Versioning

The folder name is the workspace version from
`server/Cargo.toml` `[workspace.package].version`. Bumping that
field and rebuilding the api-server / restarting the python venv
rolls every future write into a new folder. Old folders stay frozen
as historical record for that release. **Never rename or move old
version folders** — agents may be referring to them.

## What is NOT in this folder

- The full DuckDB store (`data/oanda.duckdb`) — gitignored, ~30 MB
- Per-trade forensic JSON snapshots (`data/trades/<run_id>/<ts>.json`)
  — gitignored, one file per closed trade with full pre/in/post-trade
  bar windows. The trades.jsonl row's `snapshot_path` field points
  to these.
- Parquet retention archive (`data/archive/<table>/<instrument>/<date>.parquet`)
  — gitignored, written before retention sweeps shed old time-series
  rows so research queries past the 28h live window remain possible.
- ONNX model artifacts (`research/artifacts/models/<model_id>/`) —
  gitignored. The DB has the bytes in `model_artifacts.onnx_blob`.

If you need any of those, ask the operator — they're available
locally but too heavy for git.

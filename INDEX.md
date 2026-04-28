# PipFission — agent navigation index

If you're an LLM agent reviewing this repo, **start here**. This file
points at the artifacts that explain the system and the data, in
recommended read order.

## 1. Operator briefing

- **[`trade_logs/v1.0.1/PERFORMANCE.md`](trade_logs/v1.0.1/PERFORMANCE.md)** —
  **portfolio P&L at a glance**. Total trades, cum R, Sharpe, Sortino,
  max drawdown, profit factor, per-instrument breakdown table, last 10
  trades across all tickers, exit-reason histogram. Auto-rendered after
  every trade close. **Start here if the question is "is the system
  making money?"**
- **[`trade_logs/v1.0.1/portfolio_summary.json`](trade_logs/v1.0.1/portfolio_summary.json)** —
  the same data as a machine-readable object.
- **[`STATUS.md`](STATUS.md)** — what the system is doing **right
  now**: live champion, current trader params, what was fixed last,
  what's still open. Updated by the human + agent at the end of each
  session.

## 2. The agent-readable trading record

The repo carries a sliding-window preview of every closed trade,
every model retrain, every gate decision, and every per-bar feature
+ champion-output snapshot. Capped at **2000 most-recent records per
file** so the repo stays manageable.

Entry points:

- **[`trade_logs/README.md`](trade_logs/README.md)** — full schema
  for every JSONL/JSON record; query recipes (jq + DuckDB).
- **[`trade_logs/SCHEMA.md`](trade_logs/SCHEMA.md)** — schema
  cheatsheet, single page, copy-pasteable.
- **[`trade_logs/v1.0.1/`](trade_logs/v1.0.1/)** — the data itself.
  The folder name is the release version (from
  `server/Cargo.toml::workspace.package.version`).

Per-ticker quick reads (auto-rendered by the api-server after every
trade close — read these first):

- **`trade_logs/v1.0.1/<ticker>/LATEST.md`** — at-a-glance: cum_R,
  hit rate, last 5 trades, last 5 actions, last 10 features.
- **`trade_logs/v1.0.1/<ticker>/summary.json`** — same data as a
  machine-readable object (consumed by tools and the dashboard).

Per-ticker raw streams (all JSONL, capped at 2000 most-recent records):

- `trade_logs/v1.0.1/<ticker>/trades.jsonl` — closed round-trips
  with full model + params + decision-chain context.
- `trade_logs/v1.0.1/<ticker>/actions.jsonl` — **opens + closes
  only** (`open_long` / `open_short` / `close`). High signal-to-noise.
- `trade_logs/v1.0.1/<ticker>/skip_summary.jsonl` — **compacted**
  runs of consecutive skip bars (count + reason histogram +
  duration). Replaces per-bar skip rows.
- `trade_logs/v1.0.1/<ticker>/features.jsonl` — 24-dim named
  feature vector + champion `p_long` / `p_short` / `calibrated` per
  closed bar.

Cross-instrument streams (written by `pipeline-orchestrator`):

- `trade_logs/v1.0.1/training_log.jsonl` — one line per pipeline
  run: zoo candidates, OOS metrics, winner, gate result, lockbox
  pass/fail, trader params, duration.
- `trade_logs/v1.0.1/deployment_gate.jsonl` — gate decisions +
  thresholds (from the Rust `deploy-gate` crate).
- `trade_logs/v1.0.1/champion_changes.jsonl` — ONNX hot-swap events.

## 3. Architecture + design

- **[`CLAUDE.md`](CLAUDE.md)** — project rules + always-on
  architecture (supervisor + api-server + sentinel handoff). Has the
  env-var reference and the "what stays Python and why" split.
- **[`tips/`](tips/)** — design source documents (purged CPCV,
  triple-barrier labelling, two-model architecture, deflated Sharpe,
  ONNX-in-Rust, …). 12 markdown files, 10–30 KB each.

## 4. Source tree

- `server/` — Rust workspace. Two binaries the operator runs directly:
  - `target/release/api-server` — always-on streaming + REST/WS +
    live-trader. Exits 75 to ask for a retrain.
  - `target/release/pipeline-orchestrator` — Rust retrain entry point.
    Owns observability + JSONL artifacts; spawns
    `python -m research pipeline run` for the ML core.
- `server/crates/` — 18-crate Rust workspace. Notable crates:
  - `api-server` — REST/WS + DuckDB + ONNX + live-trader.
  - `pipeline-orchestrator` — see above.
  - `deploy-gate` — pure-Rust port of the Python deployment gate
    (OOS floors, lockbox check).
  - `lockbox` — single-shot sealed-slice holdout (composes inference
    + backtest + metrics).
  - `observability` — `RunTracker` for `pipeline_runs` row tracking.
  - `metrics` — includes `metrics::pbo` (CSCV-based probability of
    backtest overfitting) and `metrics::bootstrap` (percentile CI).
  - plus persistence (DuckDB), inference (ONNX via `ort`),
    bar-aggregator, labeling, backtest, trader, …
- `research/` — Python pipeline (label → train → finetune → lockbox
  → export). Retained for the hard-constraint ML libs only:
  LightGBM/XGBoost/CatBoost, sklearn (LogReg/MLP/ExtraTrees/HistGB/
  CalibratedClassifierCV/SelectKBest), skl2onnx + onnxmltools, Optuna
  (TPE side-training, NSGA-II trader fine-tune). Everything else
  (gate, lockbox composition, observability, JSONL artifacts, stats)
  is Rust now. See [`STATUS.md`](STATUS.md) "Rust rewrite" section.
- `dashboard/` — Vite + React + TypeScript live dashboard.
- `scripts/` — operator scripts:
  - `api-server-supervisor.sh` — **production launcher**. Runs
    `api-server` in a loop; on exit 75, drains
    `data/.retrain-pending` one instrument at a time via
    `pipeline-orchestrator`, then restarts the api-server. Sets the
    relaxed-regime env vars (`AUTO_RETRAIN_VIA_SENTINEL`,
    `MIN_OOS_AUC`, `MAX_OOS_LOG_LOSS`, `REQUIRE_LOCKBOX_PASS`,
    `RELAX_TRADER_PARAMS`).
  - `sync_trade_logs.sh` — opt-in periodic git push of `trade_logs/`.
  - `com.oanda.api-server.plist` / `com.oanda.trade-logs-sync.plist`
    — launchd templates (not installed by default).

## 5. Recommended agent read order

1. `STATUS.md` (current state)
2. `trade_logs/README.md` (data semantics)
3. `trade_logs/v1.0.1/<ticker>/LATEST.md` for whichever ticker is
   interesting (e.g. EUR_USD)
4. Drill into `trades.jsonl` / `features.jsonl` for that ticker if
   the LATEST view shows something worth investigating
5. Cross-reference against `training_log.jsonl` to see what model
   produced those decisions
6. `CLAUDE.md` for env / architecture context if needed

## 6. Common ad-hoc queries

```bash
# Cumulative R per ticker (current version)
for f in trade_logs/v1.0.1/*/trades.jsonl; do
  inst=$(basename $(dirname "$f"))
  echo "$inst $(jq -s 'map(.realized_r)|add // 0' "$f")"
done

# Models that failed the deployment gate, with reasons
jq 'select(.passed_gate==false) | {model_id, blocked_reasons}' \
  trade_logs/v1.0.1/training_log.jsonl

# Latest p_long per ticker
for f in trade_logs/v1.0.1/*/features.jsonl; do
  inst=$(basename $(dirname "$f"))
  echo "$inst $(tail -1 "$f" | jq -r '.p_long')"
done

# How many bars did the trader skip vs act, last 24h?
jq -s 'length' trade_logs/v1.0.1/EUR_USD/actions.jsonl
jq -s 'map(.count)|add' trade_logs/v1.0.1/EUR_USD/skip_summary.jsonl
```

DuckDB reads JSONL natively (use a separate process — the live
api-server holds the lock on `data/oanda.duckdb`):

```sql
SELECT instrument, COUNT(*) AS n_trades,
       SUM(realized_r) AS cum_r,
       AVG(realized_r) AS avg_r,
       SUM(CASE WHEN realized_r > 0 THEN 1 ELSE 0 END) AS wins
FROM read_json_auto('trade_logs/v1.0.1/*/trades.jsonl')
GROUP BY instrument
ORDER BY cum_r DESC;
```

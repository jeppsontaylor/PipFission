# PipFission — agent navigation index

If you're an LLM agent reviewing this repo, **start here**. This file
points at the artifacts that explain the system and the data, in
recommended read order.

## 1. Operator briefing

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

Per-ticker quick reads (auto-rendered after every trade):

- **`trade_logs/v1.0.1/<ticker>/LATEST.md`** — at-a-glance: cum_R,
  hit rate, last 5 trades, last 5 actions, last 10 features.
- **`trade_logs/v1.0.1/<ticker>/summary.json`** — same data as a
  machine-readable object.

Per-ticker raw streams:

- `trade_logs/v1.0.1/<ticker>/trades.jsonl` — closed round-trips
  with full model + params + decision-chain context.
- `trade_logs/v1.0.1/<ticker>/actions.jsonl` — `open_long` /
  `open_short` / `close` events only (signal-to-noise high).
- `trade_logs/v1.0.1/<ticker>/skip_summary.jsonl` — compacted runs
  of consecutive skip bars (count + reason histogram + duration).
- `trade_logs/v1.0.1/<ticker>/features.jsonl` — 24-dim named
  feature vector + champion `p_long` / `p_short` / `calibrated` per
  closed bar.

Cross-instrument streams:

- `trade_logs/v1.0.1/training_log.jsonl` — one line per pipeline
  run: zoo candidates, OOS metrics, winner, gate result, lockbox
  pass/fail, trader params, duration.
- `trade_logs/v1.0.1/deployment_gate.jsonl` — gate decisions +
  thresholds.
- `trade_logs/v1.0.1/champion_changes.jsonl` — ONNX hot-swap events.

## 3. Architecture + design

- **[`CLAUDE.md`](CLAUDE.md)** — project rules + always-on
  architecture. Has the env-var reference for the api-server.
- **[`tips/`](tips/)** — design source documents (purged CPCV,
  triple-barrier labelling, two-model architecture, deflated Sharpe,
  ONNX-in-Rust, …). 12 markdown files, 10–30 KB each.

## 4. Source tree

- `server/crates/` — 18-crate Rust workspace. Always-on api-server,
  persistence (DuckDB), inference (ONNX via `ort`), live-trader,
  bar-aggregator, labeling, backtest, trader, metrics, …
- `research/` — Python pipeline (label → train → finetune → lockbox
  → export → deployment gate). Most of this will migrate to Rust;
  see [`STATUS.md`](STATUS.md) "Rust rewrite" section.
- `dashboard/` — Vite + React + TypeScript live dashboard.
- `scripts/` — operator scripts:
  - `api-server-supervisor.sh` — production launcher (handles the
    auto-retrain handoff via sentinel + exit 75).
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

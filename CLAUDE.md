# Project rules for Claude

These rules apply to every Claude session that opens this folder.

## Hard rules

### 1. File length limit: 1500 LOC

No source file in this project may exceed **1500 lines of code**. If a file
crosses that line, refactor it before continuing — split by responsibility
into a module/folder, extract helpers, or move types into their own file.
This applies to every language used in the repo (Python, Rust, TypeScript,
TSX, etc.) but excludes generated files, lock files, vendored data, and JSONL
data dumps under `oanda_data/` and `data_chunks/`.

When approaching the limit (≥ 1200 lines), proactively flag it and propose a
refactor rather than waiting until the file blows past 1500.

How to check the active source files:

```bash
find . -type f \
  \( -name '*.py' -o -name '*.rs' -o -name '*.ts' -o -name '*.tsx' \
     -o -name '*.js' -o -name '*.jsx' \) \
  -not -path './*/node_modules/*' \
  -not -path './*/target/*' \
  -not -path './venv/*' -not -path './.venv/*' \
  -not -path './oanda_data/*' -not -path './data_chunks/*' \
  -exec wc -l {} + | sort -nr | head
```

## Project layout

- `oanda_collector.py` / `build_features.py` / `analyze_data.py` — original
  Python data collection + ML feature pipeline.
- `server/` — Rust streaming server (axum + reqwest). Connects to OANDA v20
  pricing + transactions streams, polls account summary, runs an
  estimated-vs-actual reconciler, fans the result out over a WebSocket. The
  server also owns the auto-retrain orchestrator and bridges to the Python
  research layer via `tokio::process::Command`.
- `research/` — Python research layer (uv-style project at `research/`,
  shared `.venv/` at the repo root). End-to-end pipeline: `label →
  train → finetune → lockbox → export`. Driven by `python -m research`.
- `dashboard/` — Vite + React + TypeScript live dashboard. Connects to the
  Rust server's `/ws` endpoint.
- `data/oanda.duckdb` — DuckDB store. HARD per-instrument cap of 10 000
  rows on every time-series table; user-locked. Audit tables (`paper_fills`,
  `trade_ledger`, `model_metrics`, `trader_metrics`, `optimizer_trials`,
  `lockbox_results`, `model_artifacts`, `pipeline_runs`) are append-only.
- `data/logs/pipeline/` — captured stdout/stderr per pipeline subprocess
  (one file per `run_id`). Read via `GET /api/pipeline/log`.
- `oanda_data/` — raw JSONL captures (gitignored, do not edit).
- `data_chunks/` — generated feature CSVs (gitignored, do not edit).

## OANDA env

Credentials are read from `.env` at the repo root:

- `OANDA_API_TOKEN` (required)
- `OANDA_ACCOUNT_ID` (optional — auto-discovered if absent)
- `OANDA_ENV` = `practice` | `live` (default: `practice`)

## Always-on architecture

The system is split into **two independent processes**:

1. **`api-server` (Rust, always-on)** — owns the OANDA stream subscription,
   tick → 10s-bar aggregation, DuckDB persistence, retention sweep, ONNX
   inference, hot-swap watcher, live-trader, auto-retrain orchestrator,
   and the entire REST/WebSocket surface. **This is the autonomy loop.**
   It runs as a `nohup` background process and **does not need a
   terminal, browser, or dashboard to keep running.**
2. **`vite` (dashboard, optional)** — a viewer for the api-server. Closing
   the browser does NOT stop the system. Stopping the dev server does
   NOT stop the system. The dashboard reads `/api/*` and WebSocket
   events from the api-server.

If the dashboard is closed, the api-server keeps:
- collecting ticks → bars → DuckDB,
- counting bars and firing auto-retrain when the threshold is hit,
- producing `ChampionSignal` and `TraderDecision` events,
- hot-swapping new ONNX champions when the research pipeline publishes
  one,
- writing `trade_ledger` + per-trade JSON snapshots,
- enforcing the 10 000-row retention cap per instrument.

### Starting the api-server (one-shot)

```bash
cd /Users/bentaylor/Code/oanda
DATABASE_PATH=./data/oanda.duckdb \
  LIVE_TRADER_ENABLED=true \
  PIPELINE_TRIGGER_ENABLED=true \
  AUTO_RETRAIN_ENABLED=true \
  AUTO_RETRAIN_BARS_THRESHOLD=100 \
  AUTO_RETRAIN_INSTRUMENTS=EUR_USD,USD_JPY,GBP_USD \
  RESEARCH_ARCHIVE_DIR=./data/archive \
  nohup ./server/target/release/api-server \
    > ./data/logs/server/api-server.log 2>&1 &
```

`nohup` decouples it from the terminal. The process is yours to manage:

```bash
pgrep -lf 'target/release/api-server'      # is it running?
pkill -INT -f 'target/release/api-server'  # stop it (clean SIGINT)
tail -f data/logs/server/api-server.log    # watch the log
```

### Auto-start on boot (macOS launchd)

To survive a reboot, install the bundled launchd plist:

```bash
cp scripts/com.oanda.api-server.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.oanda.api-server.plist
launchctl list | grep oanda  # confirm registered
```

After this, `launchd` brings the api-server up at login + restarts it
automatically on crash. Stop it with `launchctl unload …`.

### Data preservation

DuckDB tables fall into two classes:

- **Time-series tables** (`price_ticks`, `bars_10s`, `labels`,
  `oof_predictions`, `signals`, `champion_signals`) — bounded at
  **10 000 rows per instrument** by the retention sweep. Older rows are
  rolled off, never the whole table. When `RESEARCH_ARCHIVE_DIR` is
  set, the soon-to-be-shed rows are **archived to parquet first** at
  `<dir>/<table>/<instrument>/<YYYY-MM-DD>.parquet` so they remain
  queryable.
- **Audit tables** (`paper_fills`, `trade_ledger`, `model_metrics`,
  `trader_metrics`, `optimizer_trials`, `lockbox_results`,
  `model_artifacts`, `pipeline_runs`, `model_candidates`,
  `model_deployment_gate`, `fitness`, `account_snapshots`) —
  append-only, never shed.

**Schema migrations** add columns through `schema::MIGRATIONS` (idempotent
`ALTER TABLE … ADD COLUMN IF NOT EXISTS`). DROP + recreate of a
time-series or audit table during a migration is forbidden — it
discards the operator's collected data.

## ML pipeline / autonomy loop

The streaming server runs an end-to-end ML pipeline that retrains itself
in the background and hot-swaps a new ONNX champion into the live path.
This section is the operator interface.

### Run the full pipeline once (manual)

```bash
source .venv/bin/activate
python -m research pipeline run --instrument EUR_USD
```

The orchestrator sequences `label → train.side → finetune → lockbox →
export.champion`, threads `model_id` / `params_id` between stages
automatically, and writes one parent `pipeline.full` row plus per-stage
rows to DuckDB.`pipeline_runs`. Lockbox is treated as a gate — a failing
lockbox keeps the prior champion live (`published=False` in the report,
exit code 1). Override with `--publish-on-lockbox-fail` for development
runs.

Individual stages still work: `python -m research {label,train,finetune,
lockbox,export} ...` — useful when iterating on a single component.

### Trigger from the dashboard

`POST /api/pipeline/run` shells out to the same orchestrator. Requires
`PIPELINE_TRIGGER_ENABLED=true` (off by default — keeps the route
disabled in shared environments where shelling out to Python is not
desired). The dashboard's `PipelineRunsCard` exposes a "Run pipeline"
button when that flag is set.

### Auto-retrain (Bar10s-driven)

The api-server subscribes to `Event::Bar10s`, counts closed bars per
configured instrument, and fires `python -m research pipeline run` when
the threshold is crossed. Single-flight: if a run is already in flight
(manual or auto), the next decision is recorded as a "skipped" reason
on `/api/pipeline/auto-retrain` and the counter keeps ticking until the
slot frees up.

Env switches (all opt-in, default off):

- `AUTO_RETRAIN_ENABLED=true`         — master switch. When false, bars
  are still counted (telemetry) but no subprocess fires.
- `AUTO_RETRAIN_BARS_THRESHOLD=100`   — bars per instrument before a fire.
- `AUTO_RETRAIN_INSTRUMENTS=EUR_USD,USD_JPY,BTC_USD` — comma-separated
  watchlist. Bars for instruments outside the list are ignored.
- `PIPELINE_TRIGGER_ENABLED=true`     — also required (auto-retrain
  routes through the same `spawn_pipeline_subprocess` as the REST
  trigger, which honors this gate).
- `PIPELINE_PYTHON_BIN=./.venv/bin/python` — override the venv.
- `PIPELINE_RESEARCH_DIR=./research`  — override the working dir.
- `PIPELINE_LOG_DIR=./data/logs/pipeline` — override per-run log dir.

### Champion deployment

When `export.champion` writes a fresh ONNX to
`research/artifacts/models/live/champion.onnx`, the api-server's
`inference::hot_swap` notify-watcher picks it up, validates the
predictor, and atomically swaps it into the live `PredictorRegistry`.
A `ChampionChanged` event flows over `/ws`; the dashboard's
`ChampionBanner` shows a green toast for ~30s.

If the load fails (bad sha, shape mismatch), `ChampionLoadFailed` flows
over `/ws` and the rose error banner stays up until the next successful
swap. The previous champion (or the neutral fallback) keeps serving in
the meantime — live decisions never go silent.

### Live trader + safety chain (real orders)

The live-trader emits `Event::TraderDecision` per closed bar; the
champion-router converts those into OrderIntents. Routing real orders
to OANDA practice requires **all three** flags to be set, in order:

1. `LIVE_TRADER_ENABLED=true`        — turns on the live-trader runner.
2. `CHAMPION_ROUTING_ENABLED=true`   — routes TraderDecision → OrderIntent.
3. `ALLOW_OANDA_ROUTING=true`        — gates the OANDA-practice router
   itself (also required by the manual order panel + mode toggle).

Plus the operator must flip the dashboard mode switch to "OANDA
practice." Practice account only — there is no live-money path.

### REST surface (dashboard reads)

- `GET  /api/strategy/champion`        — current champion id + kind.
- `GET  /api/model/metrics?instrument=`— last classifier-training row.
- `GET  /api/trader/metrics`           — last trader fine-tune row.
- `GET  /api/optimizer/trials?study=&limit=` — Pareto leaderboard.
- `GET  /api/lockbox/result`           — most recent sealed lockbox.
- `GET  /api/labels/recent?instrument=&limit=` — optimizer entry points.
- `GET  /api/champion/signals?instrument=&limit=` — recent live preds.
- `GET  /api/trade/ledger?instrument=&limit=` — closed round-trips.
- `GET  /api/pipeline/runs?limit=`     — recent Python invocations.
- `GET  /api/pipeline/status`          — `{enabled, current?}` flight.
- `GET  /api/pipeline/last-completed`  — most recent finished flight.
- `GET  /api/pipeline/log?run_id=&tail=` — captured stdout/stderr tail.
- `GET  /api/pipeline/auto-retrain`    — per-instrument counters.
- `POST /api/pipeline/run`             — trigger an orchestrator run.

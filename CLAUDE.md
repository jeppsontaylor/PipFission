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
  Python data collection + ML feature pipeline (legacy; superseded by the
  Rust server + research layer below).
- `server/` — Rust workspace. Always-on `api-server` binary plus a growing
  set of crates that own labelling, backtest, inference, the live-trader,
  the deployment gate, and the pipeline orchestrator. Bridges to the
  Python research layer via `tokio::process::Command` only for the
  hard-constraint ML libs (see "What stays Python" below).
- `server/crates/` — 18-crate Rust workspace. Notable crates:
  - `api-server` — axum REST + WebSocket surface, OANDA v20 streams,
    DuckDB persistence, retention sweep, ONNX hot-swap, live-trader.
  - `pipeline-orchestrator` — Rust binary entry point for retrain runs.
    Owns observability + JSONL artifacts in Rust, then spawns
    `python -m research pipeline run` for the ML core.
  - `deploy-gate` — pure-Rust port of `research/deployment/gate.py`
    (OOS floors, lockbox check).
  - `lockbox` — single-shot sealed-slice holdout. Composes inference +
    backtest + metrics.
  - `observability` — `RunTracker` for `pipeline_runs` row tracking
    (started/finished/failed, per-stage rows).
  - `metrics` — includes `metrics::pbo` (probability of backtest
    overfitting via CSCV) and `metrics::bootstrap` (percentile
    bootstrap CI).
- `research/` — Python research layer (uv-style project at `research/`,
  shared `.venv/` at the repo root). End-to-end pipeline: `label →
  train → finetune → lockbox → export`. Driven by `python -m research`.
- `dashboard/` — Vite + React + TypeScript live dashboard. Connects to the
  Rust server's `/ws` endpoint.
- `data/oanda.duckdb` — DuckDB store. HARD per-instrument cap of 10 000
  rows on every time-series table; user-locked. Audit tables (`paper_fills`,
  `trade_ledger`, `model_metrics`, `trader_metrics`, `optimizer_trials`,
  `lockbox_results`, `model_artifacts`, `pipeline_runs`) are append-only.
- `data/.retrain-pending` — sentinel file written by the api-server when
  it wants the supervisor to run the orchestrator. One line per
  instrument; drained one at a time. Do not edit by hand.
- `data/logs/pipeline/` — captured stdout/stderr per pipeline subprocess
  (one file per `run_id`). Read via `GET /api/pipeline/log`.
- `data/logs/server/api-server.log` — supervisor-managed api-server log.
- `trade_logs/v<version>/<ticker>/` — agent-readable trading record. See
  [`INDEX.md`](INDEX.md) for the full layout. Auto-written by the
  api-server + pipeline-orchestrator.
- `scripts/api-server-supervisor.sh` — production launcher (loop +
  sentinel handoff). See "Always-on architecture" below.
- `oanda_data/` — raw JSONL captures (gitignored, do not edit).
- `data_chunks/` — generated feature CSVs (gitignored, do not edit).

## What stays Python (and why)

The Rust port is in flight. Anything that depends on hard-constraint ML
libs without a usable Rust equivalent stays in `research/`:

- **Gradient-boosted trees** — LightGBM, XGBoost, CatBoost.
- **sklearn classical models** — LogReg, MLP, ExtraTrees, HistGB,
  CalibratedClassifierCV, SelectKBest.
- **ONNX export** — `skl2onnx` + `onnxmltools` (the only path to
  produce the ONNX files the Rust live-inference loop consumes).
- **Hyperparameter search** — Optuna (TPE for side training, NSGA-II
  for trader fine-tune).

Everything else — observability, deployment gate, lockbox composition,
PBO/bootstrap stats, JSONL artifact writing, pipeline run tracking — is
Rust. The `pipeline-orchestrator` Rust binary owns the run lifecycle
and only shells out to Python for the ML kernel.

## OANDA env

Credentials are read from `.env` at the repo root:

- `OANDA_API_TOKEN` (required)
- `OANDA_ACCOUNT_ID` (optional — auto-discovered if absent)
- `OANDA_ENV` = `practice` | `live` (default: `practice`)

## Always-on architecture

The system is split into **three cooperating processes**:

1. **`api-server-supervisor.sh` (bash, always-on)** — production
   launcher. Runs `api-server` in a loop. When the api-server exits
   with code **75**, the supervisor reads `data/.retrain-pending`,
   drains it **one instrument at a time**, runs
   `./server/target/release/pipeline-orchestrator --instrument <X>` for
   each, then restarts the api-server. Any other exit is treated as a
   crash and triggers an immediate restart.
2. **`api-server` (Rust)** — owns the OANDA stream subscription,
   tick → 10s-bar aggregation, DuckDB persistence, retention sweep,
   ONNX inference, hot-swap watcher, live-trader, JSONL artifact
   writer, and the entire REST/WebSocket surface. **This is the
   autonomy loop.** It does not need a terminal, browser, or dashboard
   to keep running. When auto-retrain is needed it writes the sentinel
   and exits 75 — the supervisor handles the rest.
3. **`vite` (dashboard, optional)** — a viewer for the api-server.
   Closing the browser does NOT stop the system. Stopping the dev
   server does NOT stop the system. The dashboard reads `/api/*` and
   WebSocket events from the api-server.

If the dashboard is closed, the supervisor + api-server keep:
- collecting ticks → bars → DuckDB,
- counting bars and firing auto-retrain via sentinel + exit 75,
- producing `ChampionSignal` and `TraderDecision` events,
- hot-swapping new ONNX champions when the orchestrator publishes one,
- writing `trade_ledger` + per-trade JSON + `trade_logs/v*/<ticker>/*`,
- enforcing the 10 000-row retention cap per instrument.

### Starting the system (production)

```bash
cd /Users/bentaylor/Code/oanda
nohup ./scripts/api-server-supervisor.sh \
  > ./data/logs/server/supervisor.log 2>&1 &
```

The supervisor sets the env vars the api-server needs (auto-retrain
flags, gate floors, sentinel mode). To inspect or stop:

```bash
pgrep -lf 'api-server-supervisor.sh'       # is the supervisor up?
pgrep -lf 'target/release/api-server'      # is the child up?
pkill -f 'api-server-supervisor.sh'        # stop the loop
pkill -INT -f 'target/release/api-server'  # ask the child to exit cleanly
tail -f data/logs/server/api-server.log    # watch the api-server log
tail -f data/logs/server/supervisor.log    # watch the supervisor log
```

### Auto-retrain handoff (sentinel + exit 75)

When `AUTO_RETRAIN_VIA_SENTINEL=true` is set (the supervisor sets
this), the api-server's auto-retrain path no longer spawns a Python
subprocess in-process. Instead:

1. The bar counter for an instrument crosses
   `AUTO_RETRAIN_BARS_THRESHOLD`.
2. The api-server appends the instrument to `data/.retrain-pending`
   (one instrument per line, deduped).
3. The api-server flushes its open WebSocket sinks and **exits with
   code 75**.
4. The supervisor sees exit 75, reads the sentinel, runs
   `./server/target/release/pipeline-orchestrator --instrument <X>`
   for **one** instrument, removes that line from the sentinel.
5. The orchestrator (Rust) creates the `pipeline_runs` row, writes the
   `trade_logs/v*/<ticker>/training_log.jsonl` entry, then spawns
   `python -m research pipeline run --instrument <X>` for the ML core.
6. On orchestrator success, the new `champion.onnx` is in place. The
   supervisor restarts the api-server, which picks up the new
   champion via the hot-swap watcher on startup.
7. If the sentinel still has lines, the next retrain is processed on
   the next exit-75 cycle (one at a time, never concurrent).

This keeps the live trading process and the heavyweight retrain
process strictly separate — no subprocess management inside the
api-server, no GIL contention, no orphaned Python procs on api-server
crashes.

### Auto-start on boot (macOS launchd)

To survive a reboot, install the bundled launchd plist (it now points
at the supervisor, not the api-server directly):

```bash
cp scripts/com.oanda.api-server.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.oanda.api-server.plist
launchctl list | grep oanda  # confirm registered
```

After this, `launchd` brings the supervisor up at login + restarts it
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

The system retrains itself in the background and hot-swaps a new ONNX
champion into the live path. The orchestration is now Rust
(`pipeline-orchestrator` binary); the ML kernel is still Python
(`python -m research pipeline run`). This section is the operator
interface.

### Run the full pipeline once (manual)

Production path (Rust orchestrator owns observability + JSONL):

```bash
./server/target/release/pipeline-orchestrator --instrument EUR_USD
```

Bare ML kernel (skips the Rust observability wrapping — useful for
iterating on a single Python stage):

```bash
source .venv/bin/activate
python -m research pipeline run --instrument EUR_USD
```

Either path sequences `label → train.side → finetune → lockbox →
export.champion`, threads `model_id` / `params_id` between stages, and
writes one parent `pipeline.full` row plus per-stage rows to
`pipeline_runs`. Lockbox is treated as a gate — a failing lockbox keeps
the prior champion live. The orchestrator additionally writes
`trade_logs/v*/training_log.jsonl` + `deployment_gate.jsonl`.

Individual Python stages still work: `python -m research
{label,train,finetune,lockbox,export} ...`.

### Trigger from the dashboard

`POST /api/pipeline/run` queues a sentinel-driven retrain. Requires
`PIPELINE_TRIGGER_ENABLED=true`. The dashboard's `PipelineRunsCard`
exposes a "Run pipeline" button when that flag is set.

### Auto-retrain (Bar10s-driven, sentinel handoff)

The api-server subscribes to `Event::Bar10s` and counts closed bars per
configured instrument. When the threshold is crossed it appends the
instrument to `data/.retrain-pending` and exits with code 75; the
supervisor drains the sentinel one instrument at a time (see
"Auto-retrain handoff" above).

Env switches (the supervisor sets these; values shown are the
production defaults for the current thin-data regime):

- `AUTO_RETRAIN_ENABLED=true`         — master switch.
- `AUTO_RETRAIN_VIA_SENTINEL=true`    — exit 75 + sentinel instead of
  in-process Python spawn. **Required in production.**
- `AUTO_RETRAIN_BARS_THRESHOLD=100`   — bars per instrument before a fire.
- `AUTO_RETRAIN_INSTRUMENTS=EUR_USD,USD_JPY,GBP_USD,…` — watchlist.
- `PIPELINE_TRIGGER_ENABLED=true`     — also required for REST triggers.
- `PIPELINE_PYTHON_BIN=./.venv/bin/python`
- `PIPELINE_RESEARCH_DIR=./research`
- `PIPELINE_LOG_DIR=./data/logs/pipeline`

### Deployment gate (relaxed thresholds)

The Rust `deploy-gate` crate evaluates each candidate against OOS
floors and an optional lockbox-pass requirement. Current floors are
relaxed for the thin-data regime — tighten them as the data volume
grows. The supervisor sets:

- `MIN_OOS_AUC=0.40`            — minimum out-of-sample AUC.
- `MAX_OOS_LOG_LOSS=1.0`        — maximum out-of-sample log loss.
- `REQUIRE_LOCKBOX_PASS=false`  — when true, lockbox failure blocks
  publication. False today because the sealed slice is too small to
  produce a stable lockbox verdict.
- `RELAX_TRADER_PARAMS=true`    — after each retrain, re-applies a
  workable threshold set so the live-trader actually opens trades on
  thin data instead of skipping every bar.

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

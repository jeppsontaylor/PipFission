# PipFission status as of 2026-04-28 ~11:40 MDT

## TL;DR

The system is **alive but not trading**. Live api-server runs continuously
and writes a decision per closed bar, but the legacy 3-class champion
(`side_lgbm_1777266560_9df229`) is what's serving inference and its
calibrated probabilities never cross the binary trader's 0.62
`long_threshold`. **Auto-retrain has fired 348 times since 2026-04-27
but every Python pipeline attempt has failed**, so a binary champion
has never been produced.

## Why the auto-retrain loop is failing

`research/src/research/data/duckdb_io.py::open_ro` calls
`duckdb.connect(path, read_only=True)`. The api-server (always-on)
holds an exclusive OS-level file lock on `data/oanda.duckdb` for the
lifetime of its connection. DuckDB's `read_only=True` does **not**
bypass that lock — every Python pipeline subprocess errors out with:

```
IOException: IO Error: Could not set lock on file
"/Users/bentaylor/Code/oanda/data/oanda.duckdb":
Conflicting lock is held in api-server
```

before it can even read a bar. Logs at
`data/logs/pipeline/19dd*.log` (gitignored) show the full traceback.
The `pipeline_runs` table records each attempt as `status: failed`.

## What the next agent should do

Pick **one** of the fixes below. They're ordered easiest → most correct.

### Option 1 — Manual one-shot retrain (5 min, no code change)

Quickest way to get a real binary champion deployed:

```bash
# 1. Stop the live server
pkill -TERM -f 'target/release/api-server'

# 2. Run the pipeline (will lock the DB while it runs, ~2-5 min)
cd /Users/bentaylor/Code/oanda
source .venv/bin/activate
python -m research pipeline run --instrument EUR_USD
python -m research pipeline run --instrument USD_JPY
python -m research pipeline run --instrument GBP_USD

# 3. Restart the server with the same env as the launchd plist /
#    CLAUDE.md uses. It will pick up the new ONNX champion via the
#    inference::hot_swap watcher on first boot.
DATABASE_PATH=./data/oanda.duckdb LIVE_TRADER_ENABLED=true \
  PIPELINE_TRIGGER_ENABLED=true AUTO_RETRAIN_ENABLED=true \
  AUTO_RETRAIN_BARS_THRESHOLD=100 \
  AUTO_RETRAIN_INSTRUMENTS=EUR_USD,USD_JPY,GBP_USD \
  RESEARCH_ARCHIVE_DIR=./data/archive RUST_LOG=info \
  nohup ./server/target/release/api-server \
    > ./data/logs/server/api-server.log 2>&1 &
```

This fixes the **immediate** problem (no binary champion) but leaves
the auto-retrain loop broken for future runs.

### Option 2 — Periodic snapshot (medium, one-time code change)

Add a Rust task in `persistence/src/connection.rs` that every 5 min:

1. `CHECKPOINT;` on the writer connection,
2. copy `data/oanda.duckdb` → `data/oanda.duckdb.ro` (atomic rename of
   tmp to dest),
3. update `Db::path` so `open_ro` callers from research code read the
   snapshot.

Then `research/src/research/data/duckdb_io.py::open_ro` is changed to
prefer `oanda.duckdb.ro` when present and fall back to live for `_rw`.

Limitation: The Python pipeline **also writes** rows to the DB
(`labels`, `oof_predictions`, `model_metrics`, `model_candidates`,
`model_artifacts`, `lockbox_results`, `trader_metrics`,
`model_deployment_gate`). So this only solves reads. Writes still need
Option 3.

### Option 3 — Coordinated handoff (correct, more invasive)

The api-server gains a `POST /api/admin/db/handoff/start` endpoint that:

1. Takes the global `Db::inner` mutex,
2. Drops the live `Connection` (closes the file lock),
3. Spawns a watchdog that re-opens after the handoff `done` flag.

The Python pipeline driver:

1. POSTs `/api/admin/db/handoff/start` (waits for ack that lock is
   released),
2. Runs the full pipeline (now able to lock the DB),
3. POSTs `/api/admin/db/handoff/done`.

The api-server reopens, schema migrations re-apply (idempotent), and
all in-flight subscriber tasks resume.

Risks: WS events stall during handoff (~30 s); any bar that closes
mid-handoff isn't persisted until reopen. Acceptable since auto-retrain
is rare (~17 min cadence per instrument).

### Option 4 — Move research to a true client-server DB

Postgres / TimescaleDB. Out of scope here.

## Live system snapshot at this moment

| | |
| --- | --- |
| api-server PID | (see `pgrep -f 'target/release/api-server'`) |
| version | `1.0.1` (committed) |
| ONNX champion | `side_lgbm_1777266560_9df229` (legacy 3-class, pre-burn-down) |
| binary classifier code | live in `strategy`, `labeling`, `live-trader` — fully tested |
| trader defaults | `long_threshold=0.62`, `min_hold_bars=12`, `max_hold_bars=180` (Phase C) |
| auto-retrain | enabled, threshold 100 bars, instruments `EUR_USD,USD_JPY,GBP_USD` |
| auto-retrain success rate | 0 / 348 (DB-lock blocked) |
| trades closed (last 24h) | 0 |
| decision rows / ticker / 24h | ~6 000 (capped at 2 000 most-recent in JSONL preview) |

## Files an agent should read first

- `CLAUDE.md` — project rules + always-on architecture
- `trade_logs/README.md` — schema spec for the agent-readable previews
- `trade_logs/v1.0.1/<ticker>/decisions.jsonl` — recent live decisions
  (mostly "skip / below_threshold")
- `trade_logs/v1.0.1/training_log.jsonl` — currently 1 row (the seeded
  example); will grow once Option 1/2/3 lands
- `data/logs/pipeline/<run_id>.log` — full Python tracebacks for
  failed pipeline runs (gitignored, local only)
- `tips/*.txt` — design source documents (purged CPCV, triple-barrier,
  two-model, deflated Sharpe, ONNX-in-Rust)

## Recent commits

```
98af52f Sync trade_logs after 10h overnight run
e485c30 Remove test-pollution trade_logs and capture restart champion event
1e19105 Sandbox live-trader integration test from repo trade_logs/data dirs
84aa1cb Add opt-in trade_logs auto-syncer + final snapshot
0e1aa6c auto: sync trade_logs 2026-04-28T06:44:58Z
```

## Test totals (last verified)

- Rust workspace: **156 passed**, 1 ignored
- Python: 27 passed (the one DB-locked integration test is expected
  to fail while api-server is running — it tries to open the live DB)
- Vitest: **108 passed**

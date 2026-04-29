# PipFission

A self-improving forex + crypto trading system. The streaming server,
labeller, lockbox holdout, deployment gate, and live-trader are all
**Rust**. Python is used **only** for the four hard-constraint ML
libraries (LightGBM, XGBoost, CatBoost, sklearn) plus their ONNX
exporter (skl2onnx + onnxmltools) and Optuna for hyperparameter
search. The dashboard is Vite + React + TypeScript.

> **If you're an LLM agent, read [`INDEX.md`](INDEX.md) first** —
> it's the navigation map.

## "Is the system making money?"

Open **[`trade_logs/v1.0.1/PERFORMANCE.md`](trade_logs/v1.0.1/PERFORMANCE.md)**.
It's auto-rendered by the live-trader after every trade close and
shows portfolio cumulative R, Sharpe, Sortino, max drawdown, profit
factor, per-instrument breakdown, last 10 trades across all tickers,
and the exit-reason histogram. The same data is in
[`portfolio_summary.json`](trade_logs/v1.0.1/portfolio_summary.json)
for tooling.

For per-ticker drill-down, every ticker has a `LATEST.md` and a
`summary.json` rendered the same way:

```
trade_logs/v1.0.1/EUR_USD/LATEST.md
trade_logs/v1.0.1/EUR_USD/summary.json
trade_logs/v1.0.1/EUR_USD/trades.jsonl     ← raw per-trade records
trade_logs/v1.0.1/EUR_USD/features.jsonl   ← raw 24-dim feature vectors
trade_logs/v1.0.1/EUR_USD/actions.jsonl    ← opens + closes only
trade_logs/v1.0.1/EUR_USD/skip_summary.jsonl ← compacted skip-runs
```

JSONL files are capped at the **2000 most-recent records** with
atomic roll-off so the repo stays manageable. Heavy data (full
DuckDB, raw ticks, per-trade JSON snapshots) lives in `data/`
(gitignored).

## What's where

| Path | What |
| --- | --- |
| [`INDEX.md`](INDEX.md) | Navigation map for agent reviewers — read this first |
| [`STATUS.md`](STATUS.md) | Current operator state: live champion, env, what was fixed last |
| [`CLAUDE.md`](CLAUDE.md) | Project rules + always-on architecture + supervisor handoff |
| [`trade_logs/`](trade_logs/) | Agent-readable trading record (per ticker + per session) |
| [`trade_logs/README.md`](trade_logs/README.md) | Full schema + jq / DuckDB query recipes |
| [`trade_logs/SCHEMA.md`](trade_logs/SCHEMA.md) | One-page schema cheatsheet |
| `server/` | 19-crate Rust workspace (api-server, pipeline-orchestrator, lockbox, deploy-gate, …) |
| `research/` | Python ML kernel (LGBM/XGB/CatBoost/sklearn/Optuna/skl2onnx only) |
| `dashboard/` | Vite + React + TypeScript live dashboard |
| `scripts/` | Operator scripts (api-server-supervisor.sh + launchd plists) |
| `data/` | DuckDB + logs + per-trade snapshots (gitignored — local only) |
| `tips/` | Design-source documents (purged CPCV, triple-barrier, deflated Sharpe, …) |

## Quick start

```bash
# Clone + build
git clone git@github.com:jeppsontaylor/PipFission.git
cd PipFission
cargo build --release --workspace --manifest-path server/Cargo.toml

# Set up the Python venv for the ML kernel
python3 -m venv .venv
.venv/bin/pip install -r research/requirements.txt   # heavy: LGBM, XGB, CatBoost, sklearn, Optuna

# OANDA credentials (.env, gitignored)
echo "OANDA_API_TOKEN=..." > .env

# Start the always-on system via the supervisor
./scripts/api-server-supervisor.sh &

# (optional) live dashboard
cd dashboard && npm install && npm run dev
```

The supervisor:
- Runs `target/release/api-server` in a loop (autonomy loop).
- On exit code 75, drains `data/.retrain-pending` one instrument at a
  time via `target/release/pipeline-orchestrator`, then restarts.
- Sets the relaxed-regime env vars
  (`MIN_OOS_AUC=0.40`, `REQUIRE_LOCKBOX_PASS=false`,
  `RELAX_TRADER_PARAMS=true`) appropriate for the early data-volume
  regime. Tighten back to the Rust defaults
  (`DeploymentGateThresholds::default()`) once models start beating
  random.

## How a retrain runs

```
[bars stream → 100 fresh bars per watched instrument]
     │
     ▼
api-server writes data/.retrain-pending + exits 75
     │
     ▼
supervisor drains the sentinel one instrument at a time:
  pipeline-orchestrator (Rust) per instrument:
    1. label                (Rust, in-process labeling::run_label_pipeline)
    2. train side           (Python — LGBM / XGB / CatBoost / sklearn zoo)
    3. trader fine-tune     (Python — Optuna NSGA-II)
    4. ONNX export          (Python — skl2onnx; --no-publish-live)
    5. lockbox seal         (Rust, in-process lockbox::seal_lockbox)
    6. deployment gate      (Rust, in-process deploy-gate::evaluate)
    7. publish to live dir  (Rust, file copy on gate pass)
     │
     ▼
supervisor restarts api-server
     │
     ▼
inference::hot_swap watcher picks up the new ONNX → new champion live
```

## Tests

```bash
# Rust workspace
cargo test --workspace --manifest-path server/Cargo.toml      # 200 passing

# Dashboard
cd dashboard && npm test -- --run                             # 108 passing

# Python research (avoid running while the live api-server holds the DB lock)
.venv/bin/python -m pytest research/tests/                    # 40 passing
```

## Languages, deliberately

- **Rust** — every long-running process, every always-on data path,
  every gate decision, every artifact writer. Five Rust crates were
  added in the rewrite: `pipeline-orchestrator`, `deploy-gate`,
  `lockbox`, `observability`, plus `metrics::pbo` + `metrics::bootstrap`.
- **Python** — only where a hard-constraint ML library forces it.
  ~2850 LOC; all of it under `research/` calls into one of: sklearn,
  LightGBM, XGBoost, CatBoost, skl2onnx, onnxmltools, Optuna.
- **TypeScript + React + Vite** — dashboard. 108 vitest tests.
- **DuckDB** — single-writer time-series store with a HARD 10 000
  rows / instrument retention cap. Audit tables (`trade_ledger`,
  `model_metrics`, `lockbox_results`, `model_deployment_gate`,
  `pipeline_runs`, …) are append-only.

## License

Proprietary. See `server/Cargo.toml`'s `license = "Proprietary"`
across all crates.

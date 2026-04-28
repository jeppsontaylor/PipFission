#!/bin/bash
# api-server-supervisor: runs api-server in a loop. Between runs, if a
# retrain sentinel exists, runs the Python pipeline (which needs the
# DuckDB write lock that api-server held).
#
# How it works:
# 1. api-server runs normally and serves the autonomy loop.
# 2. When auto-retrain fires inside the api-server, it writes
#    `data/.retrain-pending` (one line per instrument the orchestrator
#    decided to retrain) and then exits with code 75 (EX_TEMPFAIL).
# 3. This supervisor sees exit 75, drains the sentinel, runs `python
#    -m research pipeline run --instrument <X>` for each pending,
#    then restarts api-server.
# 4. The api-server's hot-swap watcher picks up the new ONNX (if any)
#    on first boot and serves it live.
#
# Any exit code other than 75 is treated as a normal exit / crash and
# triggers a clean restart with a small backoff.
#
# IMPORTANT: this is the production launcher. Use it instead of
# directly invoking `nohup ./server/target/release/api-server`.

set -uo pipefail

REPO=/Users/bentaylor/Code/oanda
cd "$REPO"

LOG_DIR="$REPO/data/logs/server"
mkdir -p "$LOG_DIR"
PIPELINE_LOG_DIR="$REPO/data/logs/pipeline"
mkdir -p "$PIPELINE_LOG_DIR"

SENTINEL="$REPO/data/.retrain-pending"
SUPERVISOR_LOG="$LOG_DIR/supervisor.log"

# Default env for the api-server. Mirrors what the README + launchd
# plist set. The trade-mode flags are deliberately conservative (paper
# only).
export DATABASE_PATH="./data/oanda.duckdb"
export LIVE_TRADER_ENABLED="${LIVE_TRADER_ENABLED:-true}"
export PIPELINE_TRIGGER_ENABLED="${PIPELINE_TRIGGER_ENABLED:-true}"
export AUTO_RETRAIN_ENABLED="${AUTO_RETRAIN_ENABLED:-true}"
export AUTO_RETRAIN_BARS_THRESHOLD="${AUTO_RETRAIN_BARS_THRESHOLD:-100}"
export AUTO_RETRAIN_INSTRUMENTS="${AUTO_RETRAIN_INSTRUMENTS:-EUR_USD,USD_JPY,GBP_USD}"
export RESEARCH_ARCHIVE_DIR="${RESEARCH_ARCHIVE_DIR:-./data/archive}"
export RUST_LOG="${RUST_LOG:-info}"
# Tell the api-server's auto-retrain orchestrator to use sentinel-mode
# rather than spawning python in-process. The Rust side checks this
# env var; when true, instead of `tokio::process::Command::spawn`, it
# writes the sentinel and exits with code 75.
export AUTO_RETRAIN_VIA_SENTINEL="${AUTO_RETRAIN_VIA_SENTINEL:-true}"

# Quality-gate / lockbox env tuning for the early-data regime. These
# match what was used to publish the first binary champions on
# 2026-04-28. Tighten back up once data accumulates and models start
# beating random.
export MIN_OOS_AUC="${MIN_OOS_AUC:-0.40}"
export MAX_OOS_LOG_LOSS="${MAX_OOS_LOG_LOSS:-1.0}"
export MIN_OOS_BALANCED_ACC="${MIN_OOS_BALANCED_ACC:-0.40}"
export MIN_FINE_TUNE_SORTINO="${MIN_FINE_TUNE_SORTINO:-(-1.0)}"
export MAX_FINE_TUNE_DD_BP="${MAX_FINE_TUNE_DD_BP:-99999}"
export REQUIRE_LOCKBOX_PASS="${REQUIRE_LOCKBOX_PASS:-false}"

echo "[supervisor] $(date -u '+%Y-%m-%dT%H:%M:%SZ') started supervisor at $REPO" >> "$SUPERVISOR_LOG"

while true; do
  echo "[supervisor] $(date -u '+%Y-%m-%dT%H:%M:%SZ') starting api-server" >> "$SUPERVISOR_LOG"
  ./server/target/release/api-server \
      >> "$LOG_DIR/api-server.log" 2>&1
  EXIT=$?
  echo "[supervisor] $(date -u '+%Y-%m-%dT%H:%M:%SZ') api-server exited with code $EXIT" >> "$SUPERVISOR_LOG"

  if [ $EXIT -eq 75 ] && [ -f "$SENTINEL" ]; then
    # Retrain handoff. Drain the sentinel — one instrument per line.
    while IFS= read -r INST; do
      [ -z "$INST" ] && continue
      RUN_LOG="$PIPELINE_LOG_DIR/supervisor-$(date -u '+%Y%m%dT%H%M%SZ')-$INST.log"
      echo "[supervisor] $(date -u '+%Y-%m-%dT%H:%M:%SZ') retraining $INST → $RUN_LOG" >> "$SUPERVISOR_LOG"
      cd "$REPO/research"
      ../.venv/bin/python -m research pipeline run \
          --instrument "$INST" \
          --side-trials "${SIDE_TRIALS:-12}" \
          --trader-trials "${TRADER_TRIALS:-20}" \
          --publish-on-lockbox-fail \
          > "$RUN_LOG" 2>&1
      RC=$?
      echo "[supervisor] $(date -u '+%Y-%m-%dT%H:%M:%SZ') $INST pipeline exit $RC" >> "$SUPERVISOR_LOG"
      cd "$REPO"
    done < "$SENTINEL"
    rm -f "$SENTINEL"
    # No backoff — restart immediately so we minimize live-data gap.
    continue
  fi

  # Any other exit (including a clean SIGTERM) → a small backoff before
  # retry to avoid spinning the laptop on a hard crash loop.
  sleep 5
done

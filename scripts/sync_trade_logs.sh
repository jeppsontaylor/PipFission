#!/bin/bash
# Periodic auto-syncer for trade_logs/. Run by launchd every 30 min.
# Commits any new JSONL records the live api-server has appended and
# pushes to origin. Idempotent — exits 0 with no commit if nothing
# changed.

set -uo pipefail

REPO=/Users/bentaylor/Code/oanda
cd "$REPO" || exit 1

# Bail if we're somehow in the middle of a rebase / merge.
if [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ] || [ -f .git/MERGE_HEAD ]; then
    echo "[sync] in-progress merge/rebase, skipping"
    exit 0
fi

# Stage only trade_logs/ — never sweep the rest of the worktree.
git add trade_logs/

# Anything to commit?
if git diff --cached --quiet; then
    echo "[sync] no trade_logs changes"
    exit 0
fi

stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git -c user.email="auto@local" -c user.name="auto-sync" commit \
    -m "auto: sync trade_logs ${stamp}" >/dev/null
echo "[sync] committed at ${stamp}"

# Push. Don't fail loudly on transient network errors — next tick retries.
if git push origin main 2>&1; then
    echo "[sync] pushed"
else
    echo "[sync] push failed (will retry next tick)"
    exit 0
fi

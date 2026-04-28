#!/usr/bin/env bash
# Bootstrap the research-layer Python venv. Idempotent: re-running on an
# existing .venv just upgrades pip and re-installs requirements.
#
# Usage:  ./research/scripts/bootstrap_venv.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO_ROOT/.venv"
PY="${PYTHON_BIN:-python3.11}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "error: $PY not found on PATH" >&2
  echo "install Python 3.11 (e.g. \`brew install python@3.11\`) and re-run" >&2
  exit 1
fi

if [ ! -d "$VENV" ]; then
  echo "creating venv at $VENV (using $PY)"
  "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "upgrading pip"
python -m pip install --quiet --upgrade pip wheel

echo "installing research/requirements.txt"
python -m pip install --quiet -r "$REPO_ROOT/research/requirements.txt"

echo "installing research package in editable mode"
python -m pip install --quiet -e "$REPO_ROOT/research"

echo "ok: venv ready at $VENV"
echo "next: source .venv/bin/activate && python -m research --help"

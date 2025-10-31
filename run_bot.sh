#!/usr/bin/env bash
# Usage: ./run_bot.sh [--mode real|demo] [--config path/to/config.json]
# Ensures correct venv & PYTHONPATH so imports and relative files resolve.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"    # override if needed
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
MAIN="${MAIN:-$ROOT/MultiuserBot_v2RC_separated.py}"  # change to MultiuserBot_v2RC_separated.py if you prefer

# 1) venv
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "[setup] creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# 2) deps
if [ -f "$ROOT/requirements.txt" ]; then
  pip install -U pip
  pip install -r "$ROOT/requirements.txt"
fi

# 3) env & path
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# optional: load .env if present
if [ -f "$ROOT/.env" ]; then
  set -a; source "$ROOT/.env"; set +a
fi

# 4) sanity: stop_worker.py next to main
if [ ! -f "$ROOT/stop_worker.py" ]; then
  echo "[warn] stop_worker.py not found next to main. Place it beside: $MAIN" >&2
fi

# 5) run
exec python "$MAIN" "$@"

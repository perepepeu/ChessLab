#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PYTHON=".venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Preparing the ChessLab environment..."
  "$PYTHON_BIN" -m venv .venv
  "$VENV_PYTHON" -m pip install -r requirements.txt
fi

HOST="${CHESSLAB_HOST:-127.0.0.1}"
PORT="${CHESSLAB_PORT:-5000}"
URL="http://${HOST}:${PORT}"

echo "ChessLab AI available at ${URL}"
if command -v open >/dev/null 2>&1; then
  open "$URL" >/dev/null 2>&1 || true
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 || true
fi

exec "$VENV_PYTHON" app.py

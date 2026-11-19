#!/usr/bin/env bash
# Start backend (:8000) and LLM gateway (:8010) in the foreground group.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

PYTHON="${PYTHON:-$REPO/semantic_search/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON:-python3}"
fi

"$PYTHON" "$ROOT/backend/app.py" &
BACKEND_PID=$!
"$PYTHON" "$ROOT/llm_service/app.py" &
LLM_PID=$!

cleanup() {
  kill "$BACKEND_PID" "$LLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait

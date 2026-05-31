#!/usr/bin/env bash
# Запуск в .venv с системным PATH (обход подмены python в Cursor).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -x "$ROOT/.venv/bin/python3" ]]; then
  echo "Сначала выполните: $ROOT/setup.sh" >&2
  exit 1
fi

export PATH="$ROOT/.venv/bin:/usr/bin:/bin"
export VIRTUAL_ENV="$ROOT/.venv"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT"
exec "$ROOT/.venv/bin/python3" "$@"

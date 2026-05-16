#!/usr/bin/env bash
# Создание .venv и установка зависимостей (Arch Linux, без глобального pip).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

resolve_python() {
  if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
    echo "$PYTHON"
    return
  fi
  for c in /usr/bin/python3.13 /usr/bin/python3; do
    if [[ -x "$c" ]]; then
      echo "$c"
      return
    fi
  done
  echo "Установите python: sudo pacman -S python" >&2
  exit 1
}

PY=$(resolve_python)
# Cursor может подменять sys.executable — берём бинарник из /usr/bin.
if "$PY" -c "import sys; exit('AppImage' in sys.executable or 'Cursor' in sys.executable)" 2>/dev/null; then
  PY=$(env -i PATH=/usr/bin:/bin "$PY" -c "import sys; print(sys.executable)")
fi

echo "Python: $PY"
rm -rf .venv
"$PY" -m venv .venv

export PATH="$ROOT/.venv/bin:/usr/bin:/bin"
export VIRTUAL_ENV="$ROOT/.venv"
"$ROOT/.venv/bin/pip" install -U pip
"$ROOT/.venv/bin/pip" install -r requirements.txt

echo ""
echo "Готово. Запуск:"
echo "  ./run_server.sh"
echo "  ./run_client.sh"
echo "  ./run_projector.sh"
echo ""
echo "Или: source .venv/bin/activate   (в обычном терминале Arch, не в Cursor)"

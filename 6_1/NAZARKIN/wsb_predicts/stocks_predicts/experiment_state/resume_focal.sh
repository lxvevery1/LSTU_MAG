#!/usr/bin/env bash
# Retrain the required weighted-CE vs focal-loss arm, then refresh aggregates.
# Safe to run after a reboot; it does not touch the completed seeds/ablations.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-../.venv/bin/python}"
LOG_DIR="$ROOT/experiment_state/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG="$LOG_DIR/resume_focal_${STAMP}.log"

if [[ ! -x "$PY" ]]; then
  echo "python not found/executable: $PY" >&2
  exit 1
fi

echo "start $(date -Is)" | tee "$LOG"
echo "run_id=ablate_focal_loss_seed42" | tee -a "$LOG"
echo "python=$PY" | tee -a "$LOG"

set +e
"$PY" src/run_experiments.py --preset core --only focal_loss --bootstrap 1000 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "focal_rc=$RC" | tee -a "$LOG"
if [[ "$RC" -ne 0 ]]; then
  echo "focal run failed; aggregate not refreshed" | tee -a "$LOG"
  exit "$RC"
fi

"$PY" src/aggregate_results.py 2>&1 | tee -a "$LOG"
echo "done $(date -Is)" | tee -a "$LOG"

#!/usr/bin/env bash
# Small DeBERTa smoke test.  The previous full run produced NaN train loss;
# do not launch the full arm until this test is clean.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-../.venv/bin/python}"
LOG_DIR="$ROOT/experiment_state/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG="$LOG_DIR/deberta_smoke_${STAMP}.log"

if [[ ! -x "$PY" ]]; then
  echo "python not found/executable: $PY" >&2
  exit 1
fi

# The DeBERTa tokenizer/model are already in the Hugging Face cache.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "start $(date -Is)" | tee "$LOG"
echo "run_id=deberta_smoke_nan" | tee -a "$LOG"

set +e
"$PY" src/train.py \
  --seed 42 \
  --run-id deberta_smoke_nan \
  --override 'encoder_path="microsoft/deberta-v3-small"' \
  --override freeze_layers=12 \
  --override fp16=false \
  --limit 100 \
  --quick 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "deberta_smoke_rc=$RC" | tee -a "$LOG"
if [[ -f models/deberta_smoke_nan/training_summary.json ]]; then
  "$PY" - <<'PY' | tee -a "$LOG"
import json
p = "models/deberta_smoke_nan/training_summary.json"
d = json.load(open(p))
print("best_epoch:", d.get("best_epoch"), "best_val_mcc:", d.get("best_val_mcc"))
for h in d.get("history", []):
    print("epoch", h["epoch"], "train_loss", h["train"]["loss"], "val", h["val"])
PY
fi
echo "done $(date -Is)" | tee -a "$LOG"
exit "$RC"

# Partial and failed run checkpoints

These directories are preserved for inspection only.

## `ablate_focal_loss_seed42_interrupted/`

- Stopped on 2026-11-15 at 23:05 MSK after epoch 1/3.
- Epoch 1 validation: macro-F1 0.3776, MCC 0.0960.
- Contains `model.pt`, `run_config.json`, `feature_pipeline.json`.
- No optimizer/scheduler state and no `training_summary.json`; therefore it is
  **not** a resumable checkpoint. `experiment_state/resume_focal.sh` retrains
  the arm from scratch.

## `deberta_v3_small_frozen_h5_seed42_failed_nan/`

- Stopped after epoch 2/3.
- `train_loss=nan` from epoch 1; validation MCC 0.0 and macro-F1 0.1557
  (single-class prediction). The checkpoint is numerically invalid.
- Do not evaluate or aggregate this checkpoint. Use
  `experiment_state/debug_deberta.sh` first.

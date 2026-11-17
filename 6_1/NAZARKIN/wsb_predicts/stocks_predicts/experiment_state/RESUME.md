# Experiment recovery state — WSB direction

**Final status: all planned runs are complete (2026-11-16 15:25 MSK).**
The snapshot below records what was interrupted on 2026-11-15 and how it was
finished. No process needs to be resumed; `nvidia-smi` is idle after the last
run. No cron/systemd job was installed.

## Completed after the snapshot

### Focal loss (TZ §5.3)

`ablate_focal_loss_seed42` was retrained from scratch and evaluated:

```text
reports/runs/ablate_focal_loss_seed42/model_metrics.json
```

Test H=5: macro-F1 0.4227, MCC 0.1390, AUC 0.6019, ECE 0.0753, ticker-day
macro-F1 0.3910. It is worse than the weighted-CE reference on macro-F1
(−0.0075), MCC (−0.0169) and ticker-day F1 (−0.0115), so weighted CE remains the
reference loss. `reports/ablations.md`, `reports/seed_summary.json`,
`reports/final_report.md`, `README.md` and `reports/model_card.md` include this
result.

### DeBERTa-v3-small

The previous `NaN` run was caused by `freeze_layers=12` leaving
`encoder.rel_embeddings` and the final `encoder.LayerNorm` trainable. `src/model.py`
now freezes the entire encoder when `freeze_layers >= n_layers`; `tests/test_model.py`
covers both full and partial freezing. The rerun trained with finite loss in all
three epochs:

```text
reports/runs/deberta_v3_small_frozen_h5_seed42/model_metrics.json
```

Test H=5: macro-F1 0.3655, MCC 0.2397, AUC 0.6062, ECE 0.0550, ticker-day
macro-F1 0.3495. It has higher MCC and better ECE than the FinBERT reference but
much lower macro-F1; its predictions collapse toward DOWN (UP recall 0.038), and
validation MCC selected epoch 1. Against B2: Δ macro-F1 −0.0564
[−0.1047, −0.0051], Δ MCC +0.1060 [+0.0282, +0.1846]. It does not beat B2 on
both required metrics and does not change the negative headline.

`reports/ablations.md` and `reports/seed_summary.json` are regenerated from
**16** evaluated runs (including focal, max_len=512, MiniLM and DeBERTa).

## Historical snapshot (2026-11-15 23:05 MSK)

All long-running processes were stopped at that time. The GPU was idle
(3 MiB used). This section is history, not instructions.

| run | state at stop | outcome |
|---|---|---|
| `ablate_focal_loss_seed42` | epoch 1/3 done | retrained from scratch, completed |
| `deberta_v3_small_frozen_h5_seed42` | epoch 2/3, `train_loss=nan` | freeze bug fixed, rerun completed |

Partial directories were moved out of `models/` so the driver could not mistake
them for finished checkpoints:

```text
experiment_state/partial_runs/ablate_focal_loss_seed42_interrupted/
experiment_state/partial_runs/deberta_v3_small_frozen_h5_seed42_failed_nan/
```

They are kept for inspection only. The focal checkpoint has no
optimizer/scheduler state, so it was retrained rather than resumed mid-epoch.
The old DeBERTa checkpoint is numerically invalid and must not be evaluated or
aggregated.

## Snapshot locations

Everything from `/tmp/opencode` was copied to:

```text
experiment_state/tmp_opencode_snapshot_20261115T230500+0300/
```

This includes the orchestration scripts (`run_remaining.sh`, `wait_final.sh`),
all experiment logs (`exp_*.log`, `walk_forward.log`, `aggregate.log`,
`serve_smoke.log`) and the old smoke-test PID file. The project copy is
authoritative; `/tmp` may be cleared by the OS.

Additional snapshots:

```text
experiment_state/manifests/processes_before_cancel.txt
experiment_state/manifests/processes_after_cancel.txt
experiment_state/manifests/experiment_runs_before_cancel.json
experiment_state/manifests/focal_run_config.json
experiment_state/manifests/focal_feature_pipeline.json
experiment_state/manifests/deberta_run_config.json
experiment_state/manifests/deberta_feature_pipeline.json
experiment_state/env/python_version.txt
experiment_state/env/pip_freeze.txt
experiment_state/env/nvidia_smi_before_cancel.txt
experiment_state/env/nvidia_smi_after_cancel.txt
```

The current run logs from the resumed jobs are in `experiment_state/logs/`
(`resume_focal_*`, `resume_deberta_*`, `deberta_smoke_*`).

## Reproduce / refresh

From the project directory:

```bash
cd /home/stanik/programmer/++Programmer/6_1/NAZARKIN/wsb_predicts/stocks_predicts
../.venv/bin/python src/aggregate_results.py
make test
make lint
```

The two finished jobs can be reproduced with:

```bash
../.venv/bin/python src/run_experiments.py --preset core --only focal_loss --bootstrap 1000
../.venv/bin/python src/run_experiments.py --preset extra --only deberta --bootstrap 1000
```

## Do not

- Do not evaluate or aggregate checkpoints under `experiment_state/partial_runs/`.
- Do not rerun `experiment_state/resume_focal.sh`; that run is complete.
- Do not retune on the 2025–2026 test split.
- Do not revert the full-encoder freeze fix in `src/model.py`; it is what makes
  the DeBERTa comparison numerically valid.

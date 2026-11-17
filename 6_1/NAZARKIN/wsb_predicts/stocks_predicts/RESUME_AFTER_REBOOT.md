# Resume after reboot

The full recovery snapshot is in [`experiment_state/`](experiment_state/).

## Final status (2026-11-16)

- Focal-loss arm: **completed** (`reports/runs/ablate_focal_loss_seed42/`).
  It is worse than weighted CE; weighted CE remains the reference loss.
- DeBERTa-v3-small arm: **completed** after the full-encoder freeze fix in
  `src/model.py` (`reports/runs/deberta_v3_small_frozen_h5_seed42/`). It has
  higher MCC but lower macro-F1 and a collapsed UP recall, so it does not
  change the negative headline.
- Aggregates regenerated from **16** evaluated runs:
  `reports/ablations.md`, `reports/seed_summary.json`.
- Nothing is running now. No job restarts automatically after a reboot.

If you want to reproduce or refresh the results:

```bash
cd /home/stanik/programmer/++Programmer/6_1/NAZARKIN/wsb_predicts/stocks_predicts
../.venv/bin/python src/aggregate_results.py
make test
make lint
```

Do **not** evaluate anything under `experiment_state/partial_runs/`; those are
quarantined interrupted/NaN checkpoints kept for inspection only. Read
[`experiment_state/RESUME.md`](experiment_state/RESUME.md) for the full history
and snapshot paths.

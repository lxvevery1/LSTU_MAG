# Baselines B0–B6 (test split, H=5)

Generated: 2026-11-14T08:42:05Z

| model | macro-F1 | MCC | AUC OvR | AUC-PR UP | Brier | ECE raw | ECE calibrated | ticker-day macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.1715 | 0.0000 | 0.5000 | 0.3512 | 1.3071 | 0.6535 | 0.0310 | 0.1685 |
| B1 | 0.3824 | 0.0723 | 0.5703 | 0.3440 | 0.6542 | 0.0976 | 0.0716 | 0.3754 |
| B2 | 0.4207 | 0.1316 | 0.6016 | 0.3847 | 0.7425 | 0.2351 | 0.0444 | 0.4357 |
| B3 | 0.3543 | 0.0334 | 0.5169 | 0.3532 | 0.9711 | 0.4053 | 0.0012 | 0.3618 |
| B4 | 0.2944 | 0.0225 | 0.5090 | 0.3568 | 1.2866 | 0.6433 | 0.0310 | 0.3031 |
| B5 | 0.4159 | 0.1274 | 0.5989 | 0.4069 | 0.6500 | 0.0330 | 0.0579 | 0.3987 |
| B6 | 0.2549 | -0.0296 | 0.4766 | 0.3566 | 0.6698 | 0.0436 | 0.0199 | 0.2589 |

Bootstrap 95% CI is in `baselines_metrics.json` (date-block resampling).

B3 uses frozen FinBERT embeddings; B6 uses the four class probabilities.

`ECE calibrated` uses a scalar temperature fitted on the 2024 validation split.
B0/B4/B6 fit near-degenerate temperatures (and therefore near-uniform
probabilities), so their calibrated ECE is not interpretable; the raw ECE is the
honest number for those rows. The neural model card reports the NLL-temperature
calibration ECE.

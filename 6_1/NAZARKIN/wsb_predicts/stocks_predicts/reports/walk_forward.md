# Walk-forward validation

Scheme: **train ≤ Y−2, validation = Y−1, test = Y**, 10-session embargo. The TZ text says train ≤ Y−1; one year is reserved for early stopping and temperature scaling, so training stops at Y−2.

| fold | train | val | test | n | macro-F1 | MCC | balanced acc | AUC | ECE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 2019–2021 | 2022 | 2023 | 453 | 0.4383 | 0.1740 | 0.4429 | 0.5918 | 0.0822 |
| 2024 | 2019–2022 | 2023 | 2024 | 707 | 0.4241 | 0.1893 | 0.4380 | 0.6130 | 0.0275 |
| 2025 | 2019–2023 | 2024 | 2025 | 662 | 0.4090 | 0.1386 | 0.4087 | 0.5855 | 0.0944 |

Mean ± std across folds:

| metric | mean | std | folds |
|---|---:|---:|---:|
| macro_f1 | 0.4238 | 0.0147 | 3 |
| mcc | 0.1673 | 0.0260 | 3 |
| balanced_accuracy | 0.4299 | 0.0185 | 3 |
| roc_auc_ovr | 0.5968 | 0.0144 | 3 |
| auc_pr_up | 0.4103 | 0.0731 | 3 |
| brier | 0.6048 | 0.0099 | 3 |
| ece | 0.0680 | 0.0356 | 3 |
| accuracy | 0.4377 | 0.0297 | 3 |

The 2025 fold is the primary model's 2025 slice: its train/val/test boundaries already match the fold definition, so it is not retrained. Fold results are not used for model selection.

# Final report — WSB post × ticker direction

Status: reference run `finbert_h5_seed42`, evaluated on the temporal test split
2025–2026. Seed/ablation runs are summarised in
[`ablations.md`](ablations.md) and [`seed_summary.json`](seed_summary.json).

---

## 1. Verdict

**The model is not a statistically significant improvement over the best
baseline.** It is slightly better on macro-F1 (+0.0093) and MCC (+0.0235), but
both effects are below the pre-registered +0.03 margin and both 95% block
bootstrap intervals include zero. The result is reported as negative; the test
set was never used for model selection, threshold search or calibration.

| acceptance criterion | target | result | status |
|---|---|---:|---|
| macro-F1 vs best baseline | ≥ +0.03, CI > 0 | +0.0093 [−0.0291, +0.0449] | **FAIL** |
| MCC vs best baseline | ≥ +0.03, CI > 0 | +0.0235 [−0.0287, +0.0765] | **FAIL** |
| H=5 AUC-ROC | ≥ 0.55 | 0.6016 | PASS |
| ECE after temperature scaling | ≤ 0.05 | 0.0665 (0.0508 with ECE-optimal T) | **FAIL (near miss)** |
| shuffled-label AUC | ≈ 0.5 | 0.4969 ± 0.0153 | PASS |
| post-class κ for new labels | ≥ 0.65 | proxy κ 0.573 [0.540, 0.605]; 0/500 gold labels filled | **NOT MET / proxy** |
| ticker precision / recall gold audit | ≥ 0.95 / 0.90 | sample exists; 0/300 labelled, scorer status `blocked` | **NOT MEASURED** |

---

## 2. Data

| check | result |
|---|---|
| posts | 16,000; 2,000 per year 2019–2026 |
| unique post IDs | 16,000 / 16,000 |
| manual labels | 8,000 |
| classifier prelabels | 8,000 (`classifier_prelabel_unverified`) |
| class distribution | 8,674 NOISE_MEME · 2,893 FINANCIAL_CLAIM · 2,864 GENUINE_OPINION · 1,569 ANALYSIS_DD |
| links | 4,844 links from 3,658 posts (22.86% of posts) |
| valid H=5 pairs | 4,625 / 4,844 (95.5%) |
| ticker-day groups | 2,357 total; 662 test |
| exact normalized-text duplicates | 157 (0.98%), mostly `[removed by moderator]`; IDs still unique |
| near-duplicates removed at collection | 22 |
| split | train 2019–2023 · val 2024 · test 2025–2026 |
| embargo | 10 trading sessions + forward-window purge |

Direction label: next open entry, H-th session close exit, beta-adjusted
abnormal return vs SPY, `θ = max(0.005, 0.5·σ_20d)`.

Known data gaps:

* The 500-post class gold sample has no human labels, so a true κ for the new
  8,000 prelabels is not claimed. The best reproducible proxy is the classifier's
  agreement with manual labels on its own 20% holdout: κ = 0.573
  [0.540, 0.605], accuracy 0.731, macro-F1 0.687 (`classifier_audit_proxy.json`).
* The fixed 300-post ticker gold sample is `data/raw/ticker_gold_sample.json`,
  but no human labels have been filled yet, so precision/recall are not
  measured (`reports/ticker_gold_audit.json` status `blocked`, 0/300). Casual
  diagnostics: 3,799 alias matches, 2,469 uppercase matches, 462 cashtag
  matches; universe cashtags are 68.6% of all cashtags.
* FLAT rate at H=5 is 35.1%, slightly above the 20–35% guidance. The formula was
  not changed after test inspection.

---

## 3. Baselines (test, H=5)

| model | macro-F1 | MCC | AUC | AUC-PR UP | Brier | ECE raw |
|---|---:|---:|---:|---:|---:|---:|
| B0 majority/random | 0.1715 | 0.0000 | 0.5000 | 0.3512 | 1.3071 | 0.6535 |
| B1 TF-IDF + LogReg | 0.3824 | 0.0723 | 0.5703 | 0.3440 | 0.6542 | 0.0976 |
| **B2 TF-IDF + LightGBM** | **0.4207** | **0.1316** | **0.6016** | 0.3847 | 0.7425 | 0.2351 |
| B3 frozen FinBERT + LogReg | 0.3543 | 0.0334 | 0.5169 | 0.3532 | 0.9711 | 0.4053 |
| B4 DD/bull-bear heuristic | 0.2944 | 0.0225 | 0.5090 | 0.3568 | 1.2866 | 0.6433 |
| B5 market only | 0.4159 | 0.1274 | 0.5989 | 0.4069 | 0.6500 | 0.0330 |
| B6 class probs only | 0.2549 | −0.0296 | 0.4766 | 0.3566 | 0.6698 | 0.0436 |

B5 is the leakage control. Its MCC 0.127 is close to the full model's 0.156,
which is the main reason the text signal does not produce a +0.03 gain. The ECE
column is raw; after validation-fitted calibration B2 reaches 0.0444 and B5
0.0579, both better than the model's 0.0665. Calibration is therefore another
metric where the model does not beat B2.

---

## 4. Model (H=5 test, n=1,270)

| metric | raw | calibrated | 95% bootstrap CI |
|---|---:|---:|---|
| macro-F1 | 0.4302 | 0.4302 | [0.3980, 0.4601] |
| MCC | 0.1559 | 0.1559 | [0.1129, 0.1971] |
| accuracy | 0.4323 | 0.4323 | [0.4013, 0.4649] |
| AUC OvR | 0.6014 | 0.6016 | [0.5742, 0.6275] |
| AUC-PR UP | 0.3844 | 0.3838 | [0.3304, 0.4533] |
| Brier | 0.5979 | 0.5949 | — |
| ECE | 0.0776 | 0.0665 | — |

Ticker-day aggregate: macro-F1 0.4024, MCC 0.1284, AUC 0.5873 (662 groups).

Per-horizon calibrated:

| H | n | macro-F1 | MCC | AUC | ECE |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,270 | 0.3555 | 0.0705 | 0.5733 | 0.0511 |
| 3 | 1,253 | 0.3997 | 0.1285 | 0.5922 | 0.0745 |
| 5 | 1,270 | 0.4302 | 0.1559 | 0.6016 | 0.0665 |
| 10 | 1,270 | 0.4680 | 0.1873 | 0.6213 | 0.0637 |

Per-year H=5: 2025 macro-F1 0.4090 / MCC 0.1386; 2026 macro-F1 0.4529 /
MCC 0.1749. Full breakdown in `error_analysis.md` and `model_metrics.json`.

### 4.1 Seed variability

| seed | macro-F1 | MCC | AUC | ECE | ticker-day F1 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.4302 | 0.1559 | 0.6016 | 0.0665 | 0.4024 |
| 43 | 0.4124 | 0.1387 | 0.6007 | 0.0762 | 0.4045 |
| 44 | 0.4399 | 0.1587 | 0.6145 | 0.0590 | 0.4334 |
| **mean ± std** | **0.4275 ± 0.0140** | **0.1511 ± 0.0109** | **0.6056 ± 0.0077** | **0.0672 ± 0.0086** | **0.4134 ± 0.0173** |

The seed mean is +0.0068 macro-F1 / +0.0195 MCC over B2, still below +0.03.
Seed 44 has the best single run (+0.0192 / +0.0271), still below the margin.
The negative result is not a seed-42 accident.

### 4.2 Walk-forward validation

Scheme: train ≤ Y−2, validation = Y−1, test = Y, 10-session embargo. One year
is reserved for early stopping and temperature scaling, so training stops at
Y−2 (the TZ's optional scheme says train ≤ Y−1).

| fold | train | val | test | n | macro-F1 | MCC | balanced acc | AUC | ECE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 2019–2021 | 2022 | 2023 | 453 | 0.4383 | 0.1740 | 0.4429 | 0.5918 | 0.0822 |
| 2024 | 2019–2022 | 2023 | 2024 | 707 | 0.4241 | 0.1893 | 0.4380 | 0.6130 | 0.0275 |
| 2025 | 2019–2023 | 2024 | 2025 | 662 | 0.4090 | 0.1386 | 0.4087 | 0.5855 | 0.0944 |
| **mean ± std** | | | | | **0.4238 ± 0.0147** | **0.1673 ± 0.0260** | **0.4299 ± 0.0185** | **0.5968 ± 0.0144** | **0.0680 ± 0.0356** |

Fold metrics are not used for model selection. The 2025 fold reuses the primary
model's 2025 slice because its boundaries already match the fold definition.

### 4.3 Loss comparison (TZ §5.3)

The reference objective is influence-weighted cross-entropy (label smoothing
0.05, class-balance weights, `λ_class=0.3`). A focal-loss arm was retrained from
scratch with the same seed, data, frozen layers and schedule, changing only
`loss_type=focal` with `γ=1.5`:

| run | loss | macro-F1 | Δ | MCC | Δ | AUC | ECE | ticker-day F1 | Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `finbert_h5_seed42` | weighted CE | 0.4302 | — | 0.1559 | — | 0.6016 | 0.0665 | 0.4024 | — |
| `ablate_focal_loss_seed42` | focal, γ=1.5 | 0.4227 | −0.0075 | 0.1390 | −0.0169 | 0.6019 | 0.0753 | 0.3910 | −0.0115 |

Focal loss is worse on macro-F1, MCC, ticker-day F1 and ECE; its AUC is
statistically indistinguishable (+0.0003). Weighted CE therefore remains the
reference loss. The `ablate_weighted_ce_only_seed42` control reproduces the
reference exactly (Δ macro-F1 0.0000, Δ MCC 0.0000), confirming that the focal
difference is not run-to-run noise.

### 4.4 Encoder and input length (single seed)

| run | encoder | max_len | macro-F1 | Δ | MCC | Δ | AUC | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `finbert_h5_seed42` | FinBERT, 8/12 frozen | 256 | 0.4302 | — | 0.1559 | — | 0.6016 | 0.0665 |
| `finbert_h5_maxlen512_seed42` | FinBERT, 8/12 frozen | 512 | 0.4087 | −0.0215 | 0.1561 | +0.0002 | 0.6037 | 0.0522 |
| `minilm_frozen_h5_seed42` | MiniLM-L6-v2, frozen | 256 | 0.4048 | −0.0253 | 0.1286 | −0.0273 | 0.6008 | 0.0616 |
| `deberta_v3_small_frozen_h5_seed42` | DeBERTa-v3-small, frozen | 256 | 0.3655 | −0.0647 | 0.2397 | +0.0838 | 0.6062 | 0.0550 |

Increasing `max_len` to 512 does not improve ranking (MCC +0.0002, AUC +0.0021)
and costs 0.0215 macro-F1; it does improve ECE to 0.0522. The frozen MiniLM
comparison is 0.0253 macro-F1 / 0.0273 MCC below the reference, so most of the
signal comes from the task-tuned FinBERT initialization rather than from the
fusion head alone.

The frozen DeBERTa arm shows the opposite trade-off: it gains +0.0838 MCC and
improves ECE by 0.0115 relative to the reference, but loses 0.0647 macro-F1 and
0.0529 ticker-day F1. Its test predictions collapse toward DOWN (UP recall
0.038, confusion matrix row UP → 361 DOWN / 68 FLAT / 17 UP), and validation MCC
selected epoch 1 as "best". Against B2 the paired deltas are Δ macro-F1 −0.0564
[−0.1047, −0.0051] and Δ MCC +0.1060 [+0.0282, +0.1846], so it still fails the
joint +0.03 macro-F1 requirement. It is not a better balanced model.

The first DeBERTa attempt produced `NaN` training loss because
`freeze_layers=12` did not freeze all encoder parameters: DeBERTa keeps
`encoder.rel_embeddings` and a final `encoder.LayerNorm` outside
`encoder.layer`. The model now freezes the entire encoder when
`freeze_layers >= n_layers`; the rerun had finite loss in every epoch. The old
`NaN` checkpoint remains quarantined under `experiment_state/partial_runs/`.

These encoder/length comparisons are single-seed and were not used for model
selection.

---

## 5. Calibration

Scalar temperature scaling is fitted on validation 2024 and applied unchanged to
test. The headline uses the NLL-optimal temperature (T=1.212). The ECE-optimal
validation temperature (T=2.756) yields test ECE 0.0508, so the 0.05 target is
missed by a small margin even under the diagnostic objective. Temperature
changes probabilities, not argmax, so macro-F1 and MCC are identical before and
after calibration.

Why the miss: the 2024 → 2025–2026 class/regime shift changes the confidence
distribution. Vector scaling and class-wise temperatures were tested as
diagnostics; both improved validation ECE but generalized worse to test, so they
were not used for the headline result.

---

## 6. Leakage audit

| check | result |
|---|---|
| posts with `created_utc >= t0` | 0 |
| post IDs shared train/test | 0 |
| shuffled-label AUC | 0.4969 ± 0.0153 |
| class-prob mutual information with direction | 0.1158 total (GENUINE_OPINION 0.0520 largest) |
| stage-1 classifier overlap with direction test | 373 / 463 manual test posts (80.6%) seen during classifier training |
| clean-subset sensitivity (classifier-unseen posts) | macro-F1 0.4384, MCC 0.1642; Δ vs B2 +0.0081 / +0.0193, CIs include 0 |
| strict manual-only held-out subset (121 rows / 90 posts) | macro-F1 0.3955, MCC 0.0774; Δ vs B2 −0.0033 / −0.0022, CIs include 0 |
| temporal split | train 2019–2023, val 2024, test 2025–2026 |
| embargo | 10 sessions; non-purged train/val exits checked against cutoffs |

Residual risk: the frozen stage-1 classifier used a random 80/20 split over all
years, so most manual test posts were seen during classifier training. Its class
probabilities are model inputs (and B6), so this is auxiliary-label leakage. The
clean subset — classifier-held-out manual posts plus classifier-prelabelled
posts — gives macro-F1 0.4384 / MCC 0.1642 and still misses the +0.03 margin,
so the negative verdict does not depend on this path. The stricter manual-only
held-out subset is small (121 rows / 90 posts) and shows no positive effect
either (Δ macro-F1 −0.0033, Δ MCC −0.0022, both CIs include zero). Full report:
`classifier_overlap_audit.json`.

Residual risk: Reddit edit history is not available from the Arctic Shift
snapshot, so a post could have been edited after publication. Scores and comment
counts are also final snapshot values, so votes cast after the entry open may
leak into engagement/aggregate features. These risks are documented rather than
claimed as solved.

Full machine-readable audit: `leakage_audit.json`.

---

## 7. Economic sanity check (ticker-day, weekly rebalance)

| portfolio | annualized return | Sharpe | max drawdown | hit rate |
|---|---:|---:|---:|---:|
| long-short, 5 bps | −42.6% | −0.72 | −58.6% | 48.9% |
| long-short, 10 bps | −45.4% | −0.81 | −59.8% | 48.9% |
| random score, mean over 100 | −5.1% | — | — | — |
| SPY 5-day mean over same ticker-days | ≈ 0.0% (annualized approx +7.1%) | — | — | — |

The strategy loses money after costs. Per TZ §6.3 this is an allowed outcome and
is not used as an acceptance criterion. Overlapping 5-day windows, 662 groups and
no execution model make this a sanity check only.

---

## 8. Why the model did not clear the bar

1. Half the class labels are classifier prelabels; the classifier itself has
   κ 0.573 against manual labels, so the effective label quality is limited.
2. Market-only B5 reaches MCC 0.127; the incremental text signal is small.
3. The 3-class volatility-threshold target is noisy, and FLAT is over-represented.
4. 662 test ticker-day groups with overlapping forward windows are a small
   effective sample.
5. The pre-registered protocol forbids test-set threshold tuning, which is the
   main reason the +0.03 margin was not chased.
6. The stage-1 classifier's random split overlaps most manual test posts; the
   clean-subset check still misses the margin, so this residual leakage is not
   what creates or hides the main effect.
7. Validation-MCC early stopping is sensitive to class collapse: the frozen
   DeBERTa arm selected epoch 1 (validation MCC 0.2557) and then predicted DOWN
   for 361 of 446 true-UP test rows, so its test MCC 0.2397 came with macro-F1
   0.3655. The primary FinBERT run does not show this pattern, but the
   comparison shows why MCC alone is not sufficient.

Legitimate next steps: human-audited class labels, walk-forward retraining,
learned attention aggregation, longer context, and a larger/delisted universe.

---

## 9. Artifact index

| artifact | path |
|---|---|
| model card | `model_card.md` |
| data card | `data_card.md` |
| data quality | `data_quality.md` |
| baselines | `baselines.md`, `baselines_metrics.json` |
| model metrics | `model_metrics.json`, `runs/<run_id>/model_metrics.json` |
| errors | `error_analysis.md` |
| leakage | `leakage_audit.json` |
| seeds/ablations | `ablations.md`, `seed_summary.json` |
| walk-forward | `walk_forward.md`, `walk_forward.json` |
| classifier proxy audit | `classifier_audit_proxy.json` |
| classifier / test overlap audit | `classifier_overlap_audit.json` |
| gold class-label audit | `gold_audit.json` (status: blocked until human labels exist) |
| ticker precision/recall audit | `ticker_gold_audit.json` (status: blocked until human labels exist) |
| per-run predictions | `runs/<run_id>/model_predictions.parquet`, `ticker_day_predictions.parquet` |
| backtest PnL | `runs/<run_id>/backtest_pnl.csv` |
| API | `../src/serve.py` |

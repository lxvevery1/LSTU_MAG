# Model card — finbert_h5_seed42

## Overview
- Task: 3-class direction (`DOWN`, `FLAT`, `UP`) for a post x ticker pair and for the ticker-day aggregate.
- Primary horizon: H=5 trading days; auxiliary heads: H=1, 3, 5, 10.
- Encoder: fine-tuned FinBERT checkpoint (`ProsusAI/finbert` + 4-class WSB head), initialized from the project classifier.
- Fusion: `[CLS]` embedding (768) + numeric/market features + ticker embedding + flair embedding -> 2-layer MLP -> horizon heads.
- Frozen layers: 8/12; max_len=256.
- Loss: influence-weighted cross-entropy (label smoothing 0.05, class-balance weights, `λ_class=0.3`). Focal loss (γ=1.5) was compared as a TZ §5.3 ablation and was worse; weighted CE remains the reference.
- Influence weights are used **only** as loss weights and ticker-day aggregation weights, never as model inputs.

## Data
- 16,000 WSB posts (2,000/year, 2019–2026); 8,000 manual labels + 8,000 frozen-FinBERT prelabels.
- 4,625 valid post x ticker pairs at H=5; 2,357 ticker-day groups; temporal split with a 10-session embargo.
- Directions are beta-adjusted forward abnormal returns vs SPY; see `data_card.md`.

## Test metrics (H=5, 1270 rows)
| metric | raw | calibrated | 95% block-bootstrap CI (calibrated) |
|---|---:|---:|---|
| Macro-F1 | 0.4302 | 0.4302 | [0.3980, 0.4601] |
| MCC | 0.1559 | 0.1559 | [0.1129, 0.1971] |
| AUC OvR | 0.6014 | 0.6016 | [0.5742, 0.6275] |
| AUC-PR UP | 0.3844 | 0.3838 | [0.3304, 0.4533] |
| Brier | 0.5979 | 0.5949 | — |
| ECE | 0.0776 | 0.0665 | — |

Ticker-day aggregate: macro-F1 0.4024, MCC 0.1284, AUC 0.5873.

## Per-horizon metrics (calibrated)
| H | n | macro-F1 | MCC | AUC | ECE |
|---:|---:|---:|---:|---:|---:|
| 1 | 1270 | 0.3555 | 0.0705 | 0.5733 | 0.0511 |
| 3 | 1253 | 0.3997 | 0.1285 | 0.5922 | 0.0745 |
| 5 | 1270 | 0.4302 | 0.1559 | 0.6016 | 0.0665 |
| 10 | 1270 | 0.4680 | 0.1873 | 0.6213 | 0.0637 |

## Baseline comparison
Best baseline by test macro-F1: **B2** (macro-F1 0.4207, MCC 0.1316, AUC 0.6016).

Paired block-bootstrap differences on the 1270 common test rows (model − baseline):
- Δ macro-F1 = **+0.0093** [-0.0291, +0.0449]
- Δ MCC = **+0.0235** [-0.0287, +0.0765]

## Seed variability (test H=5)
| seed | macro-F1 | MCC | AUC | ECE |
|---:|---:|---:|---:|---:|
| 42 | 0.4302 | 0.1559 | 0.6016 | 0.0665 |
| 43 | 0.4124 | 0.1387 | 0.6007 | 0.0762 |
| 44 | 0.4399 | 0.1587 | 0.6145 | 0.0590 |
| **mean ± std** | **0.4275 ± 0.0140** | **0.1511 ± 0.0109** | **0.6056 ± 0.0077** | **0.0672 ± 0.0086** |

At the seed mean the gain over B2 is +0.0068 macro-F1 and +0.0195 MCC, still
below the +0.03 target. The negative verdict is not an artifact of seed 42.

## Loss comparison (TZ §5.3)
The reference objective is influence-weighted cross-entropy. A focal-loss arm
(`loss_type=focal`, γ=1.5) was retrained with the same seed, data, frozen layers
and schedule:

| run | loss | macro-F1 | MCC | AUC | ECE | ticker-day F1 |
|---|---|---:|---:|---:|---:|---:|
| `finbert_h5_seed42` | weighted CE | 0.4302 | 0.1559 | 0.6016 | 0.0665 | 0.4024 |
| `ablate_focal_loss_seed42` | focal, γ=1.5 | 0.4227 | 0.1390 | 0.6019 | 0.0753 | 0.3910 |
| **Δ (focal − reference)** | | **−0.0075** | **−0.0169** | **+0.0003** | **+0.0088** | **−0.0115** |

Focal loss does not improve the ranking metrics and degrades calibration, so
the reference model keeps weighted CE. The `ablate_weighted_ce_only_seed42`
control reproduces the reference exactly, so the focal difference is not
run-to-run noise.

## Encoder and max-length comparison (single seed)
| run | encoder | max_len | macro-F1 | MCC | AUC | ECE |
|---|---|---:|---:|---:|---:|---:|
| `finbert_h5_seed42` | FinBERT, 8/12 frozen | 256 | 0.4302 | 0.1559 | 0.6016 | 0.0665 |
| `finbert_h5_maxlen512_seed42` | FinBERT, 8/12 frozen | 512 | 0.4087 | 0.1561 | 0.6037 | 0.0522 |
| `minilm_frozen_h5_seed42` | MiniLM-L6-v2, frozen | 256 | 0.4048 | 0.1286 | 0.6008 | 0.0616 |
| `deberta_v3_small_frozen_h5_seed42` | DeBERTa-v3-small, frozen | 256 | 0.3655 | 0.2397 | 0.6062 | 0.0550 |

`max_len=512` does not improve MCC/AUC and costs macro-F1; the frozen MiniLM arm
is 0.0253 macro-F1 / 0.0273 MCC below the reference. The frozen DeBERTa arm has
higher MCC (+0.0838) and better ECE (−0.0115) but much lower macro-F1 (−0.0647)
and ticker-day F1 (−0.0529); its predictions collapse toward DOWN (UP recall
0.038), and validation MCC selected epoch 1. Against B2 it is Δ macro-F1 −0.0564
and Δ MCC +0.1060, so it does not meet the joint +0.03 criterion. None of these
variants changes the negative verdict. All comparisons are single-seed and not
used for model selection.

## Walk-forward check
| fold | train | val | test | n | macro-F1 | MCC | AUC | ECE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 2019–2021 | 2022 | 2023 | 453 | 0.4383 | 0.1740 | 0.5918 | 0.0822 |
| 2024 | 2019–2022 | 2023 | 2024 | 707 | 0.4241 | 0.1893 | 0.6130 | 0.0275 |
| 2025 | 2019–2023 | 2024 | 2025 | 662 | 0.4090 | 0.1386 | 0.5855 | 0.0944 |

Mean ± std across folds: macro-F1 0.4238 ± 0.0147, MCC 0.1673 ± 0.0260,
AUC 0.5968 ± 0.0144. Folds were not used for model selection.

## Acceptance checks (TZ §8.2)
| criterion | target | result | status |
|---|---|---:|---|
| Test macro-F1 vs best baseline | ≥ +0.03 and CI > 0 | +0.0093 | FAIL |
| Test MCC vs best baseline | ≥ +0.03 and CI > 0 | +0.0235 | FAIL |
| H=5 AUC-ROC | ≥ 0.55 | 0.6016 | PASS |
| ECE after temperature scaling | ≤ 0.05 | 0.0665 | FAIL |
| Shuffled-label AUC | ≈ 0.5 | 0.497 | PASS |

**Result: the neural model does not clear the +0.03 MCC / +0.03 macro-F1 margin over the best baseline.**
This is reported as a negative result. The test set was not used for model selection or calibration;
the paired bootstrap CIs include zero for both metrics. The ECE-optimal scalar temperature on
validation (T=2.756) gives test ECE 0.0508; the miss is small
but not zero. See `error_analysis.md` and `reports/ablations.md` for the analysis.

## Intended use
Research and educational analysis of social-media sentiment. Not investment advice, not a trading system, not causal evidence that a post moves a stock.

## Limitations
- Direction labels are noisy and associative, not causal.
- New-post class labels are classifier prelabels until the human review queue is completed; the classifier-vs-manual holdout Cohen's κ is 0.573 [0.540, 0.605], below the 0.65 target.
- 8,000 new posts are sampled from the local Arctic Shift pool; 2026 is partial.
- The 30-ticker universe is the TZ §4.2 fixed list (including SPY/QQQ); it suffers from survivorship/selection bias.
- Only 22.9% of posts mention a universe ticker; the rest cannot enter the pair dataset.
- Fine-tuning a 6 GB GPU model for 3 epochs is a compute-budget compromise. Seeds 42/43/44 give test macro-F1 0.4275 ± 0.0140 and MCC 0.1511 ± 0.0109; the negative result is not specific to seed 42.
- The ticker-day aggregate is a fixed influence-weighted average, not a learned attention pool.
- Reddit `score`/`num_comments` and the document text are Arctic Shift snapshot values. We cannot reconstruct the values exactly at `t0`, so votes/edits after publication are a residual leakage risk that this dataset cannot eliminate.
- The stage-1 classifier used a random (not temporal) 80/20 split over all years, so 373/463 manual test posts (80.6%) were seen during its training. Class probabilities are inputs, making this auxiliary-label leakage. The clean-subset check still gives macro-F1 0.4384 / MCC 0.1642 and misses the +0.03 margin (Δ vs B2 +0.0081 / +0.0193); the stricter manual-only held-out subset (121 rows / 90 posts) gives Δ −0.0033 / −0.0022. See `classifier_overlap_audit.json`.
- Test ECE is slightly above 0.05; the calibration shift between 2024 and 2025–2026 is documented.
- Encoder and max-length comparisons (`max_len=512`, frozen MiniLM, frozen DeBERTa-v3-small) are single-seed; they are reported in `reports/ablations.md`, not used for model selection.
- Early stopping on validation MCC can select a class-collapsed checkpoint. The frozen DeBERTa comparison did exactly that: validation MCC selected epoch 1, and test UP recall fell to 0.038. Macro-F1 and per-class recall are therefore reported alongside MCC.

## Ethical / legal
Reddit data are used for research; author names are hashed. Respect Reddit/Arctic Shift terms of use. Do not use the output as personalized financial advice.

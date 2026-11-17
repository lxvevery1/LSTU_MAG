# Model card — deberta_v3_small_frozen_h5_seed42

## Overview
- Task: 3-class direction (`DOWN`, `FLAT`, `UP`) for a post x ticker pair and for the ticker-day aggregate.
- Primary horizon: H=5 trading days; auxiliary heads: H=1, 3, 5, 10.
- Encoder: frozen DeBERTa-v3-small encoder; only the fusion layers and heads are trained.
- Fusion: pooled encoder embedding (768) + numeric/market features + ticker embedding + flair embedding -> 2-layer MLP -> horizon heads.
- Frozen layers: all 6/6; max_len=256.
- Loss: influence-weighted cross-entropy (label smoothing 0.05, class-balance weights, λ_class=0.3).
- Influence weights are used **only** as loss weights and ticker-day aggregation weights, never as model inputs.

## Data
- 16,000 WSB posts (2,000/year, 2019–2026); 8,000 manual labels + 8,000 frozen-FinBERT prelabels.
- 4,625 valid post x ticker pairs at H=5; 2,357 ticker-day groups; temporal split with a 10-session embargo.
- Directions are beta-adjusted forward abnormal returns vs SPY; see `data_card.md`.

## Test metrics (H=5, 1270 rows)
| metric | raw | calibrated | 95% block-bootstrap CI (calibrated) |
|---|---:|---:|---|
| Macro-F1 | 0.3655 | 0.3655 | [0.3308, 0.3987] |
| MCC | 0.2397 | 0.2397 | [0.1878, 0.2873] |
| AUC OvR | 0.6065 | 0.6062 | [0.5776, 0.6360] |
| AUC-PR UP | 0.3748 | 0.3742 | [0.3295, 0.4336] |
| Brier | 0.5982 | 0.5986 | — |
| ECE | 0.0497 | 0.0550 | — |

Ticker-day aggregate: macro-F1 0.3495, MCC 0.2125, AUC 0.5896.

## Per-horizon metrics (calibrated)
| H | n | macro-F1 | MCC | AUC | ECE |
|---:|---:|---:|---:|---:|---:|
| 1 | 1270 | 0.3194 | 0.0479 | 0.5826 | 0.0200 |
| 3 | 1253 | 0.3243 | 0.1353 | 0.6143 | 0.0843 |
| 5 | 1270 | 0.3655 | 0.2397 | 0.6062 | 0.0550 |
| 10 | 1270 | 0.4087 | 0.1443 | 0.6382 | 0.0429 |

## Baseline comparison
Best baseline by test macro-F1: **B2** (macro-F1 0.4207, MCC 0.1316, AUC 0.6016).

Paired block-bootstrap differences on the 1270 common test rows (model − baseline):
- Δ macro-F1 = **-0.0564** [-0.1047, -0.0051]
- Δ MCC = **+0.1060** [+0.0282, +0.1846]

## Acceptance checks (TZ §8.2)
| criterion | target | result | status |
|---|---|---:|---|
| Test macro-F1 vs best baseline | ≥ +0.03 and CI > 0 | -0.0564 | FAIL |
| Test MCC vs best baseline | ≥ +0.03 and CI > 0 | +0.1060 | PASS |
| H=5 AUC-ROC | ≥ 0.55 | 0.6062 | PASS |
| ECE after temperature scaling | ≤ 0.05 | 0.0550 | FAIL |
| Shuffled-label AUC | ≈ 0.5 | 0.500 | PASS |

**Result: the neural model does not clear both the +0.03 MCC and +0.03 macro-F1 requirements over the best baseline.**
This is reported as a negative result. The test set was not used for model selection or calibration;
the paired bootstrap intervals for the best-baseline comparison are reported above. The ECE-optimal scalar temperature on
validation (T=3.807) gives test ECE 0.0684; the miss is small
but not zero. The weighted-CE objective was compared with focal loss (γ=1.5) under the same
protocol; focal did not improve test macro-F1 or MCC (`reports/ablations.md`). See
`error_analysis.md` and `reports/ablations.md` for the analysis.

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
- Early stopping on validation MCC selected epoch 1 for this run, and test UP recall collapsed to 0.038. Macro-F1 and per-class recall are reported alongside MCC for this reason.

## Ethical / legal
Reddit data are used for research; author names are hashed. Respect Reddit/Arctic Shift terms of use. Do not use the output as personalized financial advice.

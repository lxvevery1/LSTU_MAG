# Model card — ablate_focal_loss_seed42

## Overview
- Task: 3-class direction (`DOWN`, `FLAT`, `UP`) for a post x ticker pair and for the ticker-day aggregate.
- Primary horizon: H=5 trading days; auxiliary heads: H=1, 3, 5, 10.
- Encoder: fine-tuned FinBERT checkpoint (`ProsusAI/finbert` + 4-class WSB head), initialized from the project classifier.
- Fusion: `[CLS]` embedding (768) + numeric/market features + ticker embedding + flair embedding -> 2-layer MLP -> horizon heads.
- Frozen layers: 8/12; max_len=256.
- Loss: focal loss (γ=1.5, label smoothing 0.05). This is the TZ §5.3 comparison arm; it is worse than the weighted-CE reference on macro-F1, MCC and ticker-day F1.
- Influence weights are used **only** as loss weights and ticker-day aggregation weights, never as model inputs.

## Data
- 16,000 WSB posts (2,000/year, 2019–2026); 8,000 manual labels + 8,000 frozen-FinBERT prelabels.
- 4,625 valid post x ticker pairs at H=5; 2,357 ticker-day groups; temporal split with a 10-session embargo.
- Directions are beta-adjusted forward abnormal returns vs SPY; see `data_card.md`.

## Test metrics (H=5, 1270 rows)
| metric | raw | calibrated | 95% block-bootstrap CI (calibrated) |
|---|---:|---:|---|
| Macro-F1 | 0.4227 | 0.4227 | [0.3903, 0.4558] |
| MCC | 0.1390 | 0.1390 | [0.0958, 0.1809] |
| AUC OvR | 0.6021 | 0.6019 | [0.5737, 0.6280] |
| AUC-PR UP | 0.3870 | 0.3875 | [0.3344, 0.4582] |
| Brier | 0.5950 | 0.5953 | — |
| ECE | 0.0714 | 0.0753 | — |

Ticker-day aggregate: macro-F1 0.3910, MCC 0.1038, AUC 0.5868.

## Per-horizon metrics (calibrated)
| H | n | macro-F1 | MCC | AUC | ECE |
|---:|---:|---:|---:|---:|---:|
| 1 | 1270 | 0.3532 | 0.0650 | 0.5705 | 0.0509 |
| 3 | 1253 | 0.4002 | 0.1323 | 0.5927 | 0.0752 |
| 5 | 1270 | 0.4227 | 0.1390 | 0.6019 | 0.0753 |
| 10 | 1270 | 0.4786 | 0.1876 | 0.6245 | 0.0545 |

## Baseline comparison
Best baseline by test macro-F1: **B2** (macro-F1 0.4207, MCC 0.1316, AUC 0.6016).

Paired block-bootstrap differences on the 1270 common test rows (model − baseline):
- Δ macro-F1 = **+0.0016** [-0.0393, +0.0395]
- Δ MCC = **+0.0065** [-0.0456, +0.0598]

## Acceptance checks (TZ §8.2)
| criterion | target | result | status |
|---|---|---:|---|
| Test macro-F1 vs best baseline | ≥ +0.03 and CI > 0 | +0.0016 | FAIL |
| Test MCC vs best baseline | ≥ +0.03 and CI > 0 | +0.0065 | FAIL |
| H=5 AUC-ROC | ≥ 0.55 | 0.6019 | PASS |
| ECE after temperature scaling | ≤ 0.05 | 0.0753 | FAIL |
| Shuffled-label AUC | ≈ 0.5 | 0.497 | PASS |

**Result: the neural model does not clear the +0.03 MCC / +0.03 macro-F1 margin over the best baseline.**
This is reported as a negative result. The test set was not used for model selection or calibration;
the paired bootstrap CIs include zero for both metrics. The ECE-optimal scalar temperature on
validation (T=2.541) gives test ECE 0.0603; the miss is small
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

## Ethical / legal
Reddit data are used for research; author names are hashed. Respect Reddit/Arctic Shift terms of use. Do not use the output as personalized financial advice.

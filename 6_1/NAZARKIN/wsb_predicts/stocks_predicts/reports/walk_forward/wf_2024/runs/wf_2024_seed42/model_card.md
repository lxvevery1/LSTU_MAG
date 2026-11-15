# Model card — wf_2024_seed42

## Overview
- Task: 3-class direction (`DOWN`, `FLAT`, `UP`) for a post x ticker pair and for the ticker-day aggregate.
- Primary horizon: H=5 trading days; auxiliary heads: H=1, 3, 5, 10.
- Encoder: fine-tuned FinBERT checkpoint (`ProsusAI/finbert` + 4-class WSB head), initialized from the project classifier.
- Fusion: `[CLS]` embedding (768) + numeric/market features + ticker embedding + flair embedding -> 2-layer MLP -> horizon heads.
- Frozen layers: 8/12; max_len=256.
- Influence weights are used **only** as loss weights and ticker-day aggregation weights, never as model inputs.

## Data
- 16,000 WSB posts (2,000/year, 2019–2026); 8,000 manual labels + 8,000 frozen-FinBERT prelabels.
- 4,625 valid post x ticker pairs at H=5; 2,357 ticker-day groups; temporal split with a 10-session embargo.
- Directions are beta-adjusted forward abnormal returns vs SPY; see `data_card.md`.

## Test metrics (H=5, 707 rows)
| metric | raw | calibrated | 95% block-bootstrap CI (calibrated) |
|---|---:|---:|---|
| Macro-F1 | 0.4241 | 0.4241 | [0.3702, 0.4766] |
| MCC | 0.1893 | 0.1893 | [0.1080, 0.2697] |
| AUC OvR | 0.6130 | 0.6130 | [0.5715, 0.6478] |
| AUC-PR UP | 0.4915 | 0.4946 | [0.3760, 0.6154] |
| Brier | 0.5967 | 0.5968 | — |
| ECE | 0.0205 | 0.0275 | — |

Ticker-day aggregate: macro-F1 0.4368, MCC 0.2108, AUC 0.6226.

## Per-horizon metrics (calibrated)
| H | n | macro-F1 | MCC | AUC | ECE |
|---:|---:|---:|---:|---:|---:|
| 1 | 707 | 0.3615 | 0.1167 | 0.5819 | 0.0844 |
| 3 | 707 | 0.4137 | 0.1660 | 0.5821 | 0.0476 |
| 5 | 707 | 0.4241 | 0.1893 | 0.6130 | 0.0275 |
| 10 | 707 | 0.4648 | 0.2017 | 0.6363 | 0.0353 |

## Baseline comparison
Best baseline by test macro-F1: **n/a** (macro-F1 nan, MCC nan, AUC nan).

Paired block-bootstrap differences on the 0 common test rows (model − baseline):
- Δ macro-F1 = **+nan** [+nan, +nan]
- Δ MCC = **+nan** [+nan, +nan]

## Acceptance checks (TZ §8.2)
| criterion | target | result | status |
|---|---|---:|---|
| Test macro-F1 vs best baseline | ≥ +0.03 and CI > 0 | +nan | FAIL |
| Test MCC vs best baseline | ≥ +0.03 and CI > 0 | +nan | FAIL |
| H=5 AUC-ROC | ≥ 0.55 | 0.6130 | PASS |
| ECE after temperature scaling | ≤ 0.05 | 0.0275 | PASS |
| Shuffled-label AUC | ≈ 0.5 | 0.499 | PASS |

**Result: the neural model does not clear the +0.03 MCC / +0.03 macro-F1 margin over the best baseline.**
This is reported as a negative result. The test set was not used for model selection or calibration;
the paired bootstrap CIs include zero for both metrics. The ECE-optimal scalar temperature on
validation (T=2.780) gives test ECE 0.0796; the miss is small
but not zero. See `error_analysis.md` and `reports/ablations.md` for the analysis.

## Intended use
Research and educational analysis of social-media sentiment. Not investment advice, not a trading system, not causal evidence that a post moves a stock.

## Limitations
- Direction labels are noisy and associative, not causal.
- New-post class labels are classifier prelabels until the human review queue is completed; the classifier-vs-manual holdout Cohen's κ is 0.573 [0.540, 0.605], below the 0.65 target.
- 8,000 new posts are sampled from the local Arctic Shift pool; 2026 is partial.
- The 30-ticker universe is the TZ §4.2 fixed list (including SPY/QQQ); it suffers from survivorship/selection bias.
- Only 22.9% of posts mention a universe ticker; the rest cannot enter the pair dataset.
- Fine-tuning a 6 GB GPU model for 3 epochs is a compute-budget compromise; seeds 42/43/44 should be reported as mean ± std.
- The ticker-day aggregate is a fixed influence-weighted average, not a learned attention pool.
- Reddit `score`/`num_comments` and the document text are Arctic Shift snapshot values. We cannot reconstruct the values exactly at `t0`, so votes/edits after publication are a residual leakage risk that this dataset cannot eliminate.
- Test ECE is slightly above 0.05; the calibration shift between 2024 and 2025–2026 is documented.
- No DeBERTa checkpoint was available offline; MiniLM and max_len=512 runs are reported separately when completed.

## Ethical / legal
Reddit data are used for research; author names are hashed. Respect Reddit/Arctic Shift terms of use. Do not use the output as personalized financial advice.

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

| model | macro-F1 | MCC | AUC | AUC-PR UP | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| B0 majority/random | 0.1715 | 0.0000 | 0.5000 | 0.3512 | 1.3071 | 0.6535 |
| B1 TF-IDF + LogReg | 0.3824 | 0.0723 | 0.5703 | 0.3440 | 0.6542 | 0.0976 |
| **B2 TF-IDF + LightGBM** | **0.4207** | **0.1316** | **0.6016** | 0.3847 | 0.7425 | 0.2351 |
| B3 frozen FinBERT + LogReg | 0.3543 | 0.0334 | 0.5169 | 0.3532 | 0.9711 | 0.4053 |
| B4 DD/bull-bear heuristic | 0.2944 | 0.0225 | 0.5090 | 0.3568 | 1.2866 | 0.6433 |
| B5 market only | 0.4159 | 0.1274 | 0.5989 | 0.4069 | 0.6500 | 0.0330 |
| B6 class probs only | 0.2549 | −0.0296 | 0.4766 | 0.3566 | 0.6698 | 0.0436 |

B5 is the leakage control. Its MCC 0.127 is close to the full model's 0.156,
which is the main reason the text signal does not produce a +0.03 gain.

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
| temporal split | train 2019–2023, val 2024, test 2025–2026 |
| embargo | 10 sessions; non-purged train/val exits checked against cutoffs |

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
| gold class-label audit | `gold_audit.json` (status: blocked until human labels exist) |
| ticker precision/recall audit | `ticker_gold_audit.json` (status: blocked until human labels exist) |
| per-run predictions | `runs/<run_id>/model_predictions.parquet`, `ticker_day_predictions.parquet` |
| backtest PnL | `runs/<run_id>/backtest_pnl.csv` |
| API | `../src/serve.py` |

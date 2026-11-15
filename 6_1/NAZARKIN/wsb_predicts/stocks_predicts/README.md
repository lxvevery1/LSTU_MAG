# WSB post × ticker direction model

End-to-end implementation of [`TZ_stock_direction.md`](TZ_stock_direction.md):
a 16,000-post r/WallStreetBets corpus, ticker linking, forward beta-adjusted
direction labels, baselines B0–B6, a FinBERT-fusion model, calibration,
leakage audit, HTTP API and tests.

> **Research use only.** This is not investment advice, not a trading system,
> and not causal evidence that a post moves a stock.

---

## 1. Headline result (honest)

The neural model is **not statistically better than the best baseline** by the
pre-registered +0.03 MCC / +0.03 macro-F1 margin. The test split was never used
for model selection, early stopping or calibration.

| test H=5 (2025–2026, n=1,270) | model | best baseline B2 | paired Δ |
|---|---:|---:|---:|
| macro-F1 | 0.4302 | 0.4207 | **+0.0093** [−0.0291, +0.0449] |
| MCC | 0.1559 | 0.1316 | **+0.0235** [−0.0287, +0.0765] |
| AUC OvR | 0.6016 | 0.6016 | +0.0000 |
| AUC-PR (UP) | 0.3838 | 0.3847 | −0.0009 |
| ECE (NLL temperature) | 0.0665 | — | target ≤ 0.05 |

* The +0.03 margin is missed on both metrics and both 95% block-bootstrap CIs
  include zero.
* H=5 AUC 0.6016 clears the 0.55 target; shuffled-label AUC is 0.4969 ± 0.0153,
  consistent with no label leakage.
* ECE is 0.0665 with the standard NLL-fitted temperature (T=1.212). The
  ECE-optimal validation temperature (T=2.76) gives test ECE 0.0508 — still
  marginally above 0.05. This is reported as a near miss, not tuned away on test.
* The long-short sanity-check portfolio loses money after costs. That is an
  allowed and honest outcome per the TZ; it is **not** an acceptance criterion.

Everything is written up in [`reports/model_card.md`](reports/model_card.md),
[`reports/error_analysis.md`](reports/error_analysis.md),
[`reports/ablations.md`](reports/ablations.md) and
[`reports/leakage_audit.json`](reports/leakage_audit.json).

---

## 2. Quick start on a clean machine

```bash
# from the repository root that contains stocks_predicts/
python -m venv .venv
source .venv/bin/activate
pip install -r stocks_predicts/requirements.txt

cd stocks_predicts

# If the FinBERT classifier checkpoint is not already on this machine, point
# configs/model.yaml -> model.encoder_path / classifier_checkpoint at a local
# ProsusAI/finbert fine-tuned checkpoint, or at `ProsusAI/finbert` itself.

make test          # fast tests, no transformer load
make data          # 16k corpus -> labels -> tickers -> price labels -> dataset
make baselines     # B0-B6
make train         # primary seed 42
make eval          # temperature scaling, bootstrap, audit, model card
make serve         # http://localhost:8080/docs
```

The checked-in artifacts were produced with these exact commands; the Makefile
uses `../.venv/bin/python` automatically when that virtualenv exists, otherwise
the active `python`.

---

## 3. Data

| item | value |
|---|---|
| posts | 16,000 (2,000 per year, 2019–2026) |
| class labels | 8,000 manual (existing project) + 8,000 frozen-FinBERT prelabels |
| class distribution | 8,674 NOISE_MEME · 2,893 FINANCIAL_CLAIM · 2,864 GENUINE_OPINION · 1,569 ANALYSIS_DD |
| ticker universe | 30-symbol fixed list from TZ §4.2 (includes SPY and QQQ) |
| linked posts | 3,658 / 16,000 (22.9%); 4,844 post × ticker links |
| valid H=5 pairs | 4,625 / 4,844 (95.5%) |
| ticker-day groups | 2,357 across all splits; 662 in test |
| split | train 2019–2023, val 2024, test 2025–2026 |
| embargo | 10 trading sessions, plus forward-window purge at boundaries |

**Direction label.** Entry is the open of the first session strictly after
`created_utc`; exit is the close of the H-th session including entry day.
`a = r − β·r_SPY`, `θ = max(0.005, 0.5·σ_20d)`, then
`UP if a ≥ θ`, `DOWN if a ≤ −θ`, else `FLAT`. β and σ use only sessions before
the entry open. H=5 is primary; H=1/3/10 are auxiliary heads.

Known issues, documented rather than hidden:

* 2026 is partial (through 2026-11-14).
* 157 posts (0.98%) share normalized text, mostly manual posts whose body is
  `[removed by moderator]`; post IDs are unique, but these rows carry little text.
* Prices are split-adjusted but not dividend-adjusted.
* New-post class labels are model-generated until the 500-post gold audit is
  completed; the sample and review queue are in `data/raw/gold_audit_sample.json`
  and `reports/prelabel_report.json`. Cohen's κ cannot be claimed without
  independent human labels. Run `make gold-audit` after filling `gold_label` to
  score it; the current status is `blocked` in `reports/gold_audit.json`. The
  classifier's agreement with manual labels on its own 20% holdout is κ=0.573
  [0.540, 0.605] — a proxy, not a gold audit.
* Ticker precision/recall on a hand-checked 300-post sample is **not measured
  yet**: the fixed sample is `data/raw/ticker_gold_sample.json`, and
  `make ticker-gold-audit` writes `reports/ticker_gold_audit.json` once a human
  fills `gold_tickers` (current status: `blocked`, 0/300 labelled). Casual
  diagnostics: universe cashtags account for 68.6% of all cashtags, and 22.9%
  of posts contain at least one universe ticker.
* FLAT rate at H=5 is 35.1%, slightly above the 20–35% design guidance. The
  formula was not changed after seeing test results.

Full details: [`reports/data_card.md`](reports/data_card.md),
[`reports/data_quality.md`](reports/data_quality.md),
[`reports/dataset_report.json`](reports/dataset_report.json).

---

## 4. Baselines (test, H=5)

| model | macro-F1 | MCC | AUC | AUC-PR UP | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| B0 majority/random | 0.1715 | 0.0000 | 0.5000 | 0.3512 | 1.3071 | 0.6535 |
| B1 TF-IDF + LogisticRegression | 0.3824 | 0.0723 | 0.5703 | 0.3440 | 0.6542 | 0.0976 |
| **B2 TF-IDF + LightGBM** | **0.4207** | **0.1316** | **0.6016** | 0.3847 | 0.7425 | 0.2351 |
| B3 frozen FinBERT + LogReg | 0.3543 | 0.0334 | 0.5169 | 0.3532 | 0.9711 | 0.4053 |
| B4 DD/bull-bear heuristic | 0.2944 | 0.0225 | 0.5090 | 0.3568 | 1.2866 | 0.6433 |
| B5 market features only | 0.4159 | 0.1274 | 0.5989 | 0.4069 | 0.6500 | 0.0330 |
| B6 class probabilities only | 0.2549 | −0.0296 | 0.4766 | 0.3566 | 0.6698 | 0.0436 |

B5 is a leakage control: it sees only market features known before the entry
open and still reaches MCC 0.127, which shows how much of the signal is market
regime, not text. B2 is the strongest baseline by macro-F1.

Source: [`reports/baselines.md`](reports/baselines.md),
[`reports/baselines_metrics.json`](reports/baselines_metrics.json).

---

## 5. Model

```
text: "$TICKER [SEP] title [SEP] ticker fragment [SEP] body" -> FinBERT encoder
      (first 8 of 12 layers frozen, last 4 fine-tuned, max_len=256)
    + 31 numeric features (post, time, market, aggregate, 4 class probs)
    + ticker embedding (16) + flair embedding (8)
      -> Linear(…,128) -> GELU -> Dropout(0.2) -> Linear(128,128) -> GELU
      -> heads: H=1,3,5,10 (3 classes each) + auxiliary 4-class post head
```

* Loss: influence-weighted cross-entropy with focal term (γ=1.5), label
  smoothing 0.05, class-balance weights, `λ_class=0.3`.
* Influence weights (`ANALYSIS_DD=1.0`, `FINANCIAL_CLAIM=0.6`,
  `GENUINE_OPINION/ADVICE=0.5`, `NOISE_MEME=0.1`, `UNCERTAIN=0`) are used only as
  loss weights and ticker-day aggregation weights — **never as inputs**.
* Optimizer: AdamW, encoder LR 2e-5, head LR 1e-3, 3 epochs, early stopping on
  validation MCC, mixed precision.
* Ticker-day aggregation: influence × link-share weighted average of post
  probabilities, grouped by `(ticker, t0_date)`.
* Calibration: temperature scaling fitted on 2024 validation, applied to test.

Seed 42 validation macro-F1 0.4325 / MCC 0.1365; test calibrated macro-F1
0.4302 / MCC 0.1559 (temperature changes probabilities, not argmax, so F1/MCC are
identical before and after calibration).

### Seeds and ablations

`make experiments` runs seeds 43/44 and the nine pre-registered ablations:
no influence weight, no ticker fragment, text-only, frozen encoder, no aux loss,
no NOISE_MEME, no class probabilities, weighted-CE-only (no focal term), and a
single-horizon (H=5 only) model. `make aggregate` writes
[`reports/ablations.md`](reports/ablations.md) and
[`reports/seed_summary.json`](reports/seed_summary.json).

`make experiments` with `--preset extra` additionally runs max_len=512 and a
frozen MiniLM encoder comparison. DeBERTa was not available in the offline model
cache; this is documented as an open item rather than silently skipped.

### Walk-forward validation

`make walk-forward` implements the additional TZ §6.1 scheme: train ≤ Y−2,
validation = Y−1, test = Y for Y ∈ {2023, 2024, 2025} with a 10-session
embargo. One year is held out for early stopping/calibration, so training stops
at Y−2 instead of the TZ's literal Y−1. The 2025 fold reuses the primary model's
2025 slice; 2023/2024 are retrained. Results: `reports/walk_forward.md`.

---

## 6. HTTP API

```bash
make serve                      # or: python src/serve.py --run-id finbert_h5_seed42 --port 8080

curl -s localhost:8080/predict_direction \
  -H 'content-type: application/json' \
  -d '{
    "title": "TSLA calls printing",
    "body": "Buying 250C for next Friday, gamma squeeze setup.",
    "ticker": "TSLA",
    "flair": "YOLO",
    "score": 120,
    "num_comments": 14
  }' | python -m json.tool
```

Response (abridged):

```json
{
  "run_id": "finbert_h5_seed42",
  "ticker": "TSLA",
  "t0_date": "2025-01-15",
  "primary_horizon": 5,
  "direction": "UP",
  "confidence": 0.41,
  "probabilities": {"DOWN": 0.27, "FLAT": 0.32, "UP": 0.41},
  "calibrated": true,
  "horizons": {
    "1": {"direction": "FLAT", "probabilities": {"...": "..."}, "confidence": 0.38},
    "3": {"...": "..."}, "5": {"...": "..."}, "10": {"...": "..."}
  },
  "post_class": {"NOISE_MEME": 0.05, "GENUINE_OPINION": 0.61,
                 "FINANCIAL_CLAIM": 0.25, "ANALYSIS_DD": 0.09},
  "post_class_label": "GENUINE_OPINION",
  "influence_weight": 0.5,
  "warnings": []
}
```

Endpoints:

| method | path | notes |
|---|---|---|
| GET | `/health` | liveness + loaded run id |
| GET | `/model_info` | horizons, universe, device, temperature |
| POST | `/predict_direction` | one post × ticker; validates universe and text |
| POST | `/predict_direction/batch` | up to 100 items; per-item errors are returned |
| GET | `/docs` | interactive OpenAPI UI |

The service builds market features from the cached price file and class
probabilities from the frozen 4-class FinBERT classifier. If `created_utc` is
beyond the cached price history, it warns and uses training medians for market
features. Unknown tickers return HTTP 422.

---

## 7. Tests

```bash
make test       # 31 fast tests: metrics, labels, splits/embargo, tickers, features, API, audits, matrix
make test-all   # additionally loads the real checkpoint and runs one forward pass
make lint       # byte-compile src/ and tests/
```

The fast API tests use a stub service so CI does not need a GPU or a 440 MB
checkpoint. The real-model test is opt-in via `WSB_TEST_REAL_MODEL=1`.

---

## 8. Repository map

```
stocks_predicts/
├── configs/                  data.yaml, tickers.yaml, model.yaml, eval.yaml
├── data/
│   ├── raw/                  posts_16k_raw.json, posts_16k_labeled.json, gold_audit_sample.json
│   ├── interim/              post_ticker_links.parquet, finbert_embeddings.npz
│   └── processed/            wsb_direction_dataset_v1.parquet, wsb_pairs_h5.parquet
├── models/
│   ├── finbert_h5_seed42/    model.pt, tokenizer/, feature_pipeline.json
│   └── …                     seed/ablation checkpoints
├── reports/
│   ├── data_card.md, data_quality.md, dataset_report.json
│   ├── baselines.md, baselines_metrics.json, baselines_predictions.parquet
│   ├── model_card.md, model_metrics.json, error_analysis.md
│   ├── leakage_audit.json, ablations.md, seed_summary.json
│   ├── gold_audit.json, ticker_gold_audit.json, walk_forward.md
│   └── runs/<run_id>/        per-run metrics, predictions, cards
├── src/
│   ├── collect_posts.py      corpus construction
│   ├── label_posts.py        FinBERT prelabels + gold-audit sample
│   ├── extract_tickers.py    cashtag/alias linking + ambiguity gate
│   ├── build_dataset.py      prices, β, θ, labels, temporal split, purge
│   ├── features.py           feature pipeline and torch dataset
│   ├── model.py              fusion model
│   ├── train.py              training loop
│   ├── baselines.py          B0–B6
│   ├── metrics.py            metrics, bootstrap, calibration, aggregation
│   ├── evaluate.py           evaluation, audit, model card
│   ├── audit_classifier.py   classifier-vs-manual kappa proxy
│   ├── sample_ticker_gold.py fixed 300-post ticker gold sample
│   ├── score_ticker_gold.py  ticker precision/recall audit
│   ├── score_gold_audit.py   human gold-label kappa audit
│   ├── run_experiments.py    seeds/ablation driver
│   ├── aggregate_results.py  seed mean ± std and ablation table
│   ├── walk_forward.py       train <= Y-2, val = Y-1, test = Y folds
│   └── serve.py              FastAPI service
├── tests/
├── Makefile
├── requirements.txt
└── README.md
```

---

## 9. Leakage controls

* Entry is next open strictly after the post timestamp (`created_utc < t0`).
* Forward-window purge at train/val and val/test boundaries + 10-session embargo.
* No post appears in two splits (tested).
* Aggregate features are formed from posts already published before the entry
  open; they do not use direction labels or future prices.
* `class_probs` mutual information with the direction label is reported
  (`reports/leakage_audit.json`), not assumed harmless.
* Shuffled-label AUC = 0.4969 ± 0.0153 (target ≈ 0.5).
* Test predictions are saved so the audit can be reproduced.

A residual risk remains: Reddit posts can be edited after publication, and the
Arctic Shift snapshot does not let us verify edit history. Scores/comments are
also final snapshot values, so votes cast after the entry open may leak into
`score`, `num_comments` and the aggregate features. Neither risk can be removed
from this snapshot; both are documented in the model card instead of being
claimed as solved.

---

## 10. Why the model did not win

* **Class-label noise.** Half the corpus is classifier-prelabelled; the strongest
  baseline (B2) uses the same text but is harder to overfit.
* **Market regime dominates.** B5 (market features only) reaches MCC 0.127 and
  AUC 0.599, nearly matching the full model.
* **Very short history per ticker-day.** 662 test groups with overlapping
  forward windows make the effective sample much smaller than 1,270 rows.
* **Label definition.** FLAT at 35% plus a volatility-scaled threshold leaves a
  noisy 3-class target; the model improves macro-F1 slightly but not enough.
* **Pre-registered, not tuned.** No test-set threshold search, no per-ticker
  calibration, no cherry-picked horizon. The negative result stands.

Next steps that would be legitimate: a properly human-audited class-label set,
walk-forward retraining per year, a learned attention aggregator, a longer
context encoder, and a larger/delisted ticker universe. None of these are
claimed here.

---

## 11. Ethics and data use

Reddit content is used for research only. Author names are hashed; the corpus
contains no usernames. Respect the Reddit and Arctic Shift terms of service.
Do not present model output as personalized financial advice. The economic
backtest is a sanity check on historical data, not a tradable strategy.

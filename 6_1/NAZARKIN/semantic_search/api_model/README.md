# WSB post classifier — API package

Classifies r/WallStreetBets posts into **4 classes**:

| class | meaning |
|---|---|
| `NOISE_MEME` | no meaningful financial info — joke, meme, reaction, P&L screenshot, process chatter |
| `GENUINE_OPINION` | genuine bullish/bearish/trading stance, little or no analysis |
| `FINANCIAL_CLAIM` | concrete checkable fact about a company, market, event, price, metric |
| `ANALYSIS_DD` | substantive reasoning: calculations, valuation, bull/bear case, catalysts, risks |

The catch-all fifth class from the original spec was removed on purpose. It is not
part of the model and does not appear anywhere.

## Which model this is

`model.joblib` is a **byte-identical copy of `logreg_hand/logreg_lr0.01_bal.joblib`**
— the best checkpoint from the training run. It was not refit, retrained or
re-parameterised. `smoke_test.py` verifies the md5 against the source.

**Performance: accuracy 0.7106, macro-F1 0.6264** on 1600 genuinely held-out posts
(5-fold CV: 0.7177 ± 0.0119 acc, 0.6373 ± 0.0139 macro-F1).

These numbers belong to *this* checkpoint. It was trained on the 80% split only, so
the 20% test set is real held-out data for it — `smoke_test.py` recomputes both
figures and asserts they match.

## Files

```
model.joblib        the model (TF-IDF word+char -> logistic regression)
predictor.py        inference API (import or CLI)
server.py           zero-dependency JSON HTTP server (stdlib only)
metadata.json       model card: config, metrics, selection rationale, weaknesses
smoke_test.py       30 checks; exits non-zero on any failure
models/             the other 4 checkpoints from logreg_hand/, plus training_results.json
```

## Requirements

`scikit-learn` and `joblib`. Nothing else — the server uses only the standard library.

## Library use

```python
from predictor import WSBClassifier

clf = WSBClassifier()                      # loads ./model.joblib

clf.predict_one("GME earnings tomorrow", "calls 12/31 $80")
# {'index': 0, 'label': 'GENUINE_OPINION', 'confidence': 0.77,
#  'probabilities': {'NOISE_MEME': 0.00, 'GENUINE_OPINION': 0.77,
#                    'FINANCIAL_CLAIM': 0.12, 'ANALYSIS_DD': 0.11},
#  'empty_input': False}

clf.predict([
    {"title": "SPY all time highs", "body": ""},
    ["$GME to the moon", ""],                       # [title, body] also works
    "just a string post",                          # bare string also works
])
clf.predict_labels([...])                           # labels only, no probabilities
```

Accepted input per post: `str`, `(title, body)`, or a dict with `title`/`body`
(also accepts `title`/`selftext` and `title`/`text`).

Every record contains `label`; `confidence` and `probabilities` are included unless
you pass `include_proba=False`. `confidence` is always the probability of the
returned `label`. Empty input does not crash — it returns a label with
`empty_input: True`, so a title-less post is still classified from the title.

## CLI

```bash
echo '{"posts":[{"title":"GME to the moon","body":""}]}' | python3 predictor.py -
python3 predictor.py posts.json --labels-only
```

## HTTP API

```bash
python3 server.py --host 0.0.0.0 --port 8000
```

| method | path | body | returns |
|---|---|---|---|
| GET | `/health` | — | status, version, classes |
| GET | `/classes` | — | class list |
| POST | `/predict` | `{"posts":[...]}` or a bare list | `{"predictions":[...]}` |
| POST | `/predict_one` | `{"title":...,"body":...}` or `{"text":...}` | one prediction |

```bash
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"posts":[{"title":"$GME to the moon","body":""},
                {"title":"Why do memes get more upvotes than DD","body":""}]}'
```

```json
{"predictions": [
  {"index": 0, "label": "GENUINE_OPINION", "confidence": 0.77,
   "probabilities": {"NOISE_MEME": 0.00, "GENUINE_OPINION": 0.77,
                    "FINANCIAL_CLAIM": 0.10, "ANALYSIS_DD": 0.13},
   "empty_input": false},
  {"index": 1, "label": "NOISE_MEME", "confidence": 0.87, "...": "..."}]}
```

Bad input returns `400 {"error": "..."}`, unknown paths `404`.

## Verify it works

```bash
python3 smoke_test.py     # exits non-zero on any failure
```

## Model

`SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, alpha=1e-4,
class_weight='balanced', 300 epochs)` over a `FeatureUnion` of word TF-IDF
(1-2 grams, min_df=2) and char_wb TF-IDF (3-5 grams, min_df=3). Trained on 6400
posts (80% of the 8000 hand-labelled ones).

It uses `SGDClassifier` rather than `LogisticRegression` because sklearn ≥ 1.7
removed `learning_rate`/`eta0` from `LogisticRegression` — they only ever applied
to the sag/saga solvers. `loss='log_loss'` is the same logistic objective with a
fixed step size, so `lr=0.01` is honoured literally.

`class_weight='balanced'` matters: without it the model collapses toward
`NOISE_MEME` and macro-F1 drops from 0.6264 to 0.5548.

## Per-class performance (held-out 1600)

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| NOISE_MEME | 0.781 | 0.856 | 0.817 | 926 |
| GENUINE_OPINION | 0.476 | 0.377 | **0.421** | 257 |
| FINANCIAL_CLAIM | 0.642 | 0.491 | 0.556 | 267 |
| ANALYSIS_DD | 0.659 | 0.773 | 0.712 | 150 |

## Swapping models

Pass a path — no code changes needed:

```python
WSBClassifier("models/logreg_lr0.01_raw.joblib")
```

| checkpoint | holdout macro-F1 |
|---|---|
| `model.joblib` (default, lr=0.01 bal) | **0.6264** |
| `models/logreg_lbfgs.joblib` | 0.6261 — best CV macro-F1 (0.6487) |
| `models/logreg_lr0.001_bal.joblib` | 0.6024 — undertrained at 300 epochs |
| `models/logreg_lr0.01_raw.joblib` | 0.5548 — no class weighting |
| `models/logreg_lr0.001_raw.joblib` | 0.4864 — worst |

`models/training_results.json` is the raw metrics file the selection was made from.

## Known weaknesses

- **`GENUINE_OPINION` is the weak class** (F1 0.42). An opinion with no analysis
  is genuinely hard to separate from a joke using text alone.
- **2019–2023 posts often have an empty body** in Arctic Shift, so those years are
  classified from the title alone. Expect lower quality on early-year posts.
- **Accuracy is dominated by `NOISE_MEME`** (58% of the data). Use macro-F1 or
  per-class metrics, not accuracy, to judge changes to this model.
- The model overfits: train accuracy 0.958 vs 0.711 held-out.
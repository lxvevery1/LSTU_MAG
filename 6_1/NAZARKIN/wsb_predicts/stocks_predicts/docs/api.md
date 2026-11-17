# HTTP API reference

Complete reference for the FastAPI backend in [`../src/serve.py`](../src/serve.py)
and [`../src/history_api.py`](../src/history_api.py).

The API has two groups of endpoints:

* **Live inference** — score a new WSB post × ticker pair with the trained
  fusion model (`/predict_direction`, `/predict_direction/batch`).
* **Historical data** — read-only prices, saved test predictions, run metadata
  and chart payloads (`/tickers`, `/runs`, `/prices/{ticker}`,
  `/predictions/{ticker}`, `/chart/{ticker}`). These endpoints do **not** load
  the transformer.

> **Research use only.** The model predicts a 3-class direction on a 5-session
> horizon. It is not a price forecast, a trading system or investment advice.

---

## 1. Running the server

```bash
cd stocks_predicts

# Reference checkpoint (seed 42)
make serve RUN_ID=finbert_h5_seed42 PORT=8080

# Or directly
../.venv/bin/python src/serve.py \
  --run-id finbert_h5_seed42 \
  --host 0.0.0.0 \
  --port 8080 \
  --device auto
```

Base URL: `http://localhost:8080`.

| URL | Purpose |
|---|---|
| `/docs` | Swagger UI |
| `/redoc` | ReDoc |
| `/openapi.json` | machine-readable OpenAPI schema |

### CLI options

| option | default | meaning |
|---|---|---|
| `--run-id` | latest `models/finbert_h5_seed*` | checkpoint directory name |
| `--host` | `0.0.0.0` | bind address |
| `--port` | `8080` | bind port |
| `--device` | `auto` | `auto`, `cuda`, `mps` or `cpu` |
| `--temperature` | from `model_metrics.json` | override calibration temperature |
| `--preload` | off | load the checkpoint before serving |
| `--reload` | off | accepted but ignored (model is in-process) |

### Environment variables

| variable | default | meaning |
|---|---|---|
| `WSB_RUN_DIR` | unset | checkpoint directory; set by `serve.py --run-id` |
| `WSB_DEVICE` | `auto` | torch device |
| `WSB_CORS_ORIGINS` | `*` | comma-separated browser origins |
| `WSB_RUN_ID` | `finbert_h5_seed42` | fallback for historical endpoints |

`make serve` passes `--run-id` explicitly, so it uses `finbert_h5_seed42` by
default (change with `make serve RUN_ID=...`). Running `python src/serve.py`
without `--run-id` picks the lexicographically latest checkpoint directory.

### Run selection for historical endpoints

`/predictions/{ticker}` and `/chart/{ticker}` accept an optional `run_id` query
parameter. When omitted, the backend uses:

1. the basename of `WSB_RUN_DIR` (the checkpoint selected by `serve.py`), if it
   has `model_predictions.parquet`;
2. otherwise `finbert_h5_seed42`;
3. otherwise the first available run with saved predictions.

Use `GET /runs` to discover valid ids. `run_id` must match
`[A-Za-z0-9][A-Za-z0-9._-]{0,127}`; path traversal is rejected.

---

## 2. Conventions

### Tickers

All tickers are validated against the fixed 30-symbol WSB universe:

```
AAPL TSLA NVDA MSFT AMZN GOOGL META AMD NFLX DIS BA GME AMC PLTR HOOD SOFI
RIVN LCID NIO MSTR COIN SNAP UBER PYPL INTC MU AVGO ARM SPY QQQ
```

Input is case-insensitive and a leading `$` is stripped (`$aapl` → `AAPL`).
Symbols outside the universe return HTTP 422. DIA/IWM exist in the price file
as extra benchmarks but are not part of the model universe.

### Dates and times

* Query dates are inclusive ISO dates: `YYYY-MM-DD`.
* `created_utc` in prediction requests accepts Unix seconds or ISO-8601. It
  defaults to the current UTC time.
* `t0_utc` is the next market open strictly after `created_utc`; `t0_date` is
  that session's date.
* The H=5 label uses the open of `t0` as entry and the close of the 5th session
  including entry day as exit.

### Direction and probabilities

`direction` / `prediction` is one of `DOWN`, `FLAT`, `UP`. Probabilities are
floats that sum to 1.0. The predicted class is the argmax.

The H=5 target is a beta-adjusted abnormal return against SPY:

```
a = r - beta * r_SPY
theta = max(0.005, 0.5 * sigma_20d)
UP   if a >=  theta
DOWN if a <= -theta
FLAT otherwise
```

`beta` and `sigma_20d` use only sessions before the entry open.

### Errors

Errors use FastAPI's standard shape:

```json
{"detail": "ticker 'FAKE' is not in the model universe (...)"}
```

| status | when |
|---|---|
| 200 | success |
| 404 | unknown run, run without predictions, or no price history |
| 422 | validation error, unknown ticker, invalid run id, `start > end`, empty text |
| 500 | unexpected inference failure |
| 503 | price file missing or no evaluated run with predictions |

### CORS

Browser access is allowed from `WSB_CORS_ORIGINS` (default `*`, credentials
disabled). For production, set explicit origins:

```bash
WSB_CORS_ORIGINS=https://dashboard.example.com make serve
```

---

## 3. Endpoint index

| method | path | loads model | purpose |
|---|---|---|---|
| GET | `/` | no | API index |
| GET | `/health` | no | liveness |
| GET | `/model_info` | yes, lazily | checkpoint metadata |
| POST | `/predict_direction` | yes, lazily | one live prediction |
| POST | `/predict_direction/batch` | yes, lazily | up to 100 live predictions |
| GET | `/tickers` | no | universe coverage |
| GET | `/runs` | no | evaluated runs and metrics |
| GET | `/prices/{ticker}` | no | daily OHLCV |
| GET | `/predictions/{ticker}` | no | saved test predictions |
| GET | `/chart/{ticker}` | no | candle + prediction chart payload |

---

## 4. `GET /`

Small service index, excluded from the OpenAPI schema.

**Response**

```json
{
  "name": "WSB direction model API",
  "version": "1.1.0",
  "docs": "/docs",
  "live_prediction": "POST /predict_direction",
  "history": [
    "/tickers",
    "/runs",
    "/prices/{ticker}",
    "/predictions/{ticker}",
    "/chart/{ticker}"
  ]
}
```

---

## 5. `GET /health`

Liveness check. It does **not** load the checkpoint.

**Response**

```json
{
  "status": "ok",
  "model_loaded": false,
  "run_id": null,
  "device": null
}
```

After the first `/model_info` or prediction request (or when the server was
started with `--preload`):

```json
{
  "status": "ok",
  "model_loaded": true,
  "run_id": "finbert_h5_seed42",
  "device": "cuda"
}
```

Use fast history endpoints (`/tickers`, `/prices/...`) as readiness probes if
you do not want to trigger a model load.

---

## 6. `GET /model_info`

Returns checkpoint metadata. This call loads the model on first use and may
take several seconds and significant GPU/CPU memory.

**Response**

```json
{
  "run_id": "finbert_h5_seed42",
  "device": "cuda",
  "primary_horizon": 5,
  "horizons": [1, 3, 5, 10],
  "max_len": 256,
  "n_numeric_features": 31,
  "ticker_universe": ["AAPL", "TSLA", "..."],
  "temperature_primary": 1.2117462487388042,
  "uses_class_probs": true
}
```

| field | type | meaning |
|---|---|---|
| `run_id` | string | checkpoint directory |
| `device` | string | `cuda`, `mps` or `cpu` |
| `primary_horizon` | int | always `5` for reported metrics |
| `horizons` | int[] | auxiliary heads `[1, 3, 5, 10]` |
| `max_len` | int | tokenizer truncation length |
| `n_numeric_features` | int | numeric feature width |
| `ticker_universe` | string[] | accepted tickers |
| `temperature_primary` | float | temperature scaling for H=5 |
| `uses_class_probs` | bool | stage-1 class probabilities are inputs |

---

## 7. `POST /predict_direction`

Score one post × ticker pair.

**Request body** (`application/json`)

| field | type | required | limits/default | meaning |
|---|---|---|---|---|
| `title` | string | no | ≤ 20,000 chars | Reddit title |
| `body` | string | no | ≤ 100,000 chars | Reddit selftext |
| `ticker` | string | yes | 1–10 chars | universe ticker, `$` optional |
| `created_utc` | number \| string \| null | no | default now | Unix seconds or ISO-8601 |
| `score` | int | no | default 0, ±1e6 | Reddit score |
| `num_comments` | int | no | default 0, ≤ 1e7 | Reddit comment count |
| `flair` | string | no | default `""`, ≤ 200 chars | post flair |

`title` or `body` must contain non-whitespace text.

**Example request**

```bash
curl -s localhost:8080/predict_direction \
  -H 'content-type: application/json' \
  -d '{
    "title": "AAPL calls printing",
    "body": "Buying 250C for next Friday, gamma squeeze setup.",
    "ticker": "AAPL",
    "created_utc": "2025-01-15T14:00:00Z",
    "flair": "YOLO",
    "score": 120,
    "num_comments": 14
  }' | python -m json.tool
```

**Response** (`PredictResponse`)

```json
{
  "run_id": "finbert_h5_seed42",
  "ticker": "AAPL",
  "created_utc": 1736949600,
  "t0_utc": "2025-01-15T14:30:00Z",
  "t0_date": "2025-01-15",
  "primary_horizon": 5,
  "direction": "UP",
  "confidence": 0.3818,
  "probabilities": {
    "DOWN": 0.3492,
    "FLAT": 0.2690,
    "UP": 0.3818
  },
  "calibrated": true,
  "horizons": {
    "1":  {"direction": "UP", "probabilities": {"DOWN": 0.2941, "FLAT": 0.3430, "UP": 0.3628}, "confidence": 0.3628},
    "3":  {"direction": "UP", "probabilities": {"DOWN": 0.3463, "FLAT": 0.3041, "UP": 0.3496}, "confidence": 0.3496},
    "5":  {"direction": "UP", "probabilities": {"DOWN": 0.3492, "FLAT": 0.2690, "UP": 0.3818}, "confidence": 0.3818},
    "10": {"direction": "UP", "probabilities": {"DOWN": 0.3485, "FLAT": 0.2020, "UP": 0.4494}, "confidence": 0.4494}
  },
  "post_class": {
    "NOISE_MEME": 0.0093,
    "GENUINE_OPINION": 0.0246,
    "FINANCIAL_CLAIM": 0.9567,
    "ANALYSIS_DD": 0.0095
  },
  "post_class_label": "FINANCIAL_CLAIM",
  "influence_weight": 0.6,
  "text_fragment": "AAPL calls printing Buying 250C for next Friday, gamma squeeze setup.",
  "warnings": []
}
```

| field | type | meaning |
|---|---|---|
| `run_id` | string | checkpoint used |
| `ticker` | string | normalized ticker |
| `created_utc` | int | parsed post timestamp |
| `t0_utc` | string \| null | next market open (UTC) |
| `t0_date` | string \| null | next session date |
| `primary_horizon` | int | `5` |
| `direction` | enum | primary H=5 class |
| `confidence` | float | max primary probability |
| `probabilities` | object | primary H=5 probabilities |
| `calibrated` | bool | temperature applied to primary head |
| `horizons` | object | `"1"`, `"3"`, `"5"`, `"10"` heads |
| `post_class` | object | 4-class stage-1 probabilities |
| `post_class_label` | string | argmax post class |
| `influence_weight` | float | class-based weight (loss/aggregation only, never an input) |
| `text_fragment` | string | ±200-char fragment around the ticker mention |
| `warnings` | string[] | e.g. ticker not mentioned, stale market window |

Notes:

* All four horizon entries contain `direction`, `probabilities` and
  `confidence`. Per-horizon `calibrated` and `logits` exist in the Python service
  but are filtered out of this endpoint by the response model.
* If `created_utc` is beyond the cached price history, market features fall back
  to training medians and `warnings` says so.
* Unknown ticker or empty text → HTTP 422.
* A ticker that is not explicitly mentioned in the text is still scorable; a
  warning is returned.

---

## 8. `POST /predict_direction/batch`

Score up to 100 pairs in one call.

**Request body**: JSON array of the same objects accepted by
`/predict_direction`.

```bash
curl -s localhost:8080/predict_direction/batch \
  -H 'content-type: application/json' \
  -d '[
    {"title": "TSLA calls printing", "body": "", "ticker": "TSLA"},
    {"title": "AAPL earnings", "body": "Buying calls", "ticker": "AAPL"},
    {"title": "bad ticker", "body": "", "ticker": "FAKE"}
  ]' | python -m json.tool
```

**Response**

```json
{
  "n": 3,
  "results": [
    {"run_id": "finbert_h5_seed42", "ticker": "TSLA", "direction": "UP", "...": "..."},
    {"run_id": "finbert_h5_seed42", "ticker": "AAPL", "direction": "FLAT", "...": "..."},
    {"error": "ticker 'FAKE' is not in the model universe (...)", "ticker": "FAKE"}
  ]
}
```

* More than 100 items → HTTP 422 for the whole request.
* Per-item validation errors do **not** fail the request; they appear as
  `{"error": ..., "ticker": ...}`.
* Unlike the single-item endpoint, batch items are returned without a response
  model, so each horizon also includes `calibrated` and `logits`.

---

## 9. `GET /tickers`

Ticker universe plus coverage in the saved test predictions of the default run.

**Query parameters**: none.

**Response**

```json
{
  "run_id": "finbert_h5_seed42",
  "n": 30,
  "tickers": [
    {
      "ticker": "AAPL",
      "has_prices": true,
      "test_posts": 46,
      "first_prediction_date": "2025-01-02",
      "last_prediction_date": "2026-09-18"
    }
  ]
}
```

| field | meaning |
|---|---|
| `run_id` | run used for `test_posts` and date bounds |
| `n` | universe size (30) |
| `has_prices` | ticker present in the price file |
| `test_posts` | saved test post × ticker rows for this run |
| `first_prediction_date` / `last_prediction_date` | saved prediction span, `null` if none |

Some tickers have no test posts (for example `ARM`); they are still returned with
`test_posts: 0` and `null` dates.

---

## 10. `GET /runs`

Evaluated runs with saved test predictions and headline metrics.

**Query parameters**: none.

**Response**

```json
{
  "default_run_id": "finbert_h5_seed42",
  "n": 16,
  "runs": [
    {
      "run_id": "finbert_h5_seed42",
      "created_utc": "2026-11-14T22:38:31Z",
      "test_rows": 1270,
      "test_groups": 662,
      "macro_f1": 0.4301732748260801,
      "mcc": 0.15591168005886197,
      "auc": 0.6016340141284764,
      "ece": 0.06650117276680668,
      "temperature": 1.2117462487388042,
      "max_len": 256,
      "has_predictions": true
    }
  ]
}
```

* Sorted newest first by `created_utc`.
* Metrics come from `reports/runs/<run_id>/model_metrics.json`; the calibrated
  test metrics are used when present.
* `auc` is one-vs-rest ROC-AUC; `ece` is expected calibration error.
* Includes seed and ablation runs. Missing values are `null`.
* `default_run_id` is the run new historical requests use when `run_id` is
  omitted.

---

## 11. `GET /prices/{ticker}`

Daily OHLCV history from `stocks_dataset/prices_long.csv`.

**Path parameter**

| name | type | meaning |
|---|---|---|
| `ticker` | string | universe ticker |

**Query parameters**

| name | type | default | meaning |
|---|---|---|---|
| `start` | ISO date | none | inclusive lower bound |
| `end` | ISO date | none | inclusive upper bound |
| `limit` | int 1–20,000 | none | keep the latest N candles after filtering |

```bash
curl -s 'localhost:8080/prices/AAPL?start=2026-09-14&end=2026-09-18'
```

**Response**

```json
{
  "ticker": "AAPL",
  "interval": "1d",
  "start": "2026-09-14",
  "end": "2026-09-18",
  "n": 5,
  "candles": [
    {
      "date": "2026-09-14",
      "open": 334.79,
      "high": 335.5,
      "low": 331.34,
      "close": 333.08,
      "adj_close": 333.08,
      "volume": 39269150.0
    }
  ]
}
```

* `start` / `end` are the actual first/last returned candle dates.
* `limit` is applied after date filtering and returns the **most recent** rows.
* Prices are split-adjusted but not dividend-adjusted.
* No rows for the ticker → HTTP 404; unknown ticker → HTTP 422;
  `start > end` → HTTP 422.

---

## 12. `GET /predictions/{ticker}`

Saved test-period predictions for one ticker, both post-level and daily
aggregates. No model forward pass is performed.

**Path parameter**: `ticker`.

**Query parameters**

| name | type | default | meaning |
|---|---|---|---|
| `run_id` | string | served run | checkpoint/run to read |
| `start` | ISO date | none | inclusive filter on `t0_date` |
| `end` | ISO date | none | inclusive filter on `t0_date` |

```bash
curl -s 'localhost:8080/predictions/AAPL?start=2025-01-01&end=2025-06-30' \
  | python -m json.tool
```

**Response**

```json
{
  "run_id": "finbert_h5_seed42",
  "ticker": "AAPL",
  "primary_horizon": 5,
  "start": "2026-09-01",
  "end": "2026-09-30",
  "summary": {
    "n_posts": 4,
    "n_dates": 3,
    "post_hit_rate": 0.25,
    "date_hit_rate": 0.3333333333333333,
    "always_up_baseline": 0.3333333333333333,
    "mean_confidence": 0.4385494117882291,
    "prediction_counts": {"DOWN": 2, "FLAT": 0, "UP": 1},
    "true_counts": {"DOWN": 0, "FLAT": 2, "UP": 1},
    "first_date": "2026-09-01",
    "last_date": "2026-09-18"
  },
  "daily": [
    {
      "date": "2026-09-01",
      "n_posts": 1,
      "prediction": "DOWN",
      "true_direction": "FLAT",
      "probabilities": {
        "DOWN": 0.4427,
        "FLAT": 0.2172,
        "UP": 0.3401
      },
      "confidence": 0.4427,
      "score": -0.1025,
      "abnormal_return": -0.003310886150744105,
      "hit": false
    }
  ],
  "posts": [
    {
      "post_id": "1w3xo4z",
      "t0_date": "2026-09-01",
      "created_utc": 1788226103,
      "title": "End of an era",
      "flair": "Meme",
      "class_label": "FINANCIAL_CLAIM",
      "influence_weight": 0.6,
      "prediction": "DOWN",
      "true_direction": "FLAT",
      "probabilities": {
        "DOWN": 0.4427,
        "FLAT": 0.2172,
        "UP": 0.3401
      },
      "confidence": 0.4427,
      "abnormal_return": -0.003310886150744105,
      "hit": false
    }
  ]
}
```

### Summary definitions

| field | definition |
|---|---|
| `n_posts` | post × ticker rows after filtering |
| `n_dates` | unique `t0_date` values |
| `post_hit_rate` | share of posts where `prediction == true_direction` |
| `date_hit_rate` | share of dates where the daily aggregate is correct |
| `always_up_baseline` | share of dates whose true label is `UP` |
| `mean_confidence` | mean post-level max probability |
| `prediction_counts` | daily aggregate counts by predicted class |
| `true_counts` | daily counts by true class |
| `first_date` / `last_date` | span of the returned daily rows |

### Daily aggregation

For each `t0_date` the backend averages `p_DOWN`, `p_FLAT`, `p_UP` across all
posts for that ticker/day, takes the argmax as `prediction`, and reports:

```
score = mean P(UP) - mean P(DOWN)
```

`true_direction` and `abnormal_return` are taken from the first post row for
that date (all rows on a ticker/day share the same forward label).
`confidence` is the max of the averaged probabilities.

### Field notes

* `probabilities` always has keys `DOWN`, `FLAT`, `UP` and sums to 1.
* `abnormal_return` is the beta-adjusted H=5 return; it is `null` only if the
  saved artifact contains a missing value.
* `influence_weight` is used in loss weighting and ticker-day aggregation, never
  as a model input.
* `start` / `end` echo the requested filters, or the returned span when omitted.
* A valid ticker with no predictions in the window returns HTTP 200 with empty
  `daily` / `posts` and zeroed summary (not 404).

---

## 13. `GET /chart/{ticker}`

One payload for the chart used in the demo: candles, daily predictions and the
hit-rate summary.

**Path parameter**: `ticker`.

**Query parameters**

| name | type | default | meaning |
|---|---|---|---|
| `run_id` | string | served run | run to read |
| `start` | ISO date | none | inclusive prediction filter |
| `end` | ISO date | none | inclusive prediction filter |

Default price window when dates are omitted:

* from **first prediction date − 30 calendar days**
* to **last prediction date + 10 calendar days**

If the ticker has no predictions, the full available price history is returned.
When `start`/`end` are supplied they are applied to both predictions and
candles.

```bash
curl -s 'localhost:8080/chart/AAPL?run_id=finbert_h5_seed42' \
  | python -m json.tool
```

**Response**

```json
{
  "ticker": "AAPL",
  "run_id": "finbert_h5_seed42",
  "primary_horizon": 5,
  "interval": "1d",
  "start": "2026-09-01",
  "end": "2026-09-30",
  "candles": [
    {
      "date": "2026-09-01",
      "open": 316.98,
      "high": 327.3,
      "low": 314.73,
      "close": 325.13,
      "adj_close": 325.13,
      "volume": 53167390.0
    }
  ],
  "daily": [
    {
      "date": "2026-09-01",
      "n_posts": 1,
      "prediction": "DOWN",
      "true_direction": "FLAT",
      "probabilities": {"DOWN": 0.4427, "FLAT": 0.2172, "UP": 0.3401},
      "confidence": 0.4427,
      "score": -0.1025,
      "abnormal_return": -0.003310886150744105,
      "hit": false
    }
  ],
  "summary": {
    "n_posts": 4,
    "n_dates": 3,
    "post_hit_rate": 0.25,
    "date_hit_rate": 0.3333333333333333,
    "always_up_baseline": 0.3333333333333333,
    "mean_confidence": 0.4385494117882291,
    "prediction_counts": {"DOWN": 2, "FLAT": 0, "UP": 1},
    "true_counts": {"DOWN": 0, "FLAT": 2, "UP": 1},
    "first_date": "2026-09-01",
    "last_date": "2026-09-18"
  },
  "disclaimer": "Research use only. This is a direction classifier evaluated on historical data, not a price forecast or investment advice."
}
```

* `candles` use raw split-adjusted OHLC; `daily` uses the same aggregation and
  summary definitions as `/predictions/{ticker}`.
* `start` / `end` are the actual candle range returned.
* The endpoint is intended for chart rendering; use `/predictions/{ticker}` when
  post-level rows are needed.

---

## 14. Frontend examples

### JavaScript

```js
const base = "http://localhost:8080";

const chart = await fetch(`${base}/chart/AAPL?run_id=finbert_h5_seed42`)
  .then((r) => r.json());

const live = await fetch(`${base}/predict_direction`, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    title: "AAPL calls printing",
    body: "Buying 250C for next Friday, gamma squeeze setup.",
    ticker: "AAPL",
    created_utc: "2025-01-15T14:00:00Z",
    flair: "YOLO",
  }),
}).then((r) => r.json());
```

### curl quickstart

```bash
curl -s localhost:8080/health
curl -s localhost:8080/tickers
curl -s localhost:8080/runs
curl -s 'localhost:8080/prices/AAPL?limit=30'
curl -s 'localhost:8080/predictions/AAPL?start=2025-01-01'
curl -s 'localhost:8080/chart/AAPL?run_id=finbert_h5_seed42'
```

---

## 15. Data sources and limitations

| endpoint group | source |
|---|---|
| `/predict_direction`, `/model_info` | `models/<run_id>/model.pt`, tokenizer, `reports/runs/<run_id>/model_metrics.json` |
| `/prices/{ticker}` | `stocks_dataset/prices_long.csv` |
| `/predictions/{ticker}`, `/chart/{ticker}` | `reports/runs/<run_id>/model_predictions.parquet` |
| `/runs` | `reports/runs/*/model_metrics.json` |

* No database: artifacts are read from disk and cached in process.
* The history endpoints are read-only and safe to call while the model is not
  loaded.
* The API has no authentication. Do not expose it publicly without a reverse
  proxy, rate limiting and access control.
* Historical predictions are the saved test split (2025–2026). Do not use them
  as out-of-sample evidence for any other period.
* The model’s measured quality is weak: test macro-F1 ≈ 0.43 and MCC ≈ 0.15,
  and it does not clear the pre-registered margin over the best baseline. See
  [`../reports/final_report.md`](../reports/final_report.md) and
  [`../reports/model_card.md`](../reports/model_card.md).

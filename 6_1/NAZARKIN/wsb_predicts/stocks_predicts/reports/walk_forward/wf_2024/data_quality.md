# Data quality report — WSB direction dataset

Generated: 2026-11-15T09:13:17Z

Data version: `wsb16k-wf2024`

## Corpus and linking

- Posts: **16000**
- Posts with at least one ticker: **3658** (22.9%)
- Post x ticker links: **4844**
- Valid post x ticker x horizon rows: **13664**
- Valid pairs at primary horizon H=5: **3384**

## Direction distribution by horizon

| H | UP | FLAT | DOWN | FLAT rate |
|---:|---:|---:|---:|---:|
| 1 | 651 | 2077 | 688 | 60.8% |
| 3 | 972 | 1390 | 1054 | 40.7% |
| 5 | 1165 | 1204 | 1047 | 35.2% |
| 10 | 1297 | 998 | 1121 | 29.2% |

## Split counts

| split | purged | rows |
|---|---:|---:|
| test | False | 2828 |
| train | False | 8976 |
| train | True | 48 |
| val | False | 1719 |
| val | True | 93 |

## Coverage / exclusions

- `links_total`: 4844
- `no_post`: 0
- `no_calendar`: 0
- `not_in_split`: 1270
- `invalid_prices`: 632
- `invalid_benchmark`: 0
- `purged`: 141
- `valid_by_horizon`: 1=3416, 3=3416, 5=3416, 10=3416
- `invalid_reasons`: missing ticker entry/exit price=632

## Threshold and beta

| H | beta mean | beta p05 | beta p95 | threshold mean |
|---:|---:|---:|---:|---:|
| 1 | 1.0905 | 0.4043 | 2.9317 | 0.02167 |
| 3 | 1.0905 | 0.4043 | 2.9317 | 0.02167 |
| 5 | 1.0905 | 0.4043 | 2.9317 | 0.02167 |
| 10 | 1.0905 | 0.4043 | 2.9317 | 0.02167 |

## Known limitations

- New posts are classifier-prelabelled; manual gold-audit coverage is reported separately.
- Prices are split-adjusted but not dividend-adjusted (Nasdaq source); label noise from dividends is small for 1–10 day horizons.
- Benchmark beta uses 60 prior sessions and falls back to 1.0 when fewer than 30 observations are available.
- The 10-day embargo removes training/validation rows whose forward window crosses the next split.
- Class labels for new posts are not ground truth; see `prelabel_report.json` and `gold_audit_sample.json`.
- Classifier-vs-manual agreement on its 20% holdout is κ=0.573 [0.540, 0.605] (`classifier_audit_proxy.json`); this is a proxy, not a gold audit of the 8,000 new prelabels.
- 157 posts (0.98%) share normalized text, mostly manual posts whose body is `[removed by moderator]`; post IDs are still unique.
- Ticker precision/recall on the fixed 300-post sample is not measured until humans fill `gold_tickers` (`ticker_gold_audit.json`, status blocked).

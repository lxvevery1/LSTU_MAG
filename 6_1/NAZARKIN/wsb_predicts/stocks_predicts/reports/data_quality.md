# Data quality report — WSB direction dataset

Generated: 2026-11-14T08:09:30Z

Data version: `wsb16k-v1`

## Corpus and linking

- Posts: **16000**
- Posts with at least one ticker: **3658** (22.9%)
- Post x ticker links: **4844**
- Valid post x ticker x horizon rows: **18727**
- Valid pairs at primary horizon H=5: **4625**

## Direction distribution by horizon

| H | UP | FLAT | DOWN | FLAT rate |
|---:|---:|---:|---:|---:|
| 1 | 890 | 2883 | 913 | 61.5% |
| 3 | 1380 | 1914 | 1375 | 41.0% |
| 5 | 1611 | 1644 | 1431 | 35.1% |
| 10 | 1794 | 1365 | 1527 | 29.1% |

## Split counts

| split | purged | rows |
|---|---:|---:|
| test | False | 5063 |
| train | False | 10743 |
| train | True | 93 |
| val | False | 2689 |
| val | True | 139 |

## Coverage / exclusions

- `links_total`: 4844
- `no_post`: 0
- `no_calendar`: 0
- `not_in_split`: 0
- `invalid_prices`: 634
- `invalid_benchmark`: 15
- `purged`: 232
- `valid_by_horizon`: 1=4686, 3=4669, 5=4686, 10=4686
- `invalid_reasons`: missing ticker entry/exit price=634, missing benchmark price=15

## Threshold and beta

| H | beta mean | beta p05 | beta p95 | threshold mean |
|---:|---:|---:|---:|---:|
| 1 | 1.3043 | 0.5182 | 3.1653 | 0.0201 |
| 3 | 1.3032 | 0.5174 | 3.1627 | 0.02013 |
| 5 | 1.3043 | 0.5182 | 3.1653 | 0.0201 |
| 10 | 1.3043 | 0.5182 | 3.1653 | 0.0201 |

## Known limitations

- New posts are classifier-prelabelled; manual gold-audit coverage is reported separately.
- Prices are split-adjusted but not dividend-adjusted (Nasdaq source); label noise from dividends is small for 1–10 day horizons.
- Benchmark beta uses 60 prior sessions and falls back to 1.0 when fewer than 30 observations are available.
- The 10-day embargo removes training/validation rows whose forward window crosses the next split.
- Class labels for new posts are not ground truth; see `prelabel_report.json` and `gold_audit_sample.json`.
- Classifier-vs-manual agreement on its 20% holdout is κ=0.573 [0.540, 0.605] (`classifier_audit_proxy.json`); this is a proxy, not a gold audit.
- 157 posts (0.98%) share normalized text, mostly manual posts with `[removed by moderator]` bodies; post IDs are unique.
- Ticker precision/recall on the fixed 300-post sample is not measured until humans fill `gold_tickers` (`ticker_gold_audit.json`, status blocked).

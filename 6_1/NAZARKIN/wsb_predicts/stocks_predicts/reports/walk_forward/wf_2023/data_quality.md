# Data quality report — WSB direction dataset

Generated: 2026-11-15T09:13:16Z

Data version: `wsb16k-wf2023`

## Corpus and linking

- Posts: **16000**
- Posts with at least one ticker: **3658** (22.9%)
- Post x ticker links: **4844**
- Valid post x ticker x horizon rows: **10836**
- Valid pairs at primary horizon H=5: **2675**

## Direction distribution by horizon

| H | UP | FLAT | DOWN | FLAT rate |
|---:|---:|---:|---:|---:|
| 1 | 506 | 1646 | 557 | 60.8% |
| 3 | 738 | 1126 | 845 | 41.6% |
| 5 | 889 | 986 | 834 | 36.4% |
| 10 | 1010 | 848 | 851 | 31.3% |

## Split counts

| split | purged | rows |
|---|---:|---:|
| test | False | 1812 |
| train | False | 7002 |
| train | True | 66 |
| val | False | 1908 |
| val | True | 48 |

## Coverage / exclusions

- `links_total`: 4844
- `no_post`: 0
- `no_calendar`: 0
- `not_in_split`: 1977
- `invalid_prices`: 632
- `invalid_benchmark`: 0
- `purged`: 114
- `valid_by_horizon`: 1=2709, 3=2709, 5=2709, 10=2709
- `invalid_reasons`: missing ticker entry/exit price=632

## Threshold and beta

| H | beta mean | beta p05 | beta p95 | threshold mean |
|---:|---:|---:|---:|---:|
| 1 | 0.8797 | 0.1825 | 2.609 | 0.02322 |
| 3 | 0.8797 | 0.1825 | 2.609 | 0.02322 |
| 5 | 0.8797 | 0.1825 | 2.609 | 0.02322 |
| 10 | 0.8797 | 0.1825 | 2.609 | 0.02322 |

## Known limitations

- New posts are classifier-prelabelled; manual gold-audit coverage is reported separately.
- Prices are split-adjusted but not dividend-adjusted (Nasdaq source); label noise from dividends is small for 1–10 day horizons.
- Benchmark beta uses 60 prior sessions and falls back to 1.0 when fewer than 30 observations are available.
- The 10-day embargo removes training/validation rows whose forward window crosses the next split.
- Class labels for new posts are not ground truth; see `prelabel_report.json` and `gold_audit_sample.json`.
- Classifier-vs-manual agreement on its 20% holdout is κ=0.573 [0.540, 0.605] (`classifier_audit_proxy.json`); this is a proxy, not a gold audit of the 8,000 new prelabels.
- 157 posts (0.98%) share normalized text, mostly manual posts whose body is `[removed by moderator]`; post IDs are still unique.
- Ticker precision/recall on the fixed 300-post sample is not measured until humans fill `gold_tickers` (`ticker_gold_audit.json`, status blocked).

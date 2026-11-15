# Data card — wsb16k-wf2024

## Source
- Posts: r/WallStreetBets via `arctic-shift.photon-reddit.com`; existing 8,000 manual labels plus new posts from the local Arctic Shift pool.
- Prices: Nasdaq public historical endpoint, 2018-06-01 .. 2026-11-13, split-adjusted OHLCV.
- Benchmarks: SPY (primary), QQQ (secondary).

## Unit of analysis
One row is a (post, ticker, horizon) tuple.  The primary model uses H=5 and the ticker-day aggregate is formed by grouping on (ticker, t0_date).

## Labels
- Class labels: existing 8,000 are manual; new posts are frozen-FinBERT prelabels (`classifier_prelabel_unverified`) with a review queue.
- Direction labels: forward abnormal return, beta-adjusted, threshold `max(0.005, 0.5 * sigma_20d)`.
- Entry: open of the first session strictly after `created_utc`; exit: close of the H-th session including entry day.

## Known biases
- Survivorship: the universe was chosen from currently popular/liquid WSB names; delisted tickers are not represented.
- Selection: 2,000 posts/year target; 2026 is partial (through 2026-11-14).
- Class labels for the new 8,000 are model-generated until human review is completed.
- Classifier-vs-manual agreement on its 20% holdout is κ=0.573 [0.540, 0.605] (`classifier_audit_proxy.json`); this is a proxy, not a gold audit.
- 157 posts (0.98%) share normalized text, mostly manual posts with `[removed by moderator]` bodies; post IDs are unique.
- Ticker precision/recall on the fixed 300-post sample is not measured until humans fill `gold_tickers` (`ticker_gold_audit.json`, status blocked).
- WSB user base and vocabulary drift over time; temporal splits are mandatory.
- Reddit data are observational; labels are associative, not causal.

## Reproducibility
`python src/collect_posts.py && python src/label_posts.py && python src/extract_tickers.py && python src/build_dataset.py --all-horizons`

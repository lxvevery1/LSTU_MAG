# Error analysis — ablate_no_noise_meme_seed42

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 114 | 19 | 186 |
| FLAT | 90 | 136 | 134 |
| UP | 163 | 29 | 182 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2025 | 532 | 0.3901 | 0.1007 |
| 2026 | 521 | 0.4432 | 0.1348 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1ko3qh3 | AMC | UP | DOWN | 0.585 | So who’s pumping AMC?! |
| 1nijn03 | SNAP | UP | DOWN | 0.584 | $SNAP 36K OPTIONS YOLO |
| 1om999h | TSLA | FLAT | DOWN | 0.573 | Tesla gonna hit that bull market soon! |
| 1pb6ze1 | GME | UP | DOWN | 0.562 | Can we not do a game stop to MSTR? |
| 1oloz08 | AMZN | FLAT | DOWN | 0.560 | Was going to take META profit and put into AMZN. Any suggestions? |
| 1nik47g | TSLA | FLAT | DOWN | 0.552 | TSLA YOLO |
| 1j13w39 | AMD | UP | DOWN | 0.548 | AMD is on sale !!! |
| 1w5d1gy | PLTR | FLAT | DOWN | 0.547 | KTOS is the next PLTR??? |
| 1pb6ze1 | MSTR | UP | DOWN | 0.547 | Can we not do a game stop to MSTR? |
| 1ld19fl | AMD | UP | DOWN | 0.545 | AMD YOLO |
| 1tttkrx | GOOGL | UP | DOWN | 0.544 | Trending on Google - all in ! $SPCE |
| 1pble4r | PLTR | UP | DOWN | 0.542 | Cooked by PLTR |
| 1losoa3 | TSLA | FLAT | DOWN | 0.540 | TSLA yolo |
| 1joz534 | TSLA | UP | DOWN | 0.538 | Fake Tesla Pump |
| 1ko7ft1 | TSLA | UP | DOWN | 0.535 | Elon Musk’s Tesla Robotaxi Rollout Looks Like A Disaster Waiting To Happen |
| 1kccwo4 | AMZN | FLAT | DOWN | 0.534 | Almost sold for a loss yesterday. Was gonna hold through AMZN earnings but my tummy says n |
| 1ko30wx | NFLX | UP | DOWN | 0.533 | $NFLX I’m up +$205k so far |
| 1vqu1vo | TSLA | UP | DOWN | 0.529 | Tesla robotaxis rams through bollards and keeps going |
| 1iettbr | NVDA | UP | DOWN | 0.528 | NVIDIA tanks again? Or…. |
| 1ko3oo7 | AMC | UP | DOWN | 0.524 | AMC looks good today |
| 1lot86w | TSLA | FLAT | DOWN | 0.521 | Looks like TSLA calls by tmr noon?? |
| 1pbn4jz | PLTR | UP | DOWN | 0.520 | Wonder what PLTR is gonna be like in 2026 |
| 1vqjpg7 | MU | FLAT | DOWN | 0.519 | Uhm… who of you said MU was going back to $100 again? |
| 1w4t76b | INTC | UP | DOWN | 0.519 | I opened a fortune cookie from Panda. Y they hate me |
| 1iewc5u | HOOD | UP | DOWN | 0.518 | $HOOD on solana |

## Calibration

Temperature fitted on validation: **1.255**.
ECE before calibration: **0.1104**, after: **0.0934**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.498 ± 0.016** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.1031**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **-45.76%**
- Sharpe (5 bps): **-1.02**
- Max drawdown: **-59.29%**
- SPY mean 5-day return over the same ticker-days: **0.02%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

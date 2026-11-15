# Error analysis — ablate_weighted_ce_only_seed42

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 118 | 5 | 261 |
| FLAT | 89 | 156 | 195 |
| UP | 158 | 13 | 275 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2025 | 662 | 0.4090 | 0.1386 |
| 2026 | 608 | 0.4529 | 0.1749 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1lp7mgw | QQQ | UP | FLAT | 0.540 | First week of trading |
| 1t0n8k8 | HOOD | FLAT | UP | 0.526 | Make money when I'm drunk trading then lost it all when I am actually sober |
| 1i2eahd | QQQ | DOWN | FLAT | 0.508 | Doing pretty well for myself these days |
| 1snggkv | NFLX | DOWN | UP | 0.506 | DD: Semiconductors & Shoes and Their Downstream Effects on $AAPL |
| 1jp5jjk | NVDA | UP | DOWN | 0.505 | $720 --> $4330 ty $PVH and $NVDA |
| 1wjgvu1 | AAPL | FLAT | DOWN | 0.504 | What's wrong with apple juice🧃 |
| 1tttkrx | GOOGL | UP | DOWN | 0.503 | Trending on Google - all in ! $SPCE |
| 1n5ttjq | PLTR | FLAT | UP | 0.503 | Bull thesis UAL&DAL (2026 YEAR OF THE AIRLINES) |
| 1vpklqn | MSFT | FLAT | DOWN | 0.502 | 5 year chart, good week! |
| 1r7ajhr | AMZN | UP | DOWN | 0.496 | AMZN Gainz |
| 1u7099t | TSLA | FLAT | UP | 0.496 | spaceX will be the most valuable company soon |
| 1snbnmx | NFLX | DOWN | UP | 0.495 | NFLX 3x Leverage turning out amazing |
| 1lpbzzq | HOOD | FLAT | DOWN | 0.494 | Robinhood wants to create their own WSB |
| 1o8ehfw | TSLA | FLAT | UP | 0.494 | DD - Ride that $PONY AI |
| 1jp2202 | AMZN | UP | DOWN | 0.492 | My Van caught on fire, puts on Amazon for sure… 😳 |
| 1wijzku | META | UP | DOWN | 0.491 | Razvan Paraschiv on Instagram |
| 1tft05l | AMZN | FLAT | UP | 0.490 | After 4 years.. i did it. +107K (realized) YTD. |
| 1w52eib | GME | UP | DOWN | 0.490 | GOpro es la nueva GameStop? Precio estimado, hold??? |
| 1vptir4 | MSFT | FLAT | DOWN | 0.489 | How does Microsoft own GitHub and not have the best frontier coding LLM |
| 1rhlc78 | AMZN | UP | DOWN | 0.488 | If you invested $10,000 in Amazon a year ago you have $10,023 |
| 1lco2rv | PLTR | FLAT | UP | 0.488 | Aduro Clean Technologies (NASDAQ: ADUR) DD |
| 1jp6k0k | GME | UP | DOWN | 0.487 | Spotted a Bitcoin x GameStop Billboard on my way home today in NYC, bullish times incoming |
| 1ir4qx9 | TSLA | FLAT | UP | 0.486 | Am I still a regard? |
| 1n66mdf | PLTR | FLAT | UP | 0.486 | Unique dd: PLTR is the next Tesla |
| 1metymc | SNAP | DOWN | UP | 0.485 | A rap, the market is about to snap. |

## Calibration

Temperature fitted on validation: **1.212**.
ECE before calibration: **0.0776**, after: **0.0665**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.497 ± 0.015** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.1158**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **-42.59%**
- Sharpe (5 bps): **-0.72**
- Max drawdown: **-58.58%**
- SPY mean 5-day return over the same ticker-days: **-0.01%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

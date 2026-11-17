# Error analysis — ablate_focal_loss_seed42

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 121 | 7 | 256 |
| FLAT | 94 | 156 | 190 |
| UP | 170 | 17 | 259 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2025 | 662 | 0.4074 | 0.1337 |
| 2026 | 608 | 0.4395 | 0.1441 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1lp7mgw | QQQ | UP | FLAT | 0.541 | First week of trading |
| 1t0n8k8 | HOOD | FLAT | UP | 0.528 | Make money when I'm drunk trading then lost it all when I am actually sober |
| 1wjgvu1 | AAPL | FLAT | DOWN | 0.506 | What's wrong with apple juice🧃 |
| 1jp5jjk | NVDA | UP | DOWN | 0.506 | $720 --> $4330 ty $PVH and $NVDA |
| 1snggkv | NFLX | DOWN | UP | 0.506 | DD: Semiconductors & Shoes and Their Downstream Effects on $AAPL |
| 1i2eahd | QQQ | DOWN | FLAT | 0.504 | Doing pretty well for myself these days |
| 1vpklqn | MSFT | FLAT | DOWN | 0.500 | 5 year chart, good week! |
| 1tttkrx | GOOGL | UP | DOWN | 0.498 | Trending on Google - all in ! $SPCE |
| 1n5ttjq | PLTR | FLAT | UP | 0.496 | Bull thesis UAL&DAL (2026 YEAR OF THE AIRLINES) |
| 1w52eib | GME | UP | DOWN | 0.494 | GOpro es la nueva GameStop? Precio estimado, hold??? |
| 1poehcb | TSLA | DOWN | UP | 0.494 | Rule of 7 🤯 |
| 1vptir4 | MSFT | FLAT | DOWN | 0.494 | How does Microsoft own GitHub and not have the best frontier coding LLM |
| 1lpbzzq | HOOD | FLAT | DOWN | 0.492 | Robinhood wants to create their own WSB |
| 1u7099t | TSLA | FLAT | UP | 0.491 | spaceX will be the most valuable company soon |
| 1r7ajhr | AMZN | UP | DOWN | 0.490 | AMZN Gainz |
| 1lco2rv | PLTR | FLAT | UP | 0.489 | Aduro Clean Technologies (NASDAQ: ADUR) DD |
| 1ir4qx9 | TSLA | FLAT | UP | 0.489 | Am I still a regard? |
| 1metymc | SNAP | DOWN | UP | 0.488 | A rap, the market is about to snap. |
| 1o8ehfw | TSLA | FLAT | UP | 0.488 | DD - Ride that $PONY AI |
| 1nvky45 | AVGO | FLAT | UP | 0.487 | $2MM Portfolio LFG!! -> Next Goal $5MM by 2030 |
| 1nvky45 | GOOGL | DOWN | UP | 0.486 | $2MM Portfolio LFG!! -> Next Goal $5MM by 2030 |
| 1iqg8e0 | NFLX | DOWN | UP | 0.485 | FUBO earnings play |
| 1vdgmq1 | TSLA | DOWN | UP | 0.485 | Perfect July 2026: +$57K (~190% Return) |
| 1jp6k0k | GME | UP | DOWN | 0.485 | Spotted a Bitcoin x GameStop Billboard on my way home today in NYC, bullish times incoming |
| 1n66mdf | PLTR | FLAT | UP | 0.485 | Unique dd: PLTR is the next Tesla |

## Calibration

Temperature fitted on validation: **0.906**.
ECE before calibration: **0.0714**, after: **0.0753**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.497 ± 0.015** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.1158**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **-25.40%**
- Sharpe (5 bps): **-0.21**
- Max drawdown: **-54.59%**
- SPY mean 5-day return over the same ticker-days: **-0.01%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

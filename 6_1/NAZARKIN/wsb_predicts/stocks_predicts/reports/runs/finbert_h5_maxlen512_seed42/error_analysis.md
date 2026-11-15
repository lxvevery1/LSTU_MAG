# Error analysis — finbert_h5_maxlen512_seed42

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 67 | 20 | 297 |
| FLAT | 58 | 168 | 214 |
| UP | 102 | 29 | 315 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2025 | 662 | 0.4084 | 0.1725 |
| 2026 | 608 | 0.4089 | 0.1365 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1o86msj | NIO | UP | FLAT | 0.645 | Shorting NIO using options |
| 1lp7mgw | QQQ | UP | FLAT | 0.585 | First week of trading |
| 1r62kyk | MU | UP | FLAT | 0.545 | Rampant AI Demand for Memory Is Fueling a Growing Chip Crisis |
| 1porrm2 | TSLA | DOWN | UP | 0.537 | I'm thinking of killing myself bros. Please don't let this addiction destroy your life. |
| 1metymc | SNAP | DOWN | UP | 0.527 | A rap, the market is about to snap. |
| 1jd663r | COIN | UP | FLAT | 0.514 | Is Coinbase Scamming? |
| 1kcch8w | PLTR | FLAT | UP | 0.511 | portfolio advice?? learning the ropes |
| 1i2eahd | QQQ | DOWN | FLAT | 0.511 | Doing pretty well for myself these days |
| 1ir4qx9 | TSLA | FLAT | UP | 0.503 | Am I still a regard? |
| 1jolbco | QQQ | UP | FLAT | 0.503 | Tariffs on Tech |
| 1poehcb | TSLA | DOWN | UP | 0.502 | Rule of 7 🤯 |
| 1uktk0y | MU | DOWN | FLAT | 0.501 | The price of DRAM has been falling since June. Time to sell? |
| 1porrm2 | HOOD | DOWN | UP | 0.500 | I'm thinking of killing myself bros. Please don't let this addiction destroy your life. |
| 1poehcb | MSTR | DOWN | UP | 0.497 | Rule of 7 🤯 |
| 1iez7ll | AAPL | DOWN | UP | 0.492 | $800 -> $64,000 in 1 MONTH |
| 1tft05l | AMZN | FLAT | UP | 0.491 | After 4 years.. i did it. +107K (realized) YTD. |
| 1jda55z | TSLA | FLAT | UP | 0.491 | German stock Rheinmetall🚀 |
| 1ir4qx9 | HOOD | DOWN | UP | 0.486 | Am I still a regard? |
| 1lp3rql | QQQ | UP | FLAT | 0.485 | Never give up. $25k ->$10k -> $75k |
| 1kbvnu4 | TSLA | FLAT | UP | 0.484 | Mom always said she should have got me tested |
| 1jojupp | QQQ | UP | FLAT | 0.479 | Tariff Troubles |
| 1q11t8n | TSLA | DOWN | UP | 0.475 | 2026 plan |
| 1m1ea2b | PLTR | FLAT | UP | 0.475 | us rare earth minerals business |
| 1m1akwu | PLTR | FLAT | UP | 0.475 | Cathy… oh baby, do it one more time. |
| 1pog11g | MSTR | DOWN | UP | 0.472 | Let’s Push COLABOR to $5+ — The Comeback Starts Now! 🚀 |

## Calibration

Temperature fitted on validation: **0.971**.
ECE before calibration: **0.0509**, after: **0.0522**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.499 ± 0.014** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.1158**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **-24.19%**
- Sharpe (5 bps): **-0.26**
- Max drawdown: **-45.72%**
- SPY mean 5-day return over the same ticker-days: **-0.01%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

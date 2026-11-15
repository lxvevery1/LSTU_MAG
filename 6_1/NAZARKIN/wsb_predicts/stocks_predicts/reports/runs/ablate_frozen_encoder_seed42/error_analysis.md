# Error analysis — ablate_frozen_encoder_seed42

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 38 | 26 | 320 |
| FLAT | 27 | 177 | 236 |
| UP | 47 | 48 | 351 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2025 | 662 | 0.3910 | 0.1874 |
| 2026 | 608 | 0.3995 | 0.1805 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1o86msj | NIO | UP | FLAT | 0.600 | Shorting NIO using options |
| 1lp3rql | QQQ | UP | FLAT | 0.583 | Never give up. $25k ->$10k -> $75k |
| 1jolbco | QQQ | UP | FLAT | 0.560 | Tariffs on Tech |
| 1metymc | SNAP | DOWN | UP | 0.559 | A rap, the market is about to snap. |
| 1q11t8n | TSLA | DOWN | UP | 0.555 | 2026 plan |
| 1m1akwu | PLTR | FLAT | UP | 0.554 | Cathy… oh baby, do it one more time. |
| 1j1sovm | TSLA | DOWN | UP | 0.554 | I might have woken up still drunk from last night but… |
| 1jojupp | QQQ | UP | FLAT | 0.550 | Tariff Troubles |
| 1hr0czr | AMZN | FLAT | UP | 0.548 | Amzn earnings calls |
| 1lp7mgw | QQQ | UP | FLAT | 0.547 | First week of trading |
| 1ir4qx9 | HOOD | DOWN | UP | 0.544 | Am I still a regard? |
| 1porrm2 | TSLA | DOWN | UP | 0.543 | I'm thinking of killing myself bros. Please don't let this addiction destroy your life. |
| 1q11t8n | META | DOWN | UP | 0.537 | 2026 plan |
| 1q0rk84 | QQQ | DOWN | FLAT | 0.533 | 2025 not too shabby! |
| 1j1o8nn | TSLA | DOWN | UP | 0.532 | Elon Musk predicts 1000% TSLA gain in 5 years |
| 1i2eahd | QQQ | DOWN | FLAT | 0.529 | Doing pretty well for myself these days |
| 1ir4qx9 | TSLA | FLAT | UP | 0.526 | Am I still a regard? |
| 1jda55z | TSLA | FLAT | UP | 0.525 | German stock Rheinmetall🚀 |
| 1jp5xyk | QQQ | UP | FLAT | 0.522 | With a 225% return, at the end of the day, I made a 0DET
QQQ call for $10,000 at the end o |
| 1iez7ll | AAPL | DOWN | UP | 0.521 | $800 -> $64,000 in 1 MONTH |
| 1lcu225 | PLTR | FLAT | UP | 0.519 | PLTR yolo. Sold some. |
| 1q0xeo1 | TSLA | DOWN | UP | 0.517 | Michael Burry can't be wrong! |
| 1poehcb | TSLA | DOWN | UP | 0.516 | Rule of 7 🤯 |
| 1t0n8k8 | HOOD | FLAT | UP | 0.516 | Make money when I'm drunk trading then lost it all when I am actually sober |
| 1lcvax3 | TSLA | FLAT | UP | 0.513 | Tesla Calls expiring June 19th |

## Calibration

Temperature fitted on validation: **0.885**.
ECE before calibration: **0.0463**, after: **0.0523**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.498 ± 0.015** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.1158**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **-58.91%**
- Sharpe (5 bps): **-1.65**
- Max drawdown: **-60.17%**
- SPY mean 5-day return over the same ticker-days: **-0.01%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

# Error analysis — ablate_no_class_probs_seed42

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 131 | 43 | 210 |
| FLAT | 97 | 187 | 156 |
| UP | 164 | 84 | 198 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2025 | 662 | 0.4022 | 0.1125 |
| 2026 | 608 | 0.4138 | 0.1077 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1i2eahd | QQQ | DOWN | FLAT | 0.708 | Doing pretty well for myself these days |
| 1lp7mgw | QQQ | UP | FLAT | 0.695 | First week of trading |
| 1jolbco | QQQ | UP | FLAT | 0.593 | Tariffs on Tech |
| 1jojupp | QQQ | UP | FLAT | 0.584 | Tariff Troubles |
| 1lp3rql | QQQ | UP | FLAT | 0.542 | Never give up. $25k ->$10k -> $75k |
| 1i2c9rr | QQQ | DOWN | FLAT | 0.527 | I am part genius part degenerate. A late but thoughtful 2024 reflection ($190k gain) and n |
| 1wjgvu1 | AAPL | FLAT | DOWN | 0.517 | What's wrong with apple juice🧃 |
| 1nipp88 | MU | UP | FLAT | 0.515 | 4,200% Return on Play Portfolio |
| 1pore1a | MU | UP | FLAT | 0.508 | See NMR / MFG if you want to benefit from increased yen interest rate and more |
| 1jp2202 | AMZN | UP | DOWN | 0.508 | My Van caught on fire, puts on Amazon for sure… 😳 |
| 1o86msj | NIO | UP | FLAT | 0.506 | Shorting NIO using options |
| 1jov07j | NVDA | UP | FLAT | 0.499 | March was not my month |
| 1ko3qh3 | AMC | UP | DOWN | 0.498 | So who’s pumping AMC?! |
| 1ripl1u | QQQ | UP | FLAT | 0.497 | They said "Pick your War Poison", I went for a "160k War Cocktail (OXY, STNG, QQQ, SPX) |
| 1jp6m62 | NVDA | UP | FLAT | 0.495 | What will the tariff be tomorrow? |
| 1tttkrx | GOOGL | UP | DOWN | 0.495 | Trending on Google - all in ! $SPCE |
| 1w52eib | GME | UP | DOWN | 0.494 | GOpro es la nueva GameStop? Precio estimado, hold??? |
| 1wigjpy | QQQ | UP | FLAT | 0.491 | A little inspo for you degens |
| 1uxnp8b | MU | UP | FLAT | 0.490 | You're not a True Regard until the casino itself is concerned for you |
| 1vr3q22 | TSLA | UP | DOWN | 0.490 | SpaceX and Tesla |
| 1jov07j | AVGO | UP | FLAT | 0.490 | March was not my month |
| 1hrcyvi | TSLA | FLAT | DOWN | 0.490 | Tesla Cybertruck just exploded in Las Vegas |
| 1nijn03 | SNAP | UP | DOWN | 0.490 | $SNAP 36K OPTIONS YOLO |
| 1wi8aic | INTC | UP | DOWN | 0.488 | Asked my barber for Intel YTD |
| 1vpklqn | MSFT | FLAT | DOWN | 0.487 | 5 year chart, good week! |

## Calibration

Temperature fitted on validation: **1.317**.
ECE before calibration: **0.0935**, after: **0.0812**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.498 ± 0.015** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.0324**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **-72.41%**
- Sharpe (5 bps): **-2.17**
- Max drawdown: **-69.86%**
- SPY mean 5-day return over the same ticker-days: **-0.01%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

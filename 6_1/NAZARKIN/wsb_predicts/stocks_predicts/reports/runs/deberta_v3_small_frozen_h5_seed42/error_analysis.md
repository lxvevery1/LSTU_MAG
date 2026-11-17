# Error analysis — deberta_v3_small_frozen_h5_seed42

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 353 | 25 | 6 |
| FLAT | 254 | 179 | 7 |
| UP | 361 | 68 | 17 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2025 | 662 | 0.3532 | 0.2596 |
| 2026 | 608 | 0.3772 | 0.2273 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1r71jc5 | NIO | UP | FLAT | 0.645 | Nio Touts First Adjusted Profit -- Here's What It Isn't Saying |
| 1u74pvb | NIO | UP | FLAT | 0.623 | 14407 Shares of $NIO |
| 1o86msj | NIO | UP | FLAT | 0.591 | Shorting NIO using options |
| 1ripl1u | QQQ | UP | FLAT | 0.556 | They said "Pick your War Poison", I went for a "160k War Cocktail (OXY, STNG, QQQ, SPX) |
| 1ttskqb | NIO | DOWN | FLAT | 0.556 | Yolo 13k in NIO calls july17th |
| 1wj2m89 | QQQ | UP | FLAT | 0.533 | $250k invested in an eToro QQQ CFD for 20 years, should I move to IBKR despite a huge tax  |
| 1rvkttf | QQQ | UP | FLAT | 0.516 | SpaceX aims to fast track it's Nasdaq 100 listing via proposed "Fast Entry" rule |
| 1olg6gt | NFLX | FLAT | DOWN | 0.507 | "Hollywood comedy movie": Netflix thinks a global economic meltdown is funny |
| 1nijn03 | SNAP | UP | DOWN | 0.505 | $SNAP 36K OPTIONS YOLO |
| 1uy9w5d | NFLX | UP | DOWN | 0.504 | Netflix Q2 Earnings Call Live Transcript |
| 1k0ostp | NFLX | UP | DOWN | 0.503 | Pause Netflix if holding puts. I’m in indexes. Just fyi |
| 1q0rk84 | QQQ | DOWN | FLAT | 0.502 | 2025 not too shabby! |
| 1ko30wx | NFLX | UP | DOWN | 0.498 | $NFLX I’m up +$205k so far |
| 1ukod7e | QQQ | DOWN | FLAT | 0.497 | One month of gambling makes up for 3 years of trading options like a highly regarded fella |
| 1iez7ll | NFLX | UP | DOWN | 0.495 | $800 -> $64,000 in 1 MONTH |
| 1knvsxu | NFLX | FLAT | DOWN | 0.492 | Millennials , Ackman just converted Nike stocks to Call options. Is this Bullish? |
| 1vq0mx3 | NFLX | UP | DOWN | 0.486 | Why doesn’t Netflix trade around Spotify’s valuation? |
| 1niihlg | QQQ | UP | FLAT | 0.486 | How to determine your investment portfolio? |
| 1jojupp | QQQ | UP | FLAT | 0.485 | Tariff Troubles |
| 1jolbco | QQQ | UP | FLAT | 0.484 | Tariffs on Tech |
| 1uyh8yq | NFLX | UP | DOWN | 0.484 | Why is Netflix down ~8.5% after hours despite an earnings beat? |
| 1i2c9rr | QQQ | DOWN | FLAT | 0.483 | I am part genius part degenerate. A late but thoughtful 2024 reflection ($190k gain) and n |
| 1wjgvu1 | AAPL | FLAT | DOWN | 0.482 | What's wrong with apple juice🧃 |
| 1ko3qh3 | AMC | UP | DOWN | 0.478 | So who’s pumping AMC?! |
| 1jp2202 | AMZN | UP | DOWN | 0.477 | My Van caught on fire, puts on Amazon for sure… 😳 |

## Calibration

Temperature fitted on validation: **0.804**.
ECE before calibration: **0.0497**, after: **0.0550**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.500 ± 0.012** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.1158**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **17.47%**
- Sharpe (5 bps): **0.58**
- Max drawdown: **-43.46%**
- SPY mean 5-day return over the same ticker-days: **-0.01%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

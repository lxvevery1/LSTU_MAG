# Error analysis — minilm_frozen_h5_seed42

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 79 | 23 | 282 |
| FLAT | 68 | 176 | 196 |
| UP | 117 | 52 | 277 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2025 | 662 | 0.3881 | 0.1160 |
| 2026 | 608 | 0.4222 | 0.1441 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1i2eahd | QQQ | DOWN | FLAT | 0.637 | Doing pretty well for myself these days |
| 1l1nbx7 | HOOD | UP | FLAT | 0.596 | HOOD Spy Inclusion June 6th? |
| 1ko3oo7 | AMC | UP | DOWN | 0.555 | AMC looks good today |
| 1vd6y8c | AMZN | DOWN | FLAT | 0.539 | Dip buying results vs spy |
| 1lp7mgw | QQQ | UP | FLAT | 0.526 | First week of trading |
| 1ko30wx | NFLX | UP | DOWN | 0.509 | $NFLX I’m up +$205k so far |
| 1ko3qh3 | AMC | UP | DOWN | 0.506 | So who’s pumping AMC?! |
| 1wi8aic | INTC | UP | FLAT | 0.505 | Asked my barber for Intel YTD |
| 1ripl1u | QQQ | UP | FLAT | 0.486 | They said "Pick your War Poison", I went for a "160k War Cocktail (OXY, STNG, QQQ, SPX) |
| 1jp6k0k | GME | UP | DOWN | 0.476 | Spotted a Bitcoin x GameStop Billboard on my way home today in NYC, bullish times incoming |
| 1om999h | TSLA | FLAT | DOWN | 0.475 | Tesla gonna hit that bull market soon! |
| 1qtrit2 | DIS | UP | DOWN | 0.474 | The Walt Disney Company Q1 Earnings Call Live Transcript |
| 1qtdu4q | INTC | UP | FLAT | 0.472 | Intel Regard |
| 1nijn03 | SNAP | UP | DOWN | 0.472 | $SNAP 36K OPTIONS YOLO |
| 1lp4kb5 | MSTR | UP | FLAT | 0.470 | Nice 0DTE SPY scalp for almost $6k, with scaling and exit strategy explained |
| 1ieum8h | HOOD | UP | DOWN | 0.469 | I just enabled margin on robinhood as a newb |
| 1pb6ze1 | GME | UP | DOWN | 0.466 | Can we not do a game stop to MSTR? |
| 1w4moh4 | MU | UP | FLAT | 0.466 | I built an agent that buys whatever this sub is talking about. It's down 19.2%. |
| 1tttkrx | GOOGL | UP | DOWN | 0.464 | Trending on Google - all in ! $SPCE |
| 1vck9mg | TSLA | DOWN | UP | 0.463 | SPCX: from "new space age" to a measuring stick for the AI bubble |
| 1jccksx | TSLA | FLAT | UP | 0.462 | # 🚨 TESLA IS WORTH MORE THAN 6 MAJOR AUTOMAKERS COMBINED - THE MOTHER OF ALL BUBBLE PLAYS  |
| 1k0ostp | NFLX | UP | DOWN | 0.460 | Pause Netflix if holding puts. I’m in indexes. Just fyi |
| 1tf6x2k | META | DOWN | FLAT | 0.459 | Don't wanna be part of Elon's $1.75t valuation SpaceX passive investing bank heist of SPY? |
| 1vr3q22 | TSLA | UP | DOWN | 0.459 | SpaceX and Tesla |
| 1w41k5p | GME | UP | DOWN | 0.458 | Is Go Pro the next GameStop? |

## Calibration

Temperature fitted on validation: **1.320**.
ECE before calibration: **0.0748**, after: **0.0616**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.498 ± 0.015** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.1158**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **-3.90%**
- Sharpe (5 bps): **0.22**
- Max drawdown: **-43.95%**
- SPY mean 5-day return over the same ticker-days: **-0.01%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

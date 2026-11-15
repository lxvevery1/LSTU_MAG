# Error analysis — ablate_no_influence_weight_seed42

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 27 | 42 | 315 |
| FLAT | 17 | 194 | 229 |
| UP | 35 | 70 | 341 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2025 | 662 | 0.3772 | 0.1821 |
| 2026 | 608 | 0.3880 | 0.1671 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1lp7mgw | QQQ | UP | FLAT | 0.801 | First week of trading |
| 1i2eahd | QQQ | DOWN | FLAT | 0.766 | Doing pretty well for myself these days |
| 1jp5xyk | QQQ | UP | FLAT | 0.694 | With a 225% return, at the end of the day, I made a 0DET
QQQ call for $10,000 at the end o |
| 1o86msj | NIO | UP | FLAT | 0.644 | Shorting NIO using options |
| 1lp3rql | QQQ | UP | FLAT | 0.630 | Never give up. $25k ->$10k -> $75k |
| 1q0rk84 | QQQ | DOWN | FLAT | 0.605 | 2025 not too shabby! |
| 1ripl1u | QQQ | UP | FLAT | 0.569 | They said "Pick your War Poison", I went for a "160k War Cocktail (OXY, STNG, QQQ, SPX) |
| 1jolbco | QQQ | UP | FLAT | 0.546 | Tariffs on Tech |
| 1jojupp | QQQ | UP | FLAT | 0.538 | Tariff Troubles |
| 1u7er9k | MSFT | DOWN | FLAT | 0.519 | Anyone sold SPCX Puts this morning on the 1st SPCX options day |
| 1w4moh4 | MU | UP | FLAT | 0.519 | I built an agent that buys whatever this sub is talking about. It's down 19.2%. |
| 1olwc77 | NVDA | DOWN | FLAT | 0.501 | I am financially ruined (short selling Korean fried chicken companies) |
| 1ukod7e | QQQ | DOWN | FLAT | 0.496 | One month of gambling makes up for 3 years of trading options like a highly regarded fella |
| 1qth4r2 | SOFI | DOWN | FLAT | 0.495 | Forex's a hell of a drug |
| 1knvsxu | NFLX | FLAT | UP | 0.488 | Millennials , Ackman just converted Nike stocks to Call options. Is this Bullish? |
| 1t14xov | QQQ | UP | FLAT | 0.488 | 1.8M YOLO QQQ short |
| 1lp2mvo | TSLA | DOWN | UP | 0.486 | 135 shares of TSLA |
| 1uxnp8b | MU | UP | FLAT | 0.486 | You're not a True Regard until the casino itself is concerned for you |
| 1n5ttjq | PLTR | FLAT | UP | 0.481 | Bull thesis UAL&DAL (2026 YEAR OF THE AIRLINES) |
| 1qtkgt6 | MU | DOWN | FLAT | 0.480 | 🌈🐻 $SNDK $MU - Momentum Exhaustion |
| 1metymc | SNAP | DOWN | UP | 0.480 | A rap, the market is about to snap. |
| 1m1akwu | PLTR | FLAT | UP | 0.479 | Cathy… oh baby, do it one more time. |
| 1j1o8nn | TSLA | DOWN | UP | 0.479 | Elon Musk predicts 1000% TSLA gain in 5 years |
| 1kbwm8g | TSLA | FLAT | UP | 0.478 | [WSJ Exclusive] Tesla Board Opened Search for a CEO to Succeed Elon Musk |
| 1kbx94j | TSLA | FLAT | UP | 0.478 | BREAKING: Tesla, STSLA, Board of Directors have just reportedly opened a search for a CEO  |

## Calibration

Temperature fitted on validation: **0.926**.
ECE before calibration: **0.0506**, after: **0.0492**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.498 ± 0.015** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.1158**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **-35.39%**
- Sharpe (5 bps): **-0.62**
- Max drawdown: **-49.25%**
- SPY mean 5-day return over the same ticker-days: **-0.01%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

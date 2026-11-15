# Error analysis — finbert_h5_seed43

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 87 | 35 | 262 |
| FLAT | 68 | 179 | 193 |
| UP | 112 | 59 | 275 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2025 | 662 | 0.3922 | 0.1120 |
| 2026 | 608 | 0.4339 | 0.1687 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1vr5jz2 | SOFI | UP | FLAT | 0.713 | How deep do you guys go edging? |
| 1qth4r2 | SOFI | DOWN | FLAT | 0.641 | Forex's a hell of a drug |
| 1l0vzvu | SOFI | UP | FLAT | 0.633 | How to force yourself to stop gambling |
| 1i2eahd | QQQ | DOWN | FLAT | 0.615 | Doing pretty well for myself these days |
| 1jov07j | AVGO | UP | FLAT | 0.596 | March was not my month |
| 1lp4kb5 | MSTR | UP | FLAT | 0.590 | Nice 0DTE SPY scalp for almost $6k, with scaling and exit strategy explained |
| 1vcrv3s | SOFI | UP | FLAT | 0.565 | I’m dumb and broke |
| 1lp7mgw | QQQ | UP | FLAT | 0.555 | First week of trading |
| 1m1i65z | TSLA | FLAT | UP | 0.548 | Thoughts on deckers outdoors |
| 1o8ehfw | TSLA | FLAT | UP | 0.537 | DD - Ride that $PONY AI |
| 1k0fjpd | TSLA | FLAT | UP | 0.537 | Is puts on big tech this earnings season too easy? |
| 1i2rykc | SOFI | UP | FLAT | 0.536 | After my limit order executed sofi showed that i still held this plummetting stock. After  |
| 1r6lrs0 | AVGO | UP | FLAT | 0.536 | I'm in love with Nancy Pelosi |
| 1l0okjz | AVGO | DOWN | FLAT | 0.533 | What to Expect in the Markets This Week: June 2–6, 2025 |
| 1vqxn64 | AMZN | FLAT | UP | 0.529 | How to play the Unitree IPO being 8000x oversubscribed ($CCXI) |
| 1t0ump4 | GOOGL | FLAT | UP | 0.527 | $SOUN DD:The Voice AI Casino Play That Might Print or Send You Back to Wendy’s Dumpster. |
| 1qt2tox | GOOGL | DOWN | UP | 0.526 | Time to revive Netflix |
| 1t0xa90 | GOOGL | FLAT | UP | 0.521 | $SOUN DD: Big Potential or Flop There’s No In-between |
| 1t12rp2 | UBER | DOWN | UP | 0.518 | Rivian’s Real Bet: R2 Buys Time for the Autonomy Platform |
| 1jccksx | TSLA | FLAT | UP | 0.518 | # 🚨 TESLA IS WORTH MORE THAN 6 MAJOR AUTOMAKERS COMBINED - THE MOTHER OF ALL BUBBLE PLAYS  |
| 1t0ump4 | AMZN | DOWN | UP | 0.517 | $SOUN DD:The Voice AI Casino Play That Might Print or Send You Back to Wendy’s Dumpster. |
| 1tfjsvx | TSLA | FLAT | UP | 0.514 | SpaceX will crash the market and cause a lost decade or two |
| 1whwf4i | UBER | DOWN | UP | 0.513 | AUR is worth $13B. KDK is worth $750M. Both run driverless semis, and KDK targets highway  |
| 1u7g734 | TSLA | DOWN | UP | 0.513 | All the regarded SPCX doomers don’t remember TSLA |
| 1smoqu1 | TSLA | DOWN | UP | 0.510 | What's next for markets |

## Calibration

Temperature fitted on validation: **1.321**.
ECE before calibration: **0.0915**, after: **0.0762**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.497 ± 0.013** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.1158**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **-18.17%**
- Sharpe (5 bps): **0.00**
- Max drawdown: **-50.42%**
- SPY mean 5-day return over the same ticker-days: **-0.01%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

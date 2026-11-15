# Error analysis — ablate_text_only_seed42

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 93 | 38 | 253 |
| FLAT | 64 | 186 | 190 |
| UP | 125 | 56 | 265 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2025 | 662 | 0.3916 | 0.1084 |
| 2026 | 608 | 0.4475 | 0.1809 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1lp7mgw | QQQ | UP | FLAT | 0.744 | First week of trading |
| 1i2eahd | QQQ | DOWN | FLAT | 0.592 | Doing pretty well for myself these days |
| 1lp3rql | QQQ | UP | FLAT | 0.574 | Never give up. $25k ->$10k -> $75k |
| 1ripl1u | QQQ | UP | FLAT | 0.562 | They said "Pick your War Poison", I went for a "160k War Cocktail (OXY, STNG, QQQ, SPX) |
| 1o86msj | NIO | UP | FLAT | 0.561 | Shorting NIO using options |
| 1jolbco | QQQ | UP | FLAT | 0.550 | Tariffs on Tech |
| 1jojupp | QQQ | UP | FLAT | 0.538 | Tariff Troubles |
| 1jp5xyk | QQQ | UP | FLAT | 0.531 | With a 225% return, at the end of the day, I made a 0DET
QQQ call for $10,000 at the end o |
| 1n5ttjq | PLTR | FLAT | UP | 0.521 | Bull thesis UAL&DAL (2026 YEAR OF THE AIRLINES) |
| 1w52eib | GME | UP | DOWN | 0.520 | GOpro es la nueva GameStop? Precio estimado, hold??? |
| 1rhlc78 | AMZN | UP | DOWN | 0.519 | If you invested $10,000 in Amazon a year ago you have $10,023 |
| 1nirmcg | PLTR | UP | DOWN | 0.513 | Redemption 7 years later |
| 1l1nbx7 | HOOD | UP | FLAT | 0.513 | HOOD Spy Inclusion June 6th? |
| 1sa11d2 | SNAP | DOWN | UP | 0.511 | SNAP is going to get to the moon faster than Artemis II. Up 30 % in three days. Expect 100 |
| 1uy9w5d | NFLX | UP | DOWN | 0.509 | Netflix Q2 Earnings Call Live Transcript |
| 1wizfxm | MSTR | UP | DOWN | 0.507 | My Sit and chill strategy - Secret Sauce Portfolio |
| 1k0fjpd | TSLA | FLAT | UP | 0.507 | Is puts on big tech this earnings season too easy? |
| 1snbnmx | NFLX | DOWN | UP | 0.506 | NFLX 3x Leverage turning out amazing |
| 1jp2202 | AMZN | UP | DOWN | 0.505 | My Van caught on fire, puts on Amazon for sure… 😳 |
| 1ifete1 | NVDA | UP | DOWN | 0.504 | New Nvidia FUD dropped |
| 1m1i65z | TSLA | FLAT | UP | 0.503 | Thoughts on deckers outdoors |
| 1r7ajhr | AMZN | UP | DOWN | 0.501 | AMZN Gainz |
| 1metymc | SNAP | DOWN | UP | 0.499 | A rap, the market is about to snap. |
| 1k0fjpd | AMZN | FLAT | UP | 0.499 | Is puts on big tech this earnings season too easy? |
| 1o8ehfw | TSLA | FLAT | UP | 0.498 | DD - Ride that $PONY AI |

## Calibration

Temperature fitted on validation: **1.086**.
ECE before calibration: **0.0625**, after: **0.0579**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.499 ± 0.014** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.1158**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **3.81%**
- Sharpe (5 bps): **0.38**
- Max drawdown: **-49.39%**
- SPY mean 5-day return over the same ticker-days: **-0.01%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

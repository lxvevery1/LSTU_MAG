# Error analysis — ablate_no_aux_loss_seed42

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 2 | 0 | 382 |
| FLAT | 0 | 134 | 306 |
| UP | 6 | 3 | 437 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2025 | 662 | 0.3374 | 0.2731 |
| 2026 | 608 | 0.3490 | 0.2712 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1snbnmx | NFLX | DOWN | UP | 0.504 | NFLX 3x Leverage turning out amazing |
| 1qt2tox | NFLX | DOWN | UP | 0.500 | Time to revive Netflix |
| 1sa11d2 | SNAP | DOWN | UP | 0.497 | SNAP is going to get to the moon faster than Artemis II. Up 30 % in three days. Expect 100 |
| 1snggkv | NFLX | DOWN | UP | 0.496 | DD: Semiconductors & Shoes and Their Downstream Effects on $AAPL |
| 1n5ttjq | PLTR | FLAT | UP | 0.495 | Bull thesis UAL&DAL (2026 YEAR OF THE AIRLINES) |
| 1o8ehfw | TSLA | FLAT | UP | 0.494 | DD - Ride that $PONY AI |
| 1nij58k | NFLX | FLAT | UP | 0.494 | PART 2: The Larry Play - Paramount Skydance $PSKY |
| 1tfjsvx | TSLA | FLAT | UP | 0.490 | SpaceX will crash the market and cause a lost decade or two |
| 1n66mdf | PLTR | FLAT | UP | 0.490 | Unique dd: PLTR is the next Tesla |
| 1k0fjpd | AMZN | FLAT | UP | 0.490 | Is puts on big tech this earnings season too easy? |
| 1k07ynb | TSLA | FLAT | UP | 0.489 | CoreCivic 10 bagger |
| 1qt2tox | GOOGL | DOWN | UP | 0.489 | Time to revive Netflix |
| 1k0fjpd | AAPL | FLAT | UP | 0.488 | Is puts on big tech this earnings season too easy? |
| 1u7g734 | TSLA | DOWN | UP | 0.487 | All the regarded SPCX doomers don’t remember TSLA |
| 1hr0czr | AMZN | FLAT | UP | 0.487 | Amzn earnings calls |
| 1t0ump4 | AMZN | DOWN | UP | 0.486 | $SOUN DD:The Voice AI Casino Play That Might Print or Send You Back to Wendy’s Dumpster. |
| 1poctsi | TSLA | DOWN | UP | 0.486 | Tesla All-Time-High |
| 1lcu225 | PLTR | FLAT | UP | 0.486 | PLTR yolo. Sold some. |
| 1lco2rv | PLTR | FLAT | UP | 0.486 | Aduro Clean Technologies (NASDAQ: ADUR) DD |
| 1jccksx | MSTR | FLAT | UP | 0.486 | # 🚨 TESLA IS WORTH MORE THAN 6 MAJOR AUTOMAKERS COMBINED - THE MOTHER OF ALL BUBBLE PLAYS  |
| 1iey2cd | AMD | DOWN | UP | 0.485 | Intel Stock |
| 1qt2bfd | AMD | DOWN | UP | 0.485 | How Microsft could win the AI race |
| 1t0n8k8 | HOOD | FLAT | UP | 0.485 | Make money when I'm drunk trading then lost it all when I am actually sober |
| 1t12rp2 | UBER | DOWN | UP | 0.484 | Rivian’s Real Bet: R2 Buys Time for the Autonomy Platform |
| 1jccksx | TSLA | FLAT | UP | 0.484 | # 🚨 TESLA IS WORTH MORE THAN 6 MAJOR AUTOMAKERS COMBINED - THE MOTHER OF ALL BUBBLE PLAYS  |

## Calibration

Temperature fitted on validation: **2.021**.
ECE before calibration: **0.0881**, after: **0.0153**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.500 ± 0.016** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.1158**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **-9.19%**
- Sharpe (5 bps): **0.07**
- Max drawdown: **-41.05%**
- SPY mean 5-day return over the same ticker-days: **-0.01%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

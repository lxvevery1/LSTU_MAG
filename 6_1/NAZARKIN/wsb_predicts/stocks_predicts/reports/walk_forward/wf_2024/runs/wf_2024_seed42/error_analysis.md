# Error analysis — wf_2024_seed42

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 39 | 12 | 162 |
| FLAT | 37 | 72 | 109 |
| UP | 47 | 8 | 221 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2024 | 707 | 0.4241 | 0.1893 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1ehjuzj | INTC | DOWN | FLAT | 0.606 | I bought $700k worth of Intel stock today |
| 1eh3b2s | QQQ | UP | FLAT | 0.499 | Market direction |
| 1dh4umc | NFLX | FLAT | UP | 0.498 | My Grandma Just Opened an NVIDIA Strangle and Now She's More Hardcore Than Me |
| 1arzd19 | AMD | DOWN | UP | 0.496 | While everyone is earning bags of money on SMCI |
| 1ehjuzj | AMD | UP | FLAT | 0.493 | I bought $700k worth of Intel stock today |
| 18wlv24 | NFLX | FLAT | UP | 0.493 | NFLX bullish on earnings |
| 1ct7nl8 | NVDA | FLAT | UP | 0.486 | NVDA earnings next week |
| 1bt5d4a | GME | DOWN | UP | 0.481 | Ride the Reddit Rocket! Unpacking RDDT’s Wild Stock Journey |
| 1bgbnvh | AAPL | DOWN | UP | 0.481 | Think again about them leggings LULU earnings |
| 1ct15z3 | AMC | FLAT | DOWN | 0.479 | Boys, buy amc so it goes back up. Resting in the red |
| 1eh48h8 | SNAP | DOWN | UP | 0.478 | DD on snapchat |
| 1arvk79 | ARM | FLAT | UP | 0.477 | FOMC,JoblessClaimsCOIN, MSTR, XOM, DASH(is good here on dip), and any chip related(ARM/SMC |
| 1as06t7 | TSLA | DOWN | UP | 0.475 | Sora means NVDA to 2000 |
| 1g52bzh | AMZN | FLAT | DOWN | 0.475 | Amazon to the Moon!🚀🚀🚀💎💎💎 |
| 1g526j3 | AMZN | FLAT | UP | 0.470 | UPS Q4 Grim Reaper mode |
| 1ag79f6 | HOOD | DOWN | UP | 0.468 | G4meSt0p except it's DeFi so they can't Robinhood us |
| 1ehhvv4 | MSFT | DOWN | UP | 0.468 | LUMN. The future is now bright! |
| 1dh4umc | MSTR | DOWN | UP | 0.468 | My Grandma Just Opened an NVIDIA Strangle and Now She's More Hardcore Than Me |
| 1d5tioi | TSLA | DOWN | UP | 0.467 | Intuition is the best DD. |
| 1dgwkjc | AAPL | DOWN | UP | 0.467 | NOK NOK. You there? |
| 1bgadkk | NVDA | FLAT | UP | 0.466 | Eyes on MU |
| 1gsmoac | RIVN | FLAT | UP | 0.464 | Why $GM General Motors is set to fail |
| 1ct5jw8 | AAPL | FLAT | UP | 0.464 | RIVIAN IS GOING TO THE MOON 🚀🚀🚀 - DD FOR ALL YOU GOOCHERS |
| 1hfklvx | NFLX | FLAT | UP | 0.464 | Looking to marry someone with $1M+ in losses for tax sympathy and split the emotional dama |
| 1c583zf | TSLA | DOWN | UP | 0.463 | Tesla to 150 140 and 100 in next 2 weeks |

## Calibration

Temperature fitted on validation: **1.118**.
ECE before calibration: **0.0205**, after: **0.0275**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.499 ± 0.019** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.0930**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **116.85%**
- Sharpe (5 bps): **1.57**
- Max drawdown: **-20.09%**
- SPY mean 5-day return over the same ticker-days: **0.52%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

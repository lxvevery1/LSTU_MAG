# Error analysis — wf_2023_seed42

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 74 | 8 | 46 |
| FLAT | 57 | 59 | 46 |
| UP | 91 | 9 | 63 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2023 | 453 | 0.4383 | 0.1740 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 167dxpe | NVDA | DOWN | FLAT | 0.693 | One Month of Gain with 0DTE |
| 16l2qx7 | GME | FLAT | DOWN | 0.555 | GME 🚀 let’s do it for the 💎🙌 #dumbmoney |
| 17wt2yv | TSLA | FLAT | DOWN | 0.541 | Tesla bears |
| 179o32g | GME | FLAT | DOWN | 0.540 | Anybody else’s chart forever fucked up from GME in 2019? |
| 16jvoyc | GME | FLAT | DOWN | 0.539 | I’m boring… 🥱 (and poor) another GameStop? |
| 167aaxj | DIS | UP | DOWN | 0.538 | Disney, I like it at this price |
| 15samha | QQQ | UP | FLAT | 0.530 | NAAIM hit technical support, stocks go up only from here |
| 15sm787 | HOOD | UP | DOWN | 0.529 | . Fucking robinhood joining in now we still gonna 🚀🚀🚀 |
| 10dhxwe | GME | UP | DOWN | 0.525 | Game stop to the moon 🤚💎💎💎✋ |
| 10d02zf | GME | UP | DOWN | 0.524 | GME bought me a Porsche, AMC bought me a AP I'm hoping BBBY gon buy me a mansion 🤣🚀 |
| 167dxpe | TSLA | UP | FLAT | 0.521 | One Month of Gain with 0DTE |
| 188jc8z | GME | FLAT | DOWN | 0.521 | soooo… is $GME still banned here? |
| 15svlt5 | TSLA | UP | DOWN | 0.520 | Hey TSLA, I didn't hear no bell |
| 10dhvsx | GME | UP | DOWN | 0.516 | Game stop to se Moon 🤚💎💎💎✋ |
| 179euak | HOOD | UP | DOWN | 0.513 | Tesla earnings yolo |
| 10d18m7 | GME | UP | DOWN | 0.512 | HOW much did lantern_fan made from GME? |
| 11exdzh | NIO | DOWN | FLAT | 0.511 | China’s factory output smashes forecasts with decade-high growth: PMI far exceeded forecas |
| 10d4kvs | GME | UP | DOWN | 0.510 | When's the next GameStop result happening? |
| 16xnssh | TSLA | UP | DOWN | 0.509 | Put TSLA ? |
| 10dhzq3 | GME | UP | DOWN | 0.508 | Game stop to the moon wir Treiben die Aktie wider hoch 🤚💎💎✋ |
| 16x8xme | UBER | FLAT | DOWN | 0.506 | Why Is nobody talking about Uber? |
| 16ki0dp | TSLA | FLAT | UP | 0.504 | Mr. B and tesla |
| 10d172q | GME | UP | DOWN | 0.501 | How much did u/lantern_fan made from GME? |
| 188f9ey | NVDA | UP | DOWN | 0.500 | $NVDA going back to 400? |
| 179ot4f | NFLX | UP | DOWN | 0.493 | NFLX Earnings Play |

## Calibration

Temperature fitted on validation: **1.170**.
ECE before calibration: **0.0877**, after: **0.0822**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.494 ± 0.017** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.1208**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **-24.03%**
- Sharpe (5 bps): **-0.19**
- Max drawdown: **-49.06%**
- SPY mean 5-day return over the same ticker-days: **-0.07%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

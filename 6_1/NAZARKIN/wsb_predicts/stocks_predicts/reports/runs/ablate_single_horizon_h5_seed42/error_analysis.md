# Error analysis — ablate_single_horizon_h5_seed42

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 215 | 4 | 165 |
| FLAT | 167 | 151 | 122 |
| UP | 277 | 9 | 160 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2025 | 662 | 0.4203 | 0.1705 |
| 2026 | 608 | 0.4298 | 0.1095 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1t17fsb | AMD | UP | DOWN | 0.579 | Am I doing this retard thing right? |
| 1pble4r | PLTR | UP | DOWN | 0.573 | Cooked by PLTR |
| 1lp2bxe | AAPL | FLAT | DOWN | 0.568 | Make it make sense |
| 1t17fsb | INTC | UP | DOWN | 0.563 | Am I doing this retard thing right? |
| 1vpklqn | MSFT | FLAT | DOWN | 0.560 | 5 year chart, good week! |
| 1l125hv | NVDA | UP | DOWN | 0.559 | What is the next nvidia? Thoughts? |
| 1jp2202 | AMZN | UP | DOWN | 0.558 | My Van caught on fire, puts on Amazon for sure… 😳 |
| 1w52eib | GME | UP | DOWN | 0.555 | GOpro es la nueva GameStop? Precio estimado, hold??? |
| 1vr3q22 | TSLA | UP | DOWN | 0.555 | SpaceX and Tesla |
| 1hrgdya | HOOD | UP | DOWN | 0.550 | Worth the 60$ a year- robinhood gild new year gift - Bitcoin |
| 1tttkrx | GOOGL | UP | DOWN | 0.548 | Trending on Google - all in ! $SPCE |
| 1lovksf | TSLA | FLAT | DOWN | 0.548 | Round 2... Fight |
| 1ieum8h | HOOD | UP | DOWN | 0.547 | I just enabled margin on robinhood as a newb |
| 1w49i51 | MSTR | UP | DOWN | 0.544 | my trading strategy that made me 32k this month |
| 1k0acyx | GOOGL | FLAT | DOWN | 0.544 | Hey guys I got the best insider trading advice by just searching google XD |
| 1jp12zi | MSTR | UP | DOWN | 0.542 | Quick couple day flip with MSTR |
| 1wi8aic | INTC | UP | DOWN | 0.539 | Asked my barber for Intel YTD |
| 1nirmcg | PLTR | UP | DOWN | 0.539 | Redemption 7 years later |
| 1vddj0g | HOOD | FLAT | DOWN | 0.537 | The king of block trades has done it again. Just like me: I gained and lost $5 on robinhoo |
| 1ir7rgj | AMD | FLAT | DOWN | 0.537 | Can the perfect storm happen again? |
| 1wj80wm | NVDA | FLAT | DOWN | 0.536 | NVDA Hail Mary |
| 1pbpke9 | MSTR | UP | DOWN | 0.536 | Is this the bottom? $MSTR |
| 1lcnbu6 | AAPL | UP | DOWN | 0.535 | Can you buy onlyfans share? |
| 1meicdw | HOOD | UP | DOWN | 0.534 | First time on Robinhood 🚀 $FIG |
| 1wizfxm | MSTR | UP | DOWN | 0.534 | My Sit and chill strategy - Secret Sauce Portfolio |

## Calibration

Temperature fitted on validation: **1.436**.
ECE before calibration: **0.1064**, after: **0.0782**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.499 ± 0.014** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.1158**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **-64.24%**
- Sharpe (5 bps): **-1.31**
- Max drawdown: **-64.79%**
- SPY mean 5-day return over the same ticker-days: **-0.01%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

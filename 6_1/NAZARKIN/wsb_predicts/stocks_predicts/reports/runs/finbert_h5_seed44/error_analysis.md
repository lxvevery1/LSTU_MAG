# Error analysis — finbert_h5_seed44

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 182 | 38 | 164 |
| FLAT | 126 | 196 | 118 |
| UP | 179 | 91 | 176 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2025 | 662 | 0.4188 | 0.1461 |
| 2026 | 608 | 0.4605 | 0.1727 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1lp7mgw | QQQ | UP | FLAT | 0.586 | First week of trading |
| 1jolbco | QQQ | UP | FLAT | 0.579 | Tariffs on Tech |
| 1jojupp | QQQ | UP | FLAT | 0.578 | Tariff Troubles |
| 1k09a4i | INTC | UP | FLAT | 0.572 | INTC bag holder from grandmom's inheritance called into Cramer |
| 1jolbco | AAPL | DOWN | FLAT | 0.568 | Tariffs on Tech |
| 1lp3rql | QQQ | UP | FLAT | 0.535 | Never give up. $25k ->$10k -> $75k |
| 1s9mcsp | INTC | UP | FLAT | 0.524 | YOLO: $100K INTC Short for $300 |
| 1w52eib | GME | UP | DOWN | 0.523 | GOpro es la nueva GameStop? Precio estimado, hold??? |
| 1kbyyn5 | AAPL | DOWN | FLAT | 0.517 | Apple v Epic: If 20% of purchases go 0 commish off app, 2.9 bil quarterly/11.6 bil annual  |
| 1jccra7 | AAPL | UP | FLAT | 0.517 | My Setup for AAPL Put next week |
| 1i2eahd | QQQ | DOWN | FLAT | 0.514 | Doing pretty well for myself these days |
| 1jojupp | AAPL | DOWN | FLAT | 0.511 | Tariff Troubles |
| 1kcbrxs | AAPL | DOWN | FLAT | 0.511 | Follow up on the halfchub yolo. |
| 1nvd4t5 | INTC | UP | FLAT | 0.511 | $INTC $1.7m LOSS |
| 1jp5jjk | NVDA | UP | DOWN | 0.501 | $720 --> $4330 ty $PVH and $NVDA |
| 1w4dte8 | GME | UP | DOWN | 0.501 | Camera company and YouTuber |
| 1ripl1u | QQQ | UP | FLAT | 0.501 | They said "Pick your War Poison", I went for a "160k War Cocktail (OXY, STNG, QQQ, SPX) |
| 1nvky45 | NVDA | FLAT | UP | 0.499 | $2MM Portfolio LFG!! -> Next Goal $5MM by 2030 |
| 1iewfhc | INTC | DOWN | FLAT | 0.497 | 2/4 Advanced Money Destroyer ER |
| 1u78b5y | AAPL | UP | FLAT | 0.495 | ZEPP / Amazfit: tiny China wearable stock, below book, possible Garmin-lite turnaround, pr |
| 1j1q6js | NVDA | FLAT | DOWN | 0.494 | Stock Market Insights: Walmart's Strong Performance, Nvidia's Growth, and More! |
| 1lcnbu6 | NVDA | FLAT | DOWN | 0.494 | Can you buy onlyfans share? |
| 1k0ostp | NFLX | UP | DOWN | 0.492 | Pause Netflix if holding puts. I’m in indexes. Just fyi |
| 1q1aqe9 | NVDA | DOWN | UP | 0.492 | RKLB made me a millionaire |
| 1l1nbx7 | HOOD | UP | FLAT | 0.492 | HOOD Spy Inclusion June 6th? |

## Calibration

Temperature fitted on validation: **1.408**.
ECE before calibration: **0.0744**, after: **0.0590**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.500 ± 0.016** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.1158**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **28.49%**
- Sharpe (5 bps): **0.74**
- Max drawdown: **-37.29%**
- SPY mean 5-day return over the same ticker-days: **-0.01%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

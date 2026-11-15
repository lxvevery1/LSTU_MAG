# Error analysis — ablate_no_ticker_fragment_seed42

## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)

| true \ pred | DOWN | FLAT | UP |
|---|---:|---:|---:|
| DOWN | 70 | 34 | 280 |
| FLAT | 49 | 183 | 208 |
| UP | 90 | 59 | 297 |

## Per-year macro-F1

| year | n | macro-F1 | MCC |
|---|---:|---:|---:|
| 2025 | 662 | 0.3933 | 0.1373 |
| 2026 | 608 | 0.4276 | 0.1659 |

## Most confident errors

| post | ticker | true | pred | confidence | title |
|---|---|---|---|---:|---|
| 1o86msj | NIO | UP | FLAT | 0.833 | Shorting NIO using options |
| 1lp3rql | QQQ | UP | FLAT | 0.804 | Never give up. $25k ->$10k -> $75k |
| 1ukod7e | QQQ | DOWN | FLAT | 0.687 | One month of gambling makes up for 3 years of trading options like a highly regarded fella |
| 1lp7mgw | QQQ | UP | FLAT | 0.678 | First week of trading |
| 1jd663r | COIN | UP | FLAT | 0.643 | Is Coinbase Scamming? |
| 1om33fx | MSFT | DOWN | FLAT | 0.586 | Trader puts his college tuition on MSFT calls and loses it all |
| 1n6r19b | QQQ | UP | FLAT | 0.574 | 230% QQQ 0DTE call |
| 1uxnp8b | MU | UP | FLAT | 0.570 | You're not a True Regard until the casino itself is concerned for you |
| 1uk692a | MU | DOWN | FLAT | 0.565 | MU changed this broke boy’s life |
| 1metymc | SNAP | DOWN | UP | 0.560 | A rap, the market is about to snap. |
| 1povuui | MU | UP | FLAT | 0.559 | Micron Gains |
| 1t14xov | QQQ | UP | FLAT | 0.555 | 1.8M YOLO QQQ short |
| 1i2eahd | QQQ | DOWN | FLAT | 0.554 | Doing pretty well for myself these days |
| 1om3270 | MSFT | DOWN | FLAT | 0.548 | Trader puts his college tuition on MSFT calls and loses everyone |
| 1q11t8n | TSLA | DOWN | UP | 0.542 | 2026 plan |
| 1snggkv | NFLX | DOWN | UP | 0.530 | DD: Semiconductors & Shoes and Their Downstream Effects on $AAPL |
| 1u7j4ll | SNAP | DOWN | UP | 0.525 | SNAP unveils 2,195 AR glasses |
| 1ld3vev | PLTR | FLAT | UP | 0.523 | ACHR YOLO |
| 1ir4qx9 | TSLA | FLAT | UP | 0.516 | Am I still a regard? |
| 1ifgyhh | TSLA | DOWN | UP | 0.512 | Cathie Wood Swoops In To Buy AMD Shares Amid DeepSeek Tech Upheaval, Dumps Tesla Stock Wor |
| 1qth4r2 | SOFI | DOWN | FLAT | 0.505 | Forex's a hell of a drug |
| 1ifgyhh | AMD | DOWN | UP | 0.505 | Cathie Wood Swoops In To Buy AMD Shares Amid DeepSeek Tech Upheaval, Dumps Tesla Stock Wor |
| 1q0xeo1 | TSLA | DOWN | UP | 0.503 | Michael Burry can't be wrong! |
| 1sneszs | NFLX | DOWN | UP | 0.502 | Netflix just plummeted 8% immediately upon market close today. Time to buy! |
| 1kcch8w | PLTR | FLAT | UP | 0.502 | portfolio advice?? learning the ropes |

## Calibration

Temperature fitted on validation: **0.827**.
ECE before calibration: **0.0471**, after: **0.0644**.

## Leakage audit

- Posts with `created_utc >= t0`: **0**
- Post IDs shared between train and test: **0**
- Shuffled-label AUC: **0.500 ± 0.014** (target ≈ 0.5)
- Sum of mutual information between class probabilities and direction: **0.1158**

## Economic sanity check (ticker-day, weekly rebalance)

- Long-short annualized return (5 bps): **-57.75%**
- Sharpe (5 bps): **-1.47**
- Max drawdown: **-56.96%**
- SPY mean 5-day return over the same ticker-days: **-0.01%**

The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, small samples and the absence of a full execution model make it easy to overfit.

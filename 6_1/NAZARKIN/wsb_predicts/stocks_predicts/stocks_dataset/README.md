# Stock price dataset for the WSB direction model

Built from the Nasdaq public quote/historical endpoint.

## Files

```
stocks_dataset/
├── raw/<TICKER>.json          raw API rows (newest first)
├── prices/<TICKER>.csv        per-ticker OHLCV + returns
├── prices_long.csv            tidy long table (all tickers)
├── prices_close_wide.csv      date x ticker close matrix
├── prices_open_wide.csv       date x ticker open matrix
├── tickers.txt                ticker universe
├── metadata.json              provenance, schema, coverage, caveats
├── data_quality.md            coverage / gap / extreme-return report
├── build_dataset.py           reproducible downloader (Nasdaq endpoint)
└── README.md
```

Reproduce with `python3 build_dataset.py` (32 requests, ~1 req/s).

## Schema (`prices_long.csv`)

| column | meaning |
|---|---|
| `ticker` | symbol |
| `assetclass` | `stock` or `etf` |
| `date` | trading day, `YYYY-MM-DD` |
| `open`, `high`, `low`, `close` | USD, split-adjusted by the source |
| `adj_close` | same as `close`; dividends are **not** adjusted |
| `volume` | shares |
| `dollar_volume` | `close * volume` |
| `ret_1d` | `adj_close.pct_change()` |
| `log_ret_1d` | `log(adj_close / adj_close.shift(1))` |
| `ohlc_fixed` | `true` if high/low were widened to contain open/close |
| `volume_missing` | `true` if the source had no volume for that day |
| `stale_quote` | `true` if OHLC are all equal and volume is missing |
| `volume_placeholder` | `true` if volume is the source's 9,999,999 sentinel |

## Coverage

- Requested: **2018-06-01 .. 2026-11-13**
- Tickers: **32** (28 stocks + SPY/QQQ/IWM/DIA)
- Per-ticker row counts and first/last dates: see `metadata.json` and `data_quality.md`.

## Important caveats

1. **Prices are split-adjusted, not dividend-adjusted.** `adj_close` equals `close`.
   For 5-day direction labels the difference is small, but this is price return,
   not total return.
2. **Later IPOs start later** (ARM 2023-09-14, RIVN 2021-11-11, etc.).
3. **No ^VIX** — the Nasdaq index endpoint rejected it; add VIX from another source
   if the model needs it.
4. **Four source rows had inconsistent OHLC** (high < open/close or low > open/close).
   They were fixed by widening high/low and flagged with `ohlc_fixed=true`;
   see `data_quality.md` for the exact rows.
5. **SPY 2026-04-20 is a stale quote** (all OHLC 710.14, no volume; `stale_quote=true`),
   and **SPY 2026-04-17 volume is the 9,999,999 placeholder**
   (`volume_placeholder=true`). Filter both before computing returns/features.
6. **Data is for research.** Respect Nasdaq's terms of use.

## Quick use

```python
import pandas as pd

px = pd.read_csv("stocks_dataset/prices_long.csv", parse_dates=["date"])
close = px.pivot(index="date", columns="ticker", values="adj_close").sort_index()
returns = close.pct_change()

# 5-day forward return for TSLA
fwd5 = close["TSLA"].shift(-5) / close["TSLA"] - 1
```

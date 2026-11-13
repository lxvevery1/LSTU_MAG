# Stock dataset — data quality report

Generated: 2026-11-13T19:16:49Z

Source: Nasdaq public quote/historical endpoint

Requested range: 2018-06-01 .. 2026-11-13

| ticker | class | rows | first | last | weekday cov % | max gap, days | ohlc fixed | vol missing | zero vol | max |1d ret| |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| AAPL | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.153 |
| TSLA | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.227 |
| NVDA | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.244 |
| MSFT | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.155 |
| AMZN | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.153 |
| GOOGL | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.126 |
| META | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.264 |
| AMD | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.238 |
| NFLX | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.351 |
| DIS | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.144 |
| BA | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 1 | 0 | 0 | 0.243 |
| GME | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 1.348 |
| AMC | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 3.012 |
| PLTR | stocks | 1538 | 2020-09-30 | 2026-11-12 | 96.31 | 4 | 1 | 0 | 0 | 0.308 |
| HOOD | stocks | 1330 | 2021-07-29 | 2026-11-12 | 96.31 | 4 | 0 | 0 | 0 | 0.504 |
| SOFI | stocks | 1492 | 2020-12-04 | 2026-11-12 | 96.26 | 4 | 0 | 0 | 0 | 0.579 |
| RIVN | stocks | 1256 | 2021-11-11 | 2026-11-12 | 96.17 | 4 | 0 | 0 | 0 | 0.266 |
| LCID | stocks | 1546 | 2020-09-18 | 2026-11-12 | 96.32 | 4 | 0 | 0 | 0 | 0.430 |
| NIO | stocks | 2054 | 2018-09-12 | 2026-11-12 | 96.34 | 4 | 0 | 0 | 0 | 0.758 |
| MSTR | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.292 |
| COIN | stocks | 1404 | 2021-04-14 | 2026-11-12 | 96.36 | 4 | 0 | 0 | 0 | 0.311 |
| SNAP | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.588 |
| UBER | stocks | 1889 | 2019-05-10 | 2026-11-12 | 96.38 | 4 | 1 | 0 | 0 | 0.383 |
| PYPL | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.246 |
| INTC | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.286 |
| MU | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.198 |
| AVGO | stocks | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.244 |
| ARM | stocks | 795 | 2023-09-14 | 2026-11-12 | 96.25 | 4 | 0 | 0 | 0 | 0.479 |
| SPY | etf | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 1 | 0 | 0.109 |
| QQQ | etf | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.120 |
| IWM | etf | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 1 | 0 | 0 | 0.133 |
| DIA | etf | 2125 | 2018-06-01 | 2026-11-12 | 96.37 | 4 | 0 | 0 | 0 | 0.128 |

## Cleaning applied / flagged

Rows flagged by `ohlc_fixed`, `volume_missing`, `stale_quote` or `volume_placeholder`: **6**.

| ticker | date | ohlc_fixed | volume_missing | stale_quote | volume_placeholder | open | high | low | close | volume |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BA | 2023-06-05 | True | False | False | False | 213.2800 | 213.2800 | 207.5000 | 208.7800 | 4996838 |
| IWM | 2023-06-05 | True | False | False | False | 181.2600 | 181.2600 | 178.4400 | 179.5800 | 34465800 |
| PLTR | 2023-06-05 | True | False | False | False | 14.3650 | 15.7000 | 14.3650 | 15.2400 | 109017800 |
| SPY | 2026-04-17 | False | False | False | True | 706.1400 | 712.3900 | 705.7600 | 710.1400 | 9999999 |
| SPY | 2026-04-20 | False | True | True | False | 710.1400 | 710.1400 | 710.1400 | 710.1400 |  |
| UBER | 2023-06-05 | True | False | False | False | 39.5000 | 40.7900 | 39.5000 | 40.4200 | 24218600 |

## Extreme one-day moves (|ret_1d| > 0.5)

| ticker | date | ret_1d | close |
|---|---|---:|---:|
| AMC | 2021-01-27 | +3.012 | 125.1204 |
| GME | 2021-01-27 | +1.348 | 86.8775 |
| GME | 2021-02-24 | +1.039 | 22.9275 |
| AMC | 2021-06-02 | +0.952 | 393.2805 |
| GME | 2021-01-26 | +0.927 | 36.9950 |
| AMC | 2024-05-13 | +0.784 | 5.1900 |
| NIO | 2018-09-13 | +0.758 | 11.6000 |
| GME | 2024-05-13 | +0.744 | 30.4500 |
| GME | 2021-01-29 | +0.679 | 81.2500 |
| GME | 2024-05-14 | +0.601 | 48.7500 |
| GME | 2021-02-02 | -0.600 | 22.5000 |
| SNAP | 2022-02-04 | +0.588 | 38.9100 |
| SOFI | 2021-01-07 | +0.579 | 19.1400 |
| GME | 2021-01-13 | +0.574 | 7.8500 |
| AMC | 2021-01-28 | -0.566 | 54.2608 |
| NIO | 2019-12-30 | +0.537 | 3.7200 |
| AMC | 2021-01-29 | +0.536 | 83.3717 |
| GME | 2021-03-25 | +0.527 | 45.9375 |
| AMC | 2020-11-09 | +0.514 | 23.7037 |
| GME | 2021-01-22 | +0.511 | 16.2525 |
| HOOD | 2021-08-04 | +0.504 | 70.3900 |

## Notes

- `weekday cov %` = rows / weekdays between first and last date; US market holidays explain most of the gap.
- `max gap` is the largest calendar-day gap between consecutive rows; holidays and halts produce gaps.
- `ohlc fixed` means high/low were widened to contain open/close; the source row was internally inconsistent.
- `stale_quote` means OHLC are all equal and volume is missing; filter these rows before computing returns.
- `volume_placeholder` marks the source's 9,999,999 volume sentinel.
- `max |1d ret|` is diagnostic only. Large values can be real (meme squeezes) or an unhandled corporate action; inspect before training.
- `adj_close` equals `close`; dividends are not adjusted.
- Tickers with later IPO dates naturally start later.

## Failures

none

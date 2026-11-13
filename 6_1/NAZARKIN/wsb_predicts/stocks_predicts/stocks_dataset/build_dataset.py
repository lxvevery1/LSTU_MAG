#!/usr/bin/env python3
"""Build the stock-price dataset for the WSB direction model.

Source: Nasdaq public quote/historical JSON endpoint.
Output: stocks_predicts/stocks_dataset/
  - raw/<TICKER>.json          raw rows from the API
  - prices/<TICKER>.csv        per-ticker OHLCV + returns
  - prices_long.csv            tidy long table (all tickers)
  - prices_close_wide.csv      date x ticker close matrix
  - prices_open_wide.csv       date x ticker open matrix
  - tickers.txt                ticker universe
  - metadata.json              provenance, schema, coverage, caveats
  - data_quality.md            coverage / gap / extreme-return report
  - README.md                  usage
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
START = "2018-06-01"
END = "2026-11-13"

TICKERS = {
    # TZ section 4.2 universe
    "AAPL": "stocks", "TSLA": "stocks", "NVDA": "stocks", "MSFT": "stocks",
    "AMZN": "stocks", "GOOGL": "stocks", "META": "stocks", "AMD": "stocks",
    "NFLX": "stocks", "DIS": "stocks", "BA": "stocks", "GME": "stocks",
    "AMC": "stocks", "PLTR": "stocks", "HOOD": "stocks", "SOFI": "stocks",
    "RIVN": "stocks", "LCID": "stocks", "NIO": "stocks", "MSTR": "stocks",
    "COIN": "stocks", "SNAP": "stocks", "UBER": "stocks", "PYPL": "stocks",
    "INTC": "stocks", "MU": "stocks", "AVGO": "stocks", "ARM": "stocks",
    # benchmarks
    "SPY": "etf", "QQQ": "etf", "IWM": "etf", "DIA": "etf",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}
URL = ("https://api.nasdaq.com/api/quote/{ticker}/historical"
       "?assetclass={ac}&fromdate={start}&todate={end}&limit=9999")


def parse_money(s: str) -> float:
    if s is None:
        return np.nan
    s = str(s).strip().replace("$", "").replace(",", "")
    if s in ("", "N/A", "n/a", "--", "-"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_volume(s: str) -> float:
    if s is None:
        return np.nan
    s = str(s).strip().replace(",", "")
    if s in ("", "N/A", "n/a", "--", "-"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def fetch_rows(ticker: str, ac: str, retries: int = 5) -> list[dict]:
    url = URL.format(ticker=ticker, ac=ac, start=START, end=END)
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                d = r.json()
                if d.get("data") and d["data"].get("tradesTable"):
                    return d["data"]["tradesTable"]["rows"]
                return []
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** i)
                continue
            return []
        except Exception:
            time.sleep(2 ** i)
    return []


def to_frame(ticker: str, ac: str, rows: list[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        recs.append({
            "ticker": ticker,
            "assetclass": ac,
            "date": pd.to_datetime(r.get("date"), format="%m/%d/%Y", errors="coerce"),
            "open": parse_money(r.get("open")),
            "high": parse_money(r.get("high")),
            "low": parse_money(r.get("low")),
            "close": parse_money(r.get("close")),
            "volume": parse_volume(r.get("volume")),
        })
    df = pd.DataFrame.from_records(recs)
    df = df.dropna(subset=["date", "close"]).copy()
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    # Some Nasdaq rows have high < open/close or low > open/close on a given day.
    # Widen the range to contain open and close; flag the row instead of dropping it.
    df["ohlc_fixed"] = ((df["high"] < df[["open", "close"]].max(axis=1)) |
                        (df["low"] > df[["open", "close"]].min(axis=1)))
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    df["volume_missing"] = df["volume"].isna()
    df["stale_quote"] = ((df["open"] == df["high"]) & (df["high"] == df["low"]) &
                         (df["low"] == df["close"]) & df["volume_missing"])
    df["volume_placeholder"] = (df["volume"] == 9_999_999)

    # Nasdaq's historical series is split-adjusted. Keep this explicit.
    df["adj_close"] = df["close"]
    df["ret_1d"] = df["adj_close"].pct_change()
    df["log_ret_1d"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    df["dollar_volume"] = df["close"] * df["volume"]
    df = df[["ticker", "assetclass", "date", "open", "high", "low", "close",
             "adj_close", "volume", "dollar_volume", "ret_1d", "log_ret_1d",
             "ohlc_fixed", "volume_missing", "stale_quote", "volume_placeholder"]]
    return df


def main() -> None:
    (BASE / "raw").mkdir(parents=True, exist_ok=True)
    (BASE / "prices").mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    coverage: dict[str, dict] = {}
    failures: list[str] = []

    for i, (ticker, ac) in enumerate(TICKERS.items(), 1):
        print(f"[{i:02d}/{len(TICKERS)}] {ticker} ({ac}) ...", end=" ", flush=True)
        rows = fetch_rows(ticker, ac)
        if not rows:
            print("FAILED")
            failures.append(ticker)
            continue
        (BASE / "raw" / f"{ticker}.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        df = to_frame(ticker, ac, rows)
        if df.empty:
            print("EMPTY")
            failures.append(ticker)
            continue
        df.to_csv(BASE / "prices" / f"{ticker}.csv", index=False,
                  float_format="%.6f")
        frames[ticker] = df
        coverage[ticker] = {
            "assetclass": ac,
            "rows": int(len(df)),
            "first_date": df["date"].min().strftime("%Y-%m-%d"),
            "last_date": df["date"].max().strftime("%Y-%m-%d"),
            "missing_open": int(df["open"].isna().sum()),
            "missing_high": int(df["high"].isna().sum()),
            "missing_low": int(df["low"].isna().sum()),
            "missing_volume": int(df["volume"].isna().sum()),
            "zero_volume_rows": int((df["volume"] == 0).sum()),
            "ohlc_fixed_rows": int(df["ohlc_fixed"].sum()),
            "volume_missing_rows": int(df["volume_missing"].sum()),
            "stale_quote_rows": int(df["stale_quote"].sum()),
            "volume_placeholder_rows": int(df["volume_placeholder"].sum()),
            "weekday_coverage_pct": round(
                100 * len(df) / np.busday_count(
                    np.datetime64(df["date"].min().date(), "D"),
                    np.datetime64(df["date"].max().date(), "D") + np.timedelta64(1, "D")), 2),
            "max_abs_ret_1d": float(df["ret_1d"].abs().max()),
            "max_gap_calendar_days": int(df["date"].diff().dt.days.max()),
        }
        print(f"rows={len(df)} {coverage[ticker]['first_date']}..{coverage[ticker]['last_date']}")
        time.sleep(1.0)

    if not frames:
        raise SystemExit("no data downloaded")

    long = pd.concat(frames.values(), ignore_index=True).sort_values(
        ["ticker", "date"]).reset_index(drop=True)
    long.to_csv(BASE / "prices_long.csv", index=False, float_format="%.6f")

    close_wide = long.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    open_wide = long.pivot(index="date", columns="ticker", values="open").sort_index()
    close_wide.to_csv(BASE / "prices_close_wide.csv", float_format="%.6f")
    open_wide.to_csv(BASE / "prices_open_wide.csv", float_format="%.6f")

    (BASE / "tickers.txt").write_text(
        "\n".join(TICKERS.keys()) + "\n", encoding="utf-8")

    meta = {
        "dataset": "wsb-stock-prices-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "name": "Nasdaq public quote/historical endpoint",
            "url_template": URL,
            "note": "Public JSON endpoint used for research; respect Nasdaq terms of use.",
        },
        "requested_range": {"start": START, "end": END},
        "ticker_universe": {
            "source": "TZ_stock_direction.md section 4.2 + IWM/DIA benchmarks",
            "n_tickers": len(TICKERS),
            "tickers": {t: ac for t, ac in TICKERS.items()},
        },
        "schema": {
            "ticker": "str",
            "assetclass": "stock | etf",
            "date": "YYYY-MM-DD (America/New_York trading day)",
            "open/high/low/close": "USD, split-adjusted by source",
            "adj_close": "same as close; dividends are NOT adjusted",
            "volume": "shares",
            "dollar_volume": "close * volume, USD",
            "ret_1d": "adj_close.pct_change()",
            "log_ret_1d": "log(adj_close / adj_close.shift(1))",
            "ohlc_fixed": "true if high/low were widened to contain open/close (source glitch)",
            "volume_missing": "true if the source returned no volume for that day",
            "stale_quote": "true if OHLC are all equal and volume is missing",
            "volume_placeholder": "true if volume is the source's 9,999,999 placeholder",
        },
        "adjustment": {
            "splits": "adjusted by Nasdaq (validated on AAPL 2020 4:1 split)",
            "dividends": "NOT adjusted: close/adj_close are price returns, not total returns",
            "why": "Nasdaq's dividend endpoint reports raw historical cash amounts and does not expose split factors, so a correct total-return adjustment is not possible from this source alone.",
        },
        "files": {
            "raw/<TICKER>.json": "raw API rows",
            "prices/<TICKER>.csv": "per-ticker OHLCV + returns",
            "prices_long.csv": "tidy long table, all tickers",
            "prices_close_wide.csv": "date x ticker adjusted close",
            "prices_open_wide.csv": "date x ticker open",
            "tickers.txt": "ticker universe",
            "data_quality.md": "coverage and anomaly report",
        },
        "coverage": coverage,
        "failures": failures,
        "known_limitations": [
            "No dividend adjustment; 5-day direction labels are affected only marginally.",
            "No ^VIX series: Nasdaq index historical endpoint rejected ^VIX.",
            "Tickers with later IPOs start at their first trading day (e.g. ARM 2023-09-14, RIVN 2021-11-11).",
            "Nasdaq data can contain gaps around holidays and halts; see data_quality.md.",
            "Prices reflect the source's split adjustments; extreme one-day moves should be checked before use.",
            "Four rows had high/low widened to contain open/close; see the ohlc_fixed flag and data_quality.md.",
            "SPY 2026-04-20 is a stale quote (OHLC all 710.14, no volume); flagged stale_quote=true.",
            "SPY 2026-04-17 volume is 9,999,999, the source's placeholder; flagged volume_placeholder=true.",
        ],
    }
    (BASE / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # data quality report
    lines = ["# Stock dataset — data quality report", "",
             f"Generated: {meta['created_utc']}", "",
             "Source: Nasdaq public quote/historical endpoint", "",
             f"Requested range: {START} .. {END}", "",
             "| ticker | class | rows | first | last | weekday cov % | max gap, days | ohlc fixed | vol missing | zero vol | max |1d ret| |",
             "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for t in TICKERS:
        if t not in coverage:
            lines.append(f"| {t} | {TICKERS[t]} | FAILED | | | | | | | | |")
            continue
        c = coverage[t]
        lines.append(
            f"| {t} | {c['assetclass']} | {c['rows']} | {c['first_date']} | "
            f"{c['last_date']} | {c['weekday_coverage_pct']:.2f} | "
            f"{c['max_gap_calendar_days']} | {c['ohlc_fixed_rows']} | "
            f"{c['volume_missing_rows']} | {c['zero_volume_rows']} | "
            f"{c['max_abs_ret_1d']:.3f} |")

    flagged_mask = (long["ohlc_fixed"] | long["volume_missing"] |
                    long["stale_quote"] | long["volume_placeholder"])
    fixed = long[flagged_mask].copy()
    lines += ["", "## Cleaning applied / flagged", "",
              "Rows flagged by `ohlc_fixed`, `volume_missing`, `stale_quote` or "
              f"`volume_placeholder`: **{len(fixed)}**.", ""]
    if len(fixed):
        lines.append("| ticker | date | ohlc_fixed | volume_missing | stale_quote | volume_placeholder | open | high | low | close | volume |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in fixed.iterrows():
            lines.append(
                f"| {r['ticker']} | {r['date'].date()} | {bool(r['ohlc_fixed'])} | "
                f"{bool(r['volume_missing'])} | {bool(r['stale_quote'])} | "
                f"{bool(r['volume_placeholder'])} | {r['open']:.4f} | {r['high']:.4f} | "
                f"{r['low']:.4f} | {r['close']:.4f} | "
                f"{'' if pd.isna(r['volume']) else int(r['volume'])} |")
    else:
        lines.append("none")

    extreme = long.assign(abs_ret=long["ret_1d"].abs()).dropna(subset=["ret_1d"])
    extreme = extreme[extreme["abs_ret"] > 0.5].sort_values(
        "abs_ret", ascending=False).head(25)
    lines += ["", "## Extreme one-day moves (|ret_1d| > 0.5)", ""]
    if len(extreme):
        lines.append("| ticker | date | ret_1d | close |")
        lines.append("|---|---|---:|---:|")
        for _, r in extreme.iterrows():
            lines.append(f"| {r['ticker']} | {r['date'].date()} | "
                         f"{r['ret_1d']:+.3f} | {r['close']:.4f} |")
    else:
        lines.append("none")

    lines += ["", "## Notes", "",
              "- `weekday cov %` = rows / weekdays between first and last date; "
              "US market holidays explain most of the gap.",
              "- `max gap` is the largest calendar-day gap between consecutive rows; "
              "holidays and halts produce gaps.",
              "- `ohlc fixed` means high/low were widened to contain open/close; "
              "the source row was internally inconsistent.",
              "- `stale_quote` means OHLC are all equal and volume is missing; "
              "filter these rows before computing returns.",
              "- `volume_placeholder` marks the source's 9,999,999 volume sentinel.",
              "- `max |1d ret|` is diagnostic only. Large values can be real (meme "
              "squeezes) or an unhandled corporate action; inspect before training.",
              "- `adj_close` equals `close`; dividends are not adjusted.",
              "- Tickers with later IPO dates naturally start later.",
              "", "## Failures", "",
              ", ".join(failures) if failures else "none"]
    (BASE / "data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # README
    readme = f"""# Stock price dataset for the WSB direction model

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
└── README.md
```

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

- Requested: **{START} .. {END}**
- Tickers: **{len(TICKERS)}** (28 stocks + SPY/QQQ/IWM/DIA)
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
"""
    (BASE / "README.md").write_text(readme, encoding="utf-8")

    print()
    print(f"saved {len(frames)} tickers to {BASE}")
    print(f"long rows: {len(long)}")
    print(f"failures: {failures or 'none'}")


if __name__ == "__main__":
    main()

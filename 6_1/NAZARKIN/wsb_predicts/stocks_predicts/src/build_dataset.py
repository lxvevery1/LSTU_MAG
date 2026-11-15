#!/usr/bin/env python3
"""Build the post x ticker direction dataset (TZ sections 3.3, 4.4, 4.5).

For each linked (post, ticker) pair and horizon H:
  t0 = open of the first trading session whose 09:30 ET open is strictly
       after the post timestamp;
  r  = close(t0 + H - 1) / open(t0) - 1;
  r_b = same for SPY;
  a  = r - beta * r_b, with beta estimated on 60 prior trading days;
  theta = max(0.005, 0.5 * stdev(daily returns, 20 prior days));
  y = UP if a >= theta, DOWN if a <= -theta, else FLAT.

The split is temporal (train 2019-2023, val 2024, test 2025-2026) with a
10-trading-day embargo at the boundaries.

Usage:
    python src/build_dataset.py --horizon 5
    python src/build_dataset.py --all-horizons
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import logging
import math
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import (
    CLASSES,
    DIRECTIONS,
    INFLUENCE_WEIGHTS,
    load_config,
    project_path,
    read_json,
    setup_logging,
    write_json,
)

LOG = logging.getLogger("build_dataset")
ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------- prices
def load_prices(cfg: dict) -> tuple[list, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return calendar, open_wide, close_wide, ret_wide, benchmark_close."""
    pcfg = cfg["prices"]
    path = project_path(pcfg["prices_long"])
    px = pd.read_csv(path, parse_dates=["date"])
    # Stale quotes would create fake returns and fake trading sessions.
    if "stale_quote" in px.columns:
        px = px[~px["stale_quote"].fillna(False).astype(bool)].copy()
    # Do not let a placeholder volume day masquerade as a valid session.
    if "volume_placeholder" in px.columns:
        pass  # price is still valid; only volume is a sentinel
    px = px.sort_values(["ticker", "date"])
    open_wide = px.pivot(index="date", columns="ticker", values="open").sort_index()
    close_wide = px.pivot(index="date", columns="ticker", values="close").sort_index()
    ret_wide = close_wide.pct_change(fill_method=None)
    benchmark = pcfg["benchmark"]
    if benchmark not in close_wide.columns:
        raise SystemExit(f"benchmark {benchmark} missing from price file")
    calendar = list(close_wide.index)
    return calendar, open_wide, close_wide, ret_wide, close_wide[benchmark]


def open_timestamp(day) -> datetime:
    if isinstance(day, pd.Timestamp):
        day = day.date()
    return datetime.combine(day, time(9, 30), tzinfo=ET)


def next_entry_index(created_utc: int | float, calendar: list) -> int | None:
    """Index of the first session whose open is strictly after the post."""
    ts = datetime.fromtimestamp(float(created_utc), tz=timezone.utc).astimezone(ET)
    d = pd.Timestamp(ts.date())
    idx = bisect.bisect_left(calendar, d)
    if idx < len(calendar) and calendar[idx] == d:
        if open_timestamp(d) > ts:
            return idx
        return idx + 1
    return idx if idx < len(calendar) else None


# ---------------------------------------------------------------- label math
def rolling_beta(ticker: str, t0_idx: int, ret_wide: pd.DataFrame,
                 benchmark: str, window: int, min_obs: int) -> float:
    if ticker not in ret_wide.columns or benchmark not in ret_wide.columns:
        return 1.0
    if t0_idx <= 1:
        return 1.0
    start = max(0, t0_idx - int(window))
    r = ret_wide[ticker].iloc[start:t0_idx].dropna()
    b = ret_wide[benchmark].iloc[start:t0_idx].dropna()
    joined = pd.concat([r.rename("r"), b.rename("b")], axis=1).dropna()
    if len(joined) < int(min_obs):
        return 1.0
    var = float(joined["b"].var(ddof=1))
    if not np.isfinite(var) or var <= 1e-12:
        return 1.0
    beta = float(joined["r"].cov(joined["b"]) / var)
    if not np.isfinite(beta):
        return 1.0
    return beta


def rolling_sigma(ticker: str, t0_idx: int, ret_wide: pd.DataFrame, window: int) -> float:
    if ticker not in ret_wide.columns or t0_idx <= 1:
        return 0.01
    start = max(0, t0_idx - int(window))
    r = ret_wide[ticker].iloc[start:t0_idx].dropna()
    if len(r) < 2:
        return 0.01
    sigma = float(r.std(ddof=1))
    return sigma if np.isfinite(sigma) and sigma > 0 else 0.01


def direction_label(a: float, theta: float) -> str:
    if a >= theta:
        return "UP"
    if a <= -theta:
        return "DOWN"
    return "FLAT"


def market_features(ticker: str, t0_idx: int, ret_wide: pd.DataFrame,
                    benchmark: str, secondary: str) -> dict:
    """Features known at the t0 open (all use data strictly before t0)."""
    out = {}
    for name, col in (("bench", benchmark), ("bench2", secondary)):
        if col in ret_wide.columns and t0_idx > 0:
            r = ret_wide[col].iloc[:t0_idx].dropna()
            out[f"{name}_ret_5"] = float(np.prod(1 + r.iloc[-5:]) - 1) if len(r) >= 5 else np.nan
            out[f"{name}_ret_20"] = float(np.prod(1 + r.iloc[-20:]) - 1) if len(r) >= 20 else np.nan
        else:
            out[f"{name}_ret_5"] = out[f"{name}_ret_20"] = np.nan
    if ticker in ret_wide.columns and t0_idx > 0:
        r = ret_wide[ticker].iloc[:t0_idx].dropna()
        out["ticker_ret_5"] = float(np.prod(1 + r.iloc[-5:]) - 1) if len(r) >= 5 else np.nan
        out["ticker_ret_20"] = float(np.prod(1 + r.iloc[-20:]) - 1) if len(r) >= 20 else np.nan
        out["ticker_vol_20"] = float(r.iloc[-20:].std(ddof=1)) if len(r) >= 5 else np.nan
    else:
        out["ticker_ret_5"] = out["ticker_ret_20"] = out["ticker_vol_20"] = np.nan
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--horizon", type=int, default=0,
                        help="single horizon (default: primary horizon)")
    parser.add_argument("--all-horizons", action="store_true")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    horizons = list(cfg["horizons"]) if args.all_horizons else [
        args.horizon or int(cfg["primary_horizon"])]
    pcfg = cfg["prices"]
    benchmark = pcfg["benchmark"]
    secondary = pcfg.get("secondary_benchmark", benchmark)
    beta_window = int(pcfg["beta_window"])
    beta_min = int(pcfg["beta_min_obs"])
    vol_window = int(pcfg["vol_window"])
    min_theta = float(pcfg["min_threshold"])
    theta_mult = float(pcfg["threshold_vol_multiplier"])
    embargo = int(cfg["splits"]["embargo_trading_days"])

    calendar, open_wide, close_wide, ret_wide, bench_close = load_prices(cfg)
    LOG.info("calendar: %s .. %s (%s sessions)", calendar[0], calendar[-1], len(calendar))

    posts_obj = read_json(project_path(cfg["paths"]["posts_labeled"]))
    posts = {p["id"]: p for arr in posts_obj["years"].values() for p in arr}
    links = pd.read_parquet(project_path(cfg["paths"]["links"]))
    LOG.info("posts: %s; links: %s", len(posts), len(links))

    # Split cutoffs for purge/embargo.
    years = np.array([d.year for d in calendar])
    def first_index(year: int) -> int | None:
        idx = np.where(years == year)[0]
        return int(idx[0]) if len(idx) else None

    val_start = first_index(int(cfg["splits"]["val_year"]))
    test_start = first_index(min(cfg["splits"]["test_years"]))
    train_exit_cutoff = calendar[val_start - embargo - 1] if val_start is not None and val_start - embargo - 1 >= 0 else None
    val_exit_cutoff = calendar[test_start - embargo - 1] if test_start is not None and test_start - embargo - 1 >= 0 else None
    LOG.info("purge cutoffs: train exit <= %s, val exit <= %s", train_exit_cutoff, val_exit_cutoff)

    def split_for(day) -> str:
        train_start, train_end = (int(y) for y in cfg["splits"]["train_years"])
        test_start, test_end = (int(y) for y in cfg["splits"]["test_years"])
        if train_start <= day.year <= train_end:
            return "train"
        if day.year == int(cfg["splits"]["val_year"]):
            return "val"
        if test_start <= day.year <= test_end:
            return "test"
        return "out"

    rows = []
    coverage = {
        "links_total": len(links),
        "no_post": 0,
        "no_calendar": 0,
        "not_in_split": 0,
        "invalid_prices": 0,
        "invalid_benchmark": 0,
        "purged": 0,
        "valid_by_horizon": collections.Counter(),
        "invalid_reasons": collections.Counter(),
    }
    # Cache per (ticker, t0_idx) since many posts share a ticker/day.
    beta_cache: dict[tuple[str, int], float] = {}
    sigma_cache: dict[tuple[str, int], float] = {}
    mkt_cache: dict[tuple[str, int], dict] = {}

    for link in links.itertuples(index=False):
        p = posts.get(link.post_id)
        if p is None:
            coverage["no_post"] += 1
            continue
        t0_idx = next_entry_index(p["created_utc"], calendar)
        if t0_idx is None or t0_idx >= len(calendar):
            coverage["no_calendar"] += 1
            coverage["invalid_reasons"]["no trading session after post"] += 1
            continue
        t0_date = calendar[t0_idx]
        split = split_for(t0_date)
        if split == "out":
            coverage["not_in_split"] += 1
            continue
        ticker = link.ticker
        if ticker not in open_wide.columns:
            coverage["invalid_prices"] += 1
            coverage["invalid_reasons"]["ticker missing"] += 1
            continue
        key = (ticker, t0_idx)
        if key not in beta_cache:
            beta_cache[key] = rolling_beta(ticker, t0_idx, ret_wide, benchmark, beta_window, beta_min)
            sigma_cache[key] = rolling_sigma(ticker, t0_idx, ret_wide, vol_window)
            mkt_cache[key] = market_features(ticker, t0_idx, ret_wide, benchmark, secondary)
        beta = beta_cache[key]
        sigma = sigma_cache[key]
        theta = max(min_theta, theta_mult * sigma)

        for h in horizons:
            exit_idx = t0_idx + int(h) - 1
            if exit_idx >= len(calendar):
                coverage["invalid_reasons"]["exit after available prices"] += 1
                continue
            exit_date = calendar[exit_idx]
            entry_px = open_wide.at[t0_date, ticker]
            exit_px = close_wide.at[exit_date, ticker]
            bench_entry = open_wide.at[t0_date, benchmark]
            bench_exit = close_wide.at[exit_date, benchmark]
            if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px <= 0:
                coverage["invalid_prices"] += 1
                coverage["invalid_reasons"]["missing ticker entry/exit price"] += 1
                continue
            if not np.isfinite(bench_entry) or not np.isfinite(bench_exit) or bench_entry <= 0:
                coverage["invalid_benchmark"] += 1
                coverage["invalid_reasons"]["missing benchmark price"] += 1
                continue
            raw = float(exit_px / entry_px - 1)
            bench = float(bench_exit / bench_entry - 1)
            abnormal = raw - beta * bench
            label = direction_label(abnormal, theta)
            purged = False
            if split == "train" and train_exit_cutoff is not None and exit_date > train_exit_cutoff:
                purged = True
            if split == "val" and val_exit_cutoff is not None and exit_date > val_exit_cutoff:
                purged = True
            if purged:
                coverage["purged"] += 1
            coverage["valid_by_horizon"][h] += 1

            probs = p.get("class_probs") or {}
            row = {
                "post_id": p["id"],
                "created_utc": int(p["created_utc"]),
                "ticker": ticker,
                "title": p.get("title", ""),
                "body": p.get("body", ""),
                "flair": p.get("flair", ""),
                "score": int(p.get("score", 0)),
                "num_comments": int(p.get("num_comments", 0)),
                "author_hash": p.get("author_hash", ""),
                "class_label": p.get("class_label"),
                "label_source": p.get("label_source"),
                "review_status": p.get("review_status"),
                "confidence": float(p.get("confidence", 0.0) or 0.0),
                "influence_weight": float(p.get("influence_weight", INFLUENCE_WEIGHTS.get(p.get("class_label"), 0.0))),
                "mention_count": int(link.mention_count),
                "match_methods": link.match_methods,
                "weight_share": float(link.weight_share),
                "fragment": link.fragment,
                "t0_utc": open_timestamp(t0_date).astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "t0_date": pd.Timestamp(t0_date),
                "exit_date": pd.Timestamp(exit_date),
                "horizon_days": int(h),
                "entry_price": float(entry_px),
                "exit_price": float(exit_px),
                "raw_return": raw,
                "benchmark": benchmark,
                "benchmark_return": bench,
                "beta": float(beta),
                "abnormal_return": float(abnormal),
                "threshold": float(theta),
                "direction_label": label,
                "split": split,
                "purged": bool(purged),
                "data_version": cfg["project"]["data_version"],
            }
            for c in CLASSES:
                row[f"prob_{c}"] = float(probs.get(c, 0.0) or 0.0)
            row.update(mkt_cache[key])
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no valid post-ticker-horizon rows")
    df = df.sort_values(["t0_date", "ticker", "post_id", "horizon_days"]).reset_index(drop=True)
    out = project_path(cfg["paths"]["dataset"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    primary_h = int(cfg["primary_horizon"])
    if args.all_horizons or primary_h in horizons:
        pairs = df[(df["horizon_days"] == primary_h) & (~df["purged"])].copy()
        pairs.to_parquet(project_path(cfg["paths"]["pairs_h5"]), index=False)
        LOG.info("primary pairs H=%s: %s", primary_h, len(pairs))

    # ------------------------- data quality report
    report = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_version": cfg["project"]["data_version"],
        "horizons": horizons,
        "posts_total": len(posts),
        "links_total": len(links),
        "posts_with_ticker": len({r.post_id for r in links.itertuples()}),
        "pairs_valid": int(len(df[(df["horizon_days"] == primary_h) & (~df["purged"])])),
        "rows_total": int(len(df)),
        "coverage": {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in coverage.items()},
        "split_counts": df.groupby(["split", "purged"]).size().reset_index(name="n").to_dict("records"),
        "direction_by_horizon": {},
        "class_by_year": {},
        "direction_by_year": {},
        "ticker_counts": {},
        "beta": {},
        "threshold": {},
        "flat_rate_by_horizon": {},
    }
    for h, g in df.groupby("horizon_days"):
        report["direction_by_horizon"][str(h)] = g["direction_label"].value_counts().to_dict()
        report["flat_rate_by_horizon"][str(h)] = round(float((g["direction_label"] == "FLAT").mean()), 4)
        report["beta"][str(h)] = {
            "mean": round(float(g["beta"].mean()), 4),
            "median": round(float(g["beta"].median()), 4),
            "p05": round(float(g["beta"].quantile(0.05)), 4),
            "p95": round(float(g["beta"].quantile(0.95)), 4),
        }
        report["threshold"][str(h)] = {
            "mean": round(float(g["threshold"].mean()), 5),
            "median": round(float(g["threshold"].median()), 5),
        }
    for y, g in df.groupby(df["t0_date"].dt.year):
        report["direction_by_year"][str(y)] = g["direction_label"].value_counts().to_dict()
    # post-level class distribution (not pair-weighted)
    cls = collections.Counter(p.get("class_label") for p in posts.values())
    report["class_distribution_posts"] = dict(cls)
    for y, arr in posts_obj["years"].items():
        report["class_by_year"][y] = dict(collections.Counter(p.get("class_label") for p in arr))
    report["ticker_counts"] = df[df["horizon_days"] == primary_h]["ticker"].value_counts().to_dict()

    write_json(project_path(cfg["paths"]["reports"]) / "dataset_report.json", report)

    # Markdown data-quality report + data card.
    lines = ["# Data quality report — WSB direction dataset", "",
             f"Generated: {report['created_utc']}", "",
             f"Data version: `{report['data_version']}`", "",
             "## Corpus and linking", "",
             f"- Posts: **{report['posts_total']}**",
             f"- Posts with at least one ticker: **{report['posts_with_ticker']}** "
             f"({100 * report['posts_with_ticker'] / max(1, report['posts_total']):.1f}%)",
             f"- Post x ticker links: **{report['links_total']}**",
             f"- Valid post x ticker x horizon rows: **{report['rows_total']}**",
             f"- Valid pairs at primary horizon H={primary_h}: **{report['pairs_valid']}**",
             "", "## Direction distribution by horizon", "",
             "| H | UP | FLAT | DOWN | FLAT rate |", "|---:|---:|---:|---:|---:|"]
    for h in horizons:
        d = report["direction_by_horizon"].get(str(h), {})
        lines.append(f"| {h} | {d.get('UP', 0)} | {d.get('FLAT', 0)} | {d.get('DOWN', 0)} | "
                     f"{100 * report['flat_rate_by_horizon'].get(str(h), 0):.1f}% |")
    lines += ["", "## Split counts", "", "| split | purged | rows |", "|---|---:|---:|"]
    for r in report["split_counts"]:
        lines.append(f"| {r['split']} | {r['purged']} | {r['n']} |")
    lines += ["", "## Coverage / exclusions", ""]
    for k, v in coverage.items():
        if isinstance(v, dict):
            lines.append(f"- `{k}`: " + ", ".join(f"{kk}={vv}" for kk, vv in v.items()))
        else:
            lines.append(f"- `{k}`: {v}")
    lines += ["", "## Threshold and beta", "",
              "| H | beta mean | beta p05 | beta p95 | threshold mean |",
              "|---:|---:|---:|---:|---:|"]
    for h in horizons:
        b = report["beta"].get(str(h), {})
        t = report["threshold"].get(str(h), {})
        lines.append(f"| {h} | {b.get('mean')} | {b.get('p05')} | {b.get('p95')} | {t.get('mean')} |")
    lines += ["", "## Known limitations", "",
              "- New posts are classifier-prelabelled; manual gold-audit coverage is reported separately.",
              "- Prices are split-adjusted but not dividend-adjusted (Nasdaq source); label noise from dividends is small for 1–10 day horizons.",
              "- Benchmark beta uses 60 prior sessions and falls back to 1.0 when fewer than 30 observations are available.",
              "- The 10-day embargo removes training/validation rows whose forward window crosses the next split.",
              "- Class labels for new posts are not ground truth; see `prelabel_report.json` and `gold_audit_sample.json`.",
              "- Classifier-vs-manual agreement on its 20% holdout is κ=0.573 [0.540, 0.605] (`classifier_audit_proxy.json`); this is a proxy, not a gold audit of the 8,000 new prelabels.",
              "- 157 posts (0.98%) share normalized text, mostly manual posts whose body is `[removed by moderator]`; post IDs are still unique.",
              "- Ticker precision/recall on the fixed 300-post sample is not measured until humans fill `gold_tickers` (`ticker_gold_audit.json`, status blocked)."]
    (project_path(cfg["paths"]["reports"]) / "data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    data_card = f"""# Data card — {report['data_version']}

## Source
- Posts: r/WallStreetBets via `arctic-shift.photon-reddit.com`; existing 8,000 manual labels plus new posts from the local Arctic Shift pool.
- Prices: Nasdaq public historical endpoint, 2018-06-01 .. 2026-11-13, split-adjusted OHLCV.
- Benchmarks: SPY (primary), QQQ (secondary).

## Unit of analysis
One row is a (post, ticker, horizon) tuple.  The primary model uses H={primary_h} and the ticker-day aggregate is formed by grouping on (ticker, t0_date).

## Labels
- Class labels: existing 8,000 are manual; new posts are frozen-FinBERT prelabels (`classifier_prelabel_unverified`) with a review queue.
- Direction labels: forward abnormal return, beta-adjusted, threshold `max(0.005, 0.5 * sigma_20d)`.
- Entry: open of the first session strictly after `created_utc`; exit: close of the H-th session including entry day.

## Known biases
- Survivorship: the universe was chosen from currently popular/liquid WSB names; delisted tickers are not represented.
- Selection: 2,000 posts/year target; 2026 is partial (through 2026-11-14).
- Class labels for the new 8,000 are model-generated until human review is completed.
- Classifier-vs-manual agreement on its 20% holdout is κ=0.573 [0.540, 0.605] (`classifier_audit_proxy.json`); this is a proxy, not a gold audit.
- 157 posts (0.98%) share normalized text, mostly manual posts with `[removed by moderator]` bodies; post IDs are unique.
- Ticker precision/recall on the fixed 300-post sample is not measured until humans fill `gold_tickers` (`ticker_gold_audit.json`, status blocked).
- WSB user base and vocabulary drift over time; temporal splits are mandatory.
- Reddit data are observational; labels are associative, not causal.

## Reproducibility
`python src/collect_posts.py && python src/label_posts.py && python src/extract_tickers.py && python src/build_dataset.py --all-horizons`
"""
    (project_path(cfg["paths"]["reports"]) / "data_card.md").write_text(data_card, encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["posts_total", "links_total", "rows_total", "pairs_valid", "flat_rate_by_horizon", "split_counts"]}, indent=2))


if __name__ == "__main__":
    main()

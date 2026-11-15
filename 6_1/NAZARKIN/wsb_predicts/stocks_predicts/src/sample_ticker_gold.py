#!/usr/bin/env python3
"""Create the fixed 300-post ticker-linking gold sample (TZ 8.1).

The sample is stratified by year and includes the automatic extraction result
for each post.  A human annotator fills ``gold_tickers`` with the tickers that
are genuinely discussed (only symbols from the 30-name universe; use ``[]`` for
a post that mentions none).  Keep post IDs and order unchanged so the scorer is
reproducible.

Usage:
    python src/sample_ticker_gold.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from common import load_config, project_path, read_json, write_json

SAMPLE_SIZE = 300


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/raw/ticker_gold_sample.json")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing sample (discards human labels)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    universe = sorted(load_config(cfg["tickers"]["config"])["universe"])
    out = project_path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists; use --force only if you intend to "
                         "discard the human gold labels")

    labeled = read_json(project_path(cfg["paths"]["posts_labeled"]))
    posts = [p for arr in labeled["years"].values() for p in arr]
    links = pd.read_parquet(project_path(cfg["paths"]["links"]))
    by_post = links.groupby("post_id")["ticker"].apply(lambda s: sorted(set(s))).to_dict()

    rng = np.random.default_rng(args.seed)
    by_year: dict[str, list[dict]] = {}
    for p in posts:
        year = str(datetime.fromtimestamp(p["created_utc"], tz=timezone.utc).year)
        by_year.setdefault(year, []).append(p)

    years = sorted(by_year)
    base, extra = divmod(args.size, max(1, len(years)))
    chosen = []
    for i, year in enumerate(years):
        n = base + (1 if i < extra else 0)
        arr = by_year[year]
        idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
        chosen.extend(arr[int(j)] for j in idx)

    rows = []
    for p in chosen:
        rows.append({
            "post_id": p["id"],
            "year": str(datetime.fromtimestamp(p["created_utc"], tz=timezone.utc).year),
            "flair": p.get("flair", ""),
            "title": p["title"],
            "body": p["body"],
            "auto_tickers": by_post.get(p["id"], []),
            "gold_tickers": None,
        })

    write_json(out, {
        "instructions": (
            "Fill gold_tickers with the list of universe tickers genuinely discussed "
            "in the post (use [] if none). Only these symbols are valid: "
            + ", ".join(universe)
        ),
        "universe": universe,
        "n": len(rows),
        "seed": args.seed,
        "stratified_by": "year",
        "sample": rows,
    })
    print(json.dumps({
        "n": len(rows),
        "per_year": {y: sum(1 for r in rows if r["year"] == y) for y in years},
        "with_auto_ticker": sum(1 for r in rows if r["auto_tickers"]),
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()

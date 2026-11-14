#!/usr/bin/env python3
"""Extract and link tickers in WSB posts (TZ sections 4.2 and 5.1 stage B).

Matching rules:
* ``$TSLA`` cashtags are accepted for every symbol in the universe;
* standalone uppercase tokens are accepted only if they are in the universe
  and not in the ambiguity/stop list (so ``AI``, ``DD``, ``ON`` are rejected);
* company aliases are matched case-insensitively on word boundaries;
* at most five tickers per post are kept, ranked by mention count.

The output is one row per (post, ticker), including a +/-200 character text
fragment around the first mention, which the direction model receives as an
additional text segment.

Usage:
    python src/extract_tickers.py
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from common import (
    load_config,
    normalize_text,
    project_path,
    read_json,
    setup_logging,
    write_json,
)

LOG = logging.getLogger("extract_tickers")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
CASHTAG_RE = re.compile(r"(?<![A-Za-z0-9])\$([A-Za-z]{1,5})\b")
UPPER_RE = re.compile(r"\b([A-Z]{2,5})\b")


def load_ticker_config(path: str = "configs/tickers.yaml") -> dict:
    import yaml

    with open(project_path(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_for_matching(text: str) -> str:
    text = text or ""
    text = MD_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub(" ", text)
    return text


def compile_alias_patterns(aliases: dict[str, list[str]]) -> dict[str, list[tuple[str, re.Pattern]]]:
    out: dict[str, list[tuple[str, re.Pattern]]] = {}
    for ticker, names in aliases.items():
        pats = []
        for name in names:
            pat = re.compile(
                r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            pats.append((name, pat))
        out[ticker] = pats
    return out


def extract_tickers(text: str, tcfg: dict) -> list[dict]:
    """Return up to ``max_per_post`` matches with method and position."""
    universe = list(dict.fromkeys(tcfg.get("universe", [])))
    universe_set = set(universe)
    ambiguous = {x.upper() for x in tcfg.get("ambiguous", [])}
    stoplist = {x.upper() for x in tcfg.get("stoplist", [])}
    aliases = compile_alias_patterns(tcfg.get("aliases", {}))
    max_per_post = int(tcfg.get("max_per_post", 5))
    text = clean_for_matching(text)

    found: dict[str, dict] = {}

    def add(ticker: str, method: str, pos: int, raw: str) -> None:
        if ticker not in universe_set:
            return
        rec = found.setdefault(ticker, {
            "ticker": ticker,
            "mention_count": 0,
            "first_pos": pos,
            "methods": set(),
            "raw_matches": [],
        })
        rec["mention_count"] += 1
        rec["first_pos"] = min(rec["first_pos"], pos)
        rec["methods"].add(method)
        if len(rec["raw_matches"]) < 8:
            rec["raw_matches"].append(raw)

    for m in CASHTAG_RE.finditer(text):
        ticker = m.group(1).upper()
        if ticker in universe_set:
            add(ticker, "cashtag", m.start(), m.group(0))

    for m in UPPER_RE.finditer(text):
        ticker = m.group(1).upper()
        if ticker not in universe_set or ticker in stoplist or ticker in ambiguous:
            # Ambiguous uppercase symbols are accepted because in an all-caps
            # token they are much more likely to be a ticker; only the explicit
            # stoplist is rejected here.
            if ticker not in universe_set or ticker in stoplist:
                continue
        add(ticker, "uppercase", m.start(), m.group(0))

    for ticker, pats in aliases.items():
        if ticker not in universe_set:
            continue
        for name, pat in pats:
            for m in pat.finditer(text):
                add(ticker, "alias", m.start(), m.group(0))

    rows = []
    for ticker, rec in found.items():
        rows.append({
            "ticker": ticker,
            "mention_count": rec["mention_count"],
            "first_pos": rec["first_pos"],
            "methods": sorted(rec["methods"]),
            "raw_matches": rec["raw_matches"],
        })
    rows.sort(key=lambda r: (-r["mention_count"], r["first_pos"], r["ticker"]))
    return rows[:max_per_post]


def make_fragment(text: str, pos: int, width: int) -> str:
    start = max(0, pos - width)
    end = min(len(text), pos + width)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def load_posts(cfg: dict) -> list[dict]:
    obj = read_json(project_path(cfg["paths"]["posts_labeled"]))
    return [p for arr in obj["years"].values() for p in arr]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--gold", default="data/raw/ticker_gold.json",
                        help="optional file with post_id -> [tickers] gold labels")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    tcfg = load_ticker_config(cfg["tickers"]["config"])
    width = int(cfg["tickers"].get("fragment_chars", 200))
    posts = load_posts(cfg)
    LOG.info("posts: %s; universe: %s", len(posts), len(tcfg["universe"]))

    rows = []
    per_year = collections.defaultdict(lambda: {"posts": 0, "with_ticker": 0})
    mention_methods = collections.Counter()
    foreign_cashtags = collections.Counter()
    ticker_freq = collections.Counter()
    no_ticker_examples = []

    allowed = set(tcfg["universe"])
    for p in posts:
        text = ((p.get("title") or "") + "\n\n" + (p.get("body") or "")).strip()
        year = str(datetime.fromtimestamp(p["created_utc"], tz=timezone.utc).year)
        per_year[year]["posts"] += 1
        # Diagnostic: how many cashtags outside the universe appear?  This is
        # computed before the early continue so that no-ticker posts contribute
        # to the denominator of universe coverage.
        for m in CASHTAG_RE.finditer(clean_for_matching(text)):
            sym = m.group(1).upper()
            if sym not in allowed:
                foreign_cashtags[sym] += 1
        matches = extract_tickers(text, tcfg)
        if not matches:
            if len(no_ticker_examples) < 30:
                no_ticker_examples.append({"post_id": p["id"], "title": p.get("title", "")[:160]})
            continue
        per_year[year]["with_ticker"] += 1
        n = len(matches)
        for m in matches:
            ticker_freq[m["ticker"]] += 1
            mention_methods.update(m["methods"])
            rows.append({
                "post_id": p["id"],
                "ticker": m["ticker"],
                "mention_count": m["mention_count"],
                "match_methods": ",".join(m["methods"]),
                "weight_share": 1.0 / n,
                "fragment": make_fragment(text, m["first_pos"], width),
                "created_utc": p["created_utc"],
                "year": year,
            })

    links = pd.DataFrame(rows)
    out = project_path(cfg["paths"]["links"])
    out.parent.mkdir(parents=True, exist_ok=True)
    links.to_parquet(out, index=False)

    with_ticker = len({r["post_id"] for r in rows})
    n_mentions = len(rows) + sum(foreign_cashtags.values())
    coverage = 100.0 * (len(rows) / n_mentions) if n_mentions else 0.0
    report = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_posts": len(posts),
        "n_posts_with_ticker": with_ticker,
        "pct_posts_with_ticker": round(100 * with_ticker / max(1, len(posts)), 2),
        "n_links": len(rows),
        "unique_tickers": len(ticker_freq),
        "ticker_frequency": dict(ticker_freq.most_common()),
        "per_year": {y: {**v, "pct_with_ticker": round(100 * v["with_ticker"] / max(1, v["posts"]), 2)}
                     for y, v in sorted(per_year.items())},
        "match_methods": dict(mention_methods),
        "foreign_cashtags": dict(foreign_cashtags.most_common(100)),
        "universe_mention_coverage_pct": round(coverage, 2),
        "no_ticker_examples": no_ticker_examples,
    }

    # Optional gold ticker audit.
    gold_path = project_path(args.gold)
    if gold_path.exists():
        gold = read_json(gold_path)
        gold = gold.get("labels", gold) if isinstance(gold, dict) else gold
        pred = collections.defaultdict(set)
        for r in rows:
            pred[r["post_id"]].add(r["ticker"])
        tp = fp = fn = 0
        for pid, expected in gold.items():
            exp = set(expected)
            got = pred.get(pid, set())
            tp += len(exp & got)
            fp += len(got - exp)
            fn += len(exp - got)
        report["gold_ticker_audit"] = {
            "n_posts": len(gold),
            "precision": round(tp / max(1, tp + fp), 4),
            "recall": round(tp / max(1, tp + fn), 4),
            "f1": round(2 * tp / max(1, 2 * tp + fp + fn), 4),
            "tp": tp, "fp": fp, "fn": fn,
        }
    else:
        report["gold_ticker_audit"] = {
            "status": "not_available",
            "note": "Create data/raw/ticker_gold.json with 300 hand-checked posts to compute precision/recall.",
        }

    write_json(project_path(cfg["paths"]["reports"]) / "ticker_extraction_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

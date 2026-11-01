#!/usr/bin/env python3
"""Smoke test for the WSB semantic search package."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wsb_data as wd
from semantic_search import Searcher

failures = []


def check(name, condition, detail=""):
    print(f"  [{'ok ' if condition else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not condition:
        failures.append(name)


def mentions_apple(result: dict) -> bool:
    if "AAPL" in result.get("tickers", []):
        return True
    text = f"{result.get('title', '')} {result.get('snippet', '')} {result.get('body', '')}".lower()
    return any(word in text for word in ("apple", "aapl", "appl"))


def main() -> int:
    print("1. query normalisation")
    normalized = wd.normalize_query("акции APPL")
    check("APPL is mapped to AAPL", "AAPL" in normalized, normalized)
    check("Apple company name is added", "apple" in normalized.lower(), normalized)
    check("Russian 'акции' becomes stock", "stock" in normalized.lower(), normalized)
    check("eventually empty query stays empty", wd.normalize_query("") == "")

    print("2. model and index")
    searcher = Searcher()
    stats = searcher.stats()
    check("model loads", bool(searcher.model), searcher.model_path)
    check("index has vectors", stats["vectors"] > 0, str(stats["vectors"]))
    check("all posts are in metadata", stats["posts"] >= 8000, str(stats["posts"]))
    check("vector dimension is 384", stats["dimension"] == 384, str(stats["dimension"]))

    print("3. user example: 'акции APPL'")
    payload = searcher.search("акции APPL", top_k=10)
    results = payload["results"]
    check("returns 10 results", len(results) == 10, str(len(results)))
    check("normalised query contains AAPL", "AAPL" in payload["normalized_query"])
    check("results are unique posts", len({r["post_id"] for r in results}) == len(results))
    check("scores are descending",
          all(results[i]["score"] >= results[i + 1]["score"] for i in range(len(results) - 1)))
    apple_hits = sum(mentions_apple(result) for result in results)
    check("at least 6/10 results mention Apple", apple_hits >= 6, f"{apple_hits}/10")

    print("4. English semantic queries")
    gme = searcher.search("GameStop short squeeze", top_k=5)["results"]
    check("GME query returns GME posts",
          sum("GME" in r["tickers"] for r in gme) >= 3,
          str([r["tickers"] for r in gme]))
    tsla = searcher.search("Tesla earnings", top_k=5)["results"]
    check("Tesla query returns TSLA posts",
          sum("TSLA" in r["tickers"] for r in tsla) >= 3,
          str([r["tickers"] for r in tsla]))
    nvda = searcher.search("NVDA AI chips", top_k=5)["results"]
    check("NVDA query returns NVDA posts",
          sum("NVDA" in r["tickers"] for r in nvda) >= 2,
          str([r["tickers"] for r in nvda]))

    print("5. edge cases")
    check("top_k=1 works", len(searcher.search("apple", top_k=1)["results"]) == 1)
    try:
        searcher.search("apple", top_k=0)
        check("top_k=0 raises ValueError", False)
    except ValueError:
        check("top_k=0 raises ValueError", True)
    check("unknown query still returns something",
          len(searcher.search("zzzzz qqqqq", top_k=3)["results"]) == 3)
    searcher.close()

    print("6. evaluation artefacts")
    eval_path = HERE / "index" / "eval_results.json"
    check("eval_results.json exists", eval_path.exists())
    if eval_path.exists():
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        base = ev["base"]["overall"]["recall@10"]
        tuned = ev["finetuned"]["overall"]["recall@10"]
        check("fine-tuned recall@10 beats base", tuned > base, f"{tuned:.4f} > {base:.4f}")
        check("fine-tuned ticker hit@10 >= 0.5",
              ev["finetuned"]["by_kind"]["ticker"]["hit@10"] >= 0.5,
              str(ev["finetuned"]["by_kind"]["ticker"]["hit@10"]))

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

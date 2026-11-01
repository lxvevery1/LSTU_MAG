#!/usr/bin/env python3
"""CLI semantic search over the indexed WallStreetBets posts."""
from __future__ import annotations

import argparse
import json
import textwrap

from semantic_search import DEFAULT_INDEX_DIR, DEFAULT_MODEL, Searcher


def print_results(payload: dict) -> None:
    print(f"query: {payload['query']}")
    if payload["normalized_query"] != payload["query"]:
        print(f"normalized: {payload['normalized_query']}")
    print()
    for result in payload["results"]:
        meta = f"{result['year']} | {result['flair'] or 'no flair'} | score {result['post_score']}"
        print(f"{result['rank']:>2}. [{result['score']:.4f}] {result['title']}")
        print(f"    {meta} | {result['url']}")
        snippet = textwrap.shorten(result["snippet"], width=260, placeholder=" …")
        print(f"    {snippet}")
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="search query")
    ap.add_argument("-n", "--top-k", type=int, default=10)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("-i", "--interactive", action="store_true")
    args = ap.parse_args()

    searcher = Searcher(args.model, args.index_dir)
    try:
        if args.interactive:
            while True:
                try:
                    query = input("query> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not query:
                    continue
                payload = searcher.search(
                    query, top_k=args.top_k, normalize=not args.no_normalize)
                if args.json:
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
                else:
                    print_results(payload)
            return 0

        query = " ".join(args.query).strip()
        if not query:
            ap.error("provide a query or use --interactive")
        payload = searcher.search(query, top_k=args.top_k, normalize=not args.no_normalize)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print_results(payload)
    finally:
        searcher.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

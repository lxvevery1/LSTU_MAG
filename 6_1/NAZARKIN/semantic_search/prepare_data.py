#!/usr/bin/env python3
"""Build the chunk corpus, contrastive training pairs and evaluation queries."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import wsb_data as wd


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", default=str(wd.DEFAULT_POSTS))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    posts = wd.load_posts(Path(args.posts))
    train_posts, test_posts = wd.split_posts(posts, test_size=args.test_size, seed=args.seed)

    all_chunks = list(wd.iter_chunks(posts))
    chunk_rows = [wd.chunk_to_dict(c) for c in all_chunks]
    write_jsonl(out / "corpus.jsonl", chunk_rows)

    train_pairs = wd.build_training_pairs(train_posts, seed=args.seed)
    pair_rows = []
    for pair in train_pairs:
        raw = pair["query"]
        normalized = wd.normalize_query(raw)
        variants = [raw] if normalized == raw else [raw, normalized]
        for query in variants:
            row = dict(pair)
            row["query"] = query
            pair_rows.append(row)
    write_jsonl(out / "train_pairs.jsonl", pair_rows)

    eval_queries = wd.build_eval_queries(test_posts, seed=args.seed)
    (out / "eval_queries.json").write_text(
        json.dumps(eval_queries, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "posts": len(posts),
        "train_posts": len(train_posts),
        "test_posts": len(test_posts),
        "chunks": len(all_chunks),
        "chunks_with_body": sum(bool(c.body) for c in all_chunks),
        "chunks_with_tickers": sum(bool(c.tickers) for c in all_chunks),
        "train_pairs_raw": len(train_pairs),
        "train_pairs": len(pair_rows),
        "eval_queries": len(eval_queries),
        "eval_by_kind": dict(Counter(q["kind"] for q in eval_queries)),
        "top_tickers": dict(Counter(t for c in all_chunks for t in c.tickers).most_common(30)),
        "test_post_ids": [str(p.get("id") or p.get("post_id")) for p in test_posts],
    }
    (out / "prepare_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "test_post_ids"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

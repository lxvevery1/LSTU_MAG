#!/usr/bin/env python3
"""Retrieval evaluation: base MiniLM vs the fine-tuned WSB checkpoint."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

import wsb_data as wd

HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def evaluate_model(model_path: str, texts: List[str], post_ids: List[str],
                   queries: List[dict], device: str | None,
                   batch_size: int = 64, max_candidates: int = 300) -> dict:
    started = time.time()
    model = SentenceTransformer(model_path, device=device)
    corpus_vectors = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True)
    corpus_vectors = np.asarray(corpus_vectors, dtype="float32")

    query_texts = [wd.normalize_query(q["query"]) for q in queries]
    query_vectors = model.encode(
        query_texts, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False)
    query_vectors = np.asarray(query_vectors, dtype="float32")

    by_kind = defaultdict(list)
    rows = []
    for query, vector in zip(queries, query_vectors):
        scores = corpus_vectors @ vector
        k = min(max_candidates, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        ordered_posts = []
        seen = set()
        for index in top_idx:
            post_id = post_ids[int(index)]
            if post_id in seen:
                continue
            seen.add(post_id)
            ordered_posts.append(post_id)
            if len(ordered_posts) >= 50:
                break

        relevant = set(query["relevant_post_ids"])
        record = {"query": query["query"], "kind": query["kind"], "n_relevant": len(relevant)}
        for at in (10, 50):
            hits = len(set(ordered_posts[:at]) & relevant)
            record[f"recall@{at}"] = hits / len(relevant) if relevant else 0.0
            record[f"hit@{at}"] = 1.0 if hits else 0.0
        mrr = 0.0
        for rank, post_id in enumerate(ordered_posts[:10], start=1):
            if post_id in relevant:
                mrr = 1.0 / rank
                break
        record["mrr@10"] = mrr
        rows.append(record)
        by_kind[query["kind"]].append(record)

    def aggregate(items):
        return {
            "n": len(items),
            "recall@10": round(float(np.mean([r["recall@10"] for r in items])), 4),
            "recall@50": round(float(np.mean([r["recall@50"] for r in items])), 4),
            "hit@10": round(float(np.mean([r["hit@10"] for r in items])), 4),
            "hit@50": round(float(np.mean([r["hit@50"] for r in items])), 4),
            "mrr@10": round(float(np.mean([r["mrr@10"] for r in items])), 4),
        }

    result = {
        "model": str(model_path),
        "encode_seconds": round(time.time() - started, 2),
        "overall": aggregate(rows),
        "by_kind": {kind: aggregate(items) for kind, items in sorted(by_kind.items())},
        "per_query": rows,
    }
    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--finetuned-model", default=str(HERE / "model" / "wsb-minilm"))
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "index" / "eval_results.json"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-candidates", type=int, default=300)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    corpus = list(read_jsonl(data_dir / "corpus.jsonl"))
    queries = json.loads((data_dir / "eval_queries.json").read_text(encoding="utf-8"))
    texts = [row["text"] for row in corpus]
    post_ids = [row["post_id"] for row in corpus]

    results = {
        "corpus_chunks": len(corpus),
        "eval_queries": len(queries),
        "base": evaluate_model(args.base_model, texts, post_ids, queries,
                               args.device, max_candidates=args.max_candidates),
        "finetuned": evaluate_model(args.finetuned_model, texts, post_ids, queries,
                                    args.device, max_candidates=args.max_candidates),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    for name in ("base", "finetuned"):
        r = results[name]
        print(f"\n{name}: {r['overall']}")
        for kind, metrics in r["by_kind"].items():
            print(f"  {kind}: {metrics}")
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
